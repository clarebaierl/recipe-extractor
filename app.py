# app.py
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Recipe Extractor API")

# ---------- Models ----------

class ExtractRequest(BaseModel):
    url: str

# ---------- HTTP fetch ----------

_DEFAULT_HEADERS = {
    # Pretend to be a normal browser
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

def fetch_html(url: str) -> str:
    try:
        resp = requests.get(url, headers=_DEFAULT_HEADERS, timeout=18)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        raise HTTPException(status_code=460, detail=f"Failed to fetch URL: {e}")

# ---------- Small utils ----------

_ws_re = re.compile(r"\s+")

def squash_ws(s: str) -> str:
    return _ws_re.sub(" ", s or "").strip()

def to_human_duration(iso_str: Optional[str]) -> str:
    """
    Convert ISO8601 durations like PT1H30M / PT45M to '1 hr 30 mins' / '45 mins'.
    If iso_str missing, return "".
    """
    if not iso_str or not isinstance(iso_str, str):
        return ""
    # Very tolerant parse
    hours = minutes = 0
    m = re.search(r"PT(?:(\d+)H)?(?:(\d+)M)?", iso_str.upper())
    if m:
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
    parts = []
    if hours:
        parts.append(f"{hours} hr" + ("s" if hours != 1 else ""))
    if minutes:
        parts.append(f"{minutes} min" + ("s" if minutes != 1 else ""))
    return " ".join(parts) if parts else ""

# Fix “Cabernet Sauvignon3 ribs … taste2 tablespoons …” style gluing.
def deglue_numbers(text: str) -> str:
    if not text:
        return text
    # insert a space when a letter is immediately followed by a digit
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    # insert a space when a digit is immediately followed by a letter (rare)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    # collapse spaces
    return squash_ws(text)

def clean_lines(items: List[str]) -> List[str]:
    out = []
    for item in items or []:
        t = deglue_numbers(squash_ws(item))
        if t:
            out.append(t)
    return out

# ---------- JSON-LD Recipe extraction ----------

def _pick_recipe_block(ld: Any) -> Optional[Dict[str, Any]]:
    """Find the Recipe object inside any @graph / array."""
    def is_recipe(obj: Any) -> bool:
        t = obj.get("@type") if isinstance(obj, dict) else None
        if isinstance(t, list):
            return any(x == "Recipe" for x in t)
        return t == "Recipe"

    if isinstance(ld, dict):
        if is_recipe(ld):
            return ld
        if "@graph" in ld and isinstance(ld["@graph"], list):
            for node in ld["@graph"]:
                if isinstance(node, dict) and is_recipe(node):
                    return node
    if isinstance(ld, list):
        for node in ld:
            if isinstance(node, dict):
                r = _pick_recipe_block(node)
                if r:
                    return r
    return None

def extract_recipe_from_ldjson(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue
        recipe = _pick_recipe_block(data)
        if recipe:
            return recipe
    return None

# ---------- Heuristic DOM fallback for recipe (rarely used) ----------

def fallback_extract_recipe(soup: BeautifulSoup) -> Dict[str, Any]:
    # Extremely light fallback; sites we’re targeting usually have JSON-LD.
    # We attempt to pull common selectors if LD+JSON missing.
    title = ""
    t = soup.find("h1")
    if t:
        title = squash_ws(t.get_text(" "))

    ingredients = [squash_ws(li.get_text(" ")) for li in soup.select("[itemprop='recipeIngredient'], .ingredients li, .ingredient")]
    instructions = [squash_ws(li.get_text(" ")) for li in soup.select("[itemprop='recipeInstructions'] li, .instructions li, .direction")]

    return {
        "title": title,
        "description": "",
        "ingredients": clean_lines(ingredients),
        "instructions": clean_lines(instructions),
        "times": {"prepTime": "", "cookTime": "", "totalTime": ""},
        "nutrition": {},
        "tags": [],
        "author": "",
    }

# ---------- Article section extraction (grouped by heading) ----------

_SKIP_PHRASES = (
    "read more",
    "learn more",
    "watch:",
)
_SKIP_CLASSES = (
    "heading-toc",           # table of contents anchors
    "mntl-sc-block-ads",     # ad containers
    "comp ads",              # generic ads
)

def _seems_trash(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return True
    # one-liners that are just a “Learn more:” prefix or similar
    for phrase in _SKIP_PHRASES:
        if phrase in t:
            return True
    return False

def _has_skip_class(el: Tag) -> bool:
    cls = " ".join(el.get("class", [])).lower()
    return any(x in cls for x in _SKIP_CLASSES)

def _main_content_node(soup: BeautifulSoup) -> Tag:
    """
    Try to find the main article body container across SimplyRecipes/Allrecipes/Meredith (Mntl) sites.
    Fallback to <body>.
    """
    candidates = [
        # Meredith stack
        "[data-sc-sticky-offset]",  # big page wrapper
        "article",
        "#article",
        ".article-body",
        "#content",
        "main",
    ]
    for sel in candidates:
        node = soup.select_one(sel)
        if node:
            return node
    return soup.body or soup

def extract_grouped_article_sections(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """
    Produces:
      [{ "heading": "What Is Mochi?", "paragraphs": ["...", "..."] },
       { "heading": "How to Store Mochi", "paragraphs": ["..."] }]
    Also captures any intro paragraph(s) before the first heading under heading=None.
    """
    root = _main_content_node(soup)

    # Find all headings and meaningful paragraphs in DOM order within the root
    stream: List[Tuple[str, Tag]] = []
    for el in root.find_all(["h2", "h3", "p"], recursive=True):
        # Skip obvious junk
        if _has_skip_class(el):
            continue
        if el.name in ("h2", "h3"):
            txt = squash_ws(el.get_text(" "))
            if not _seems_trash(txt):
                stream.append(("heading", el))
        elif el.name == "p":
            # reject paras that are basically just links/buttons/etc.
            txt = squash_ws(el.get_text(" "))
            if not _seems_trash(txt):
                stream.append(("p", el))

    sections: List[Dict[str, Any]] = []
    current_heading: Optional[str] = None
    current_paras: List[str] = []

    def flush():
        nonlocal current_heading, current_paras
        if current_paras:
            sections.append(
                {"heading": current_heading if current_heading else None, "paragraphs": current_paras}
            )
        current_heading = None
        current_paras = []

    for kind, node in stream:
        if kind == "heading":
            # new heading starts: flush previous group
            flush()
            current_heading = squash_ws(node.get_text(" "))
        else:
            txt = squash_ws(node.get_text(" "))
            if txt:
                current_paras.append(txt)

    # flush the tail
    flush()

    # Deduplicate repeated blocks & merge identical consecutive headings if any
    merged: List[Dict[str, Any]] = []
    for sec in sections:
        if merged and merged[-1]["heading"] == sec["heading"]:
            merged[-1]["paragraphs"].extend(sec["paragraphs"])
        else:
            merged.append(sec)

    # Final cleanliness pass (remove any paragraphs that are actually CTA/footers)
    for sec in merged:
        sec["paragraphs"] = [p for p in sec["paragraphs"] if not _seems_trash(p)]
    # Remove sections that ended up empty
    merged = [s for s in merged if s["paragraphs"]]

    return merged

# ---------- Recipe normalization (schema preserved) ----------

def normalize_recipe(ld: Dict[str, Any]) -> Dict[str, Any]:
    title = squash_ws(ld.get("name") or ld.get("headline") or "")
    description = squash_ws(ld.get("description") or "")
    author = ""
    if isinstance(ld.get("author"), dict):
        author = squash_ws(ld["author"].get("name", ""))
    elif isinstance(ld.get("author"), list) and ld["author"]:
        # pick the first author name if list
        item = ld["author"][0]
        if isinstance(item, dict):
            author = squash_ws(item.get("name", ""))
        else:
            author = squash_ws(str(item))

    # ingredients
    ingredients = []
    ing = ld.get("recipeIngredient")
    if isinstance(ing, list):
        ingredients = clean_lines([str(x) for x in ing])

    # instructions
    instructions: List[str] = []
    inst = ld.get("recipeInstructions")
    if isinstance(inst, list):
        for step in inst:
            if isinstance(step, dict):
                # can be HowToStep or HowToSection
                if "text" in step:
                    instructions.append(step["text"])
                elif "itemListElement" in step and isinstance(step["itemListElement"], list):
                    for sub in step["itemListElement"]:
                        if isinstance(sub, dict) and "text" in sub:
                            instructions.append(sub["text"])
                        elif isinstance(sub, str):
                            instructions.append(sub)
            elif isinstance(step, str):
                instructions.append(step)
    elif isinstance(inst, str):
        # sometimes it's a big blob separated by newlines
        for line in inst.split("\n"):
            t = squash_ws(line)
            if t:
                instructions.append(t)

    instructions = clean_lines(instructions)

    # times to human
    times = {
        "prepTime": to_human_duration(ld.get("prepTime")),
        "cookTime": to_human_duration(ld.get("cookTime")),
        "totalTime": to_human_duration(ld.get("totalTime")),
    }

    # nutrition passthrough (keep original keys if present)
    nutrition = {}
    if isinstance(ld.get("nutrition"), dict):
        nutrition = {k: ld["nutrition"][k] for k in ld["nutrition"]}

    # tags (keywords or recipeCategory/cuisine)
    tags: List[str] = []
    for key in ("keywords", "recipeCategory", "recipeCuisine"):
        v = ld.get(key)
        if isinstance(v, list):
            tags.extend([squash_ws(str(x)) for x in v if squash_ws(str(x))])
        elif isinstance(v, str):
            # split keywords by comma if needed
            parts = [squash_ws(x) for x in v.split(",")]
            tags.extend([x for x in parts if x])
    # normalize tags uniqueness
    uniq = []
    seen = set()
    for t in tags:
        if t.lower() not in seen:
            uniq.append(t)
            seen.add(t.lower())

    return {
        "title": title,
        "description": description,
        "ingredients": ingredients,
        "instructions": instructions,
        "times": times,
        "nutrition": nutrition,
        "tags": uniq,
        "author": author,
    }

# ---------- Route ----------

@app.post("/extract")
def extract(req: ExtractRequest):
    url = req.url.strip()
    if not url.startswith("http"):
        raise HTTPException(status_code=422, detail="URL must start with http or https")

    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    # 1) Recipe (JSON-LD first; fallback keeps original schema intact)
    ld_recipe = extract_recipe_from_ldjson(soup)
    if ld_recipe:
        recipe = normalize_recipe(ld_recipe)
    else:
        recipe = fallback_extract_recipe(soup)

    # 2) Article sections (group headings with paragraphs; no duplicates)
    article_sections = extract_grouped_article_sections(soup)

    # Build response (schema preserved; only new field is article_sections)
    out = {
        "schema_version": "1.4.0",
        "source_url": url,
        # ---- Recipe block (UNCHANGED keys) ----
        "title": recipe["title"],
        "description": recipe["description"],
        "ingredients": recipe["ingredients"],
        "instructions": recipe["instructions"],
        "times": recipe["times"],
        "nutrition": recipe["nutrition"],
        "tags": recipe["tags"],
        "author": recipe["author"],
        # ---- New field ----
        "article_sections": article_sections,
        # timestamp
        "extracted_at": int(time.time()),
    }

    return out
