import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------------------------------------
# FastAPI setup
# -------------------------------------------------
app = FastAPI(
    title="Recipe Extractor",
    version="1.2.0",
    description=(
        "Extract a normalized recipe JSON (plus article body) from a public recipe URL. "
        "Now also returns article_sections (heading + paragraph pairs)."
    ),
    openapi_tags=[{"name": "Health"}, {"name": "Extract"}],
)

# -------------------------------------------------
# Request model
# -------------------------------------------------
class ExtractBody(BaseModel):
    url: str


# -------------------------------------------------
# HTTP fetch (robust headers)
# -------------------------------------------------
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

def fetch_html(url: str) -> str:
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {e}")


# -------------------------------------------------
# Time helpers (ISO8601 → human)
# -------------------------------------------------
ISO_DUR = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$",
    re.I,
)

def parse_iso8601_minutes(iso: Optional[str]) -> Optional[int]:
    if not iso:
        return None
    m = ISO_DUR.match(iso.strip())
    if not m:
        return None
    days = int(m.group("days") or 0)
    hours = int(m.group("hours") or 0)
    minutes = int(m.group("minutes") or 0)
    seconds = int(m.group("seconds") or 0)
    total_minutes = days * 24 * 60 + hours * 60 + minutes
    if total_minutes == 0 and seconds:
        total_minutes = 1
    return total_minutes or None

def humanize_minutes(total: Optional[int]) -> Optional[str]:
    if total is None:
        return None
    h, m = divmod(total, 60)
    parts: List[str] = []
    if h:
        parts.append(f"{h} hr" + ("" if h == 1 else "s"))
    if m:
        parts.append(f"{m} min" + ("" if m == 1 else "s"))
    if not parts:
        parts = ["0 mins"]
    return " ".join(parts)


# -------------------------------------------------
# Text cleaning helpers
# -------------------------------------------------
WS = re.compile(r"\s+")
GLUED_NUM = re.compile(r"([A-Za-z\)])(\d)")      # e.g., "flour1" → "flour 1"
LEADING_BULLET = re.compile(r"^[\-\u2022•\u00B7\*\u25CF]\s*")

def clean_text(s: str) -> str:
    s = s or ""
    s = WS.sub(" ", s).strip()
    s = LEADING_BULLET.sub("", s)
    s = GLUED_NUM.sub(r"\1 \2", s)
    return s

def final_trim(s: str) -> str:
    s = WS.sub(" ", s).strip()
    # trim trailing divider glyphs occasionally left by CMS
    return re.sub(r"\s*([•·/|])\s*$", "", s)

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        key = it.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


# -------------------------------------------------
# Article extraction (paragraphs + sections)
# -------------------------------------------------
_JUNK_CLASS_PATTERNS = re.compile(
    r"(related|promo|newsletter|ads?|subscribe|share|social|breadcrumb|"
    r"outbrain|taboola|sponsored|read-?more|recommended|"
    r"mntl-sc-block-callout|mntl-sc-block-inline-\w+)",
    re.I,
)
_JUNK_TEXT_PATTERNS = re.compile(
    r"(read more|watch|sponsored|shop now|subscribe|sign up|privacy policy|cookie|terms\b)",
    re.I,
)

def _is_junk_container(el: Tag) -> bool:
    cur = el
    while cur and isinstance(cur, Tag):
        classes = " ".join(cur.get("class", []))
        if _JUNK_CLASS_PATTERNS.search(classes or ""):
            return True
        cur = cur.parent
    return False

PARA_SELECTORS = [
    "p.mntl-sc-block-html",                 # Dotdash Meredith
    "article [itemprop='articleBody'] p",   # Schema.org
    "article .article-body p",
    "main .article-body p",
    "article .entry-content p",
    "main .entry-content p",
    "article p",
    "main p",
]

def _paragraph_candidates(soup: BeautifulSoup) -> List[Tag]:
    """Collect candidate <p> tags in DOM order, applying container & text filters."""
    seen = set()
    out: List[Tag] = []
    for sel in PARA_SELECTORS:
        for p in soup.select(sel):
            if not isinstance(p, Tag) or p.name != "p":
                continue
            if p in seen:
                continue
            seen.add(p)

            if _is_junk_container(p):
                continue

            text = p.get_text(" ", strip=True)
            if not text or _JUNK_TEXT_PATTERNS.search(text):
                continue

            # Skip mostly-link blocks ("Read more", lists of links, etc.)
            links = p.find_all("a")
            if links:
                link_text = "".join(a.get_text("", strip=True) for a in links)
                if len(link_text) > len(text) * 0.7:
                    continue

            # Skip tiny fragments that don't look like article copy
            if len(text) < 25 and not text.endswith("."):
                continue

            out.append(p)

    return out

def extract_article_paragraphs(soup: BeautifulSoup) -> List[str]:
    """Existing behavior: a flat list of cleaned article paragraphs."""
    paras: List[str] = []
    seen = set()
    for p in _paragraph_candidates(soup):
        cleaned = final_trim(p.get_text(" ", strip=True))
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        paras.append(cleaned)

    # Prefer sentence-like paragraphs; stable sort keeps DOM order on ties
    def score(t: str) -> int:
        sc = 0
        if t.endswith((".", "!", "?")):
            sc += 2
        if len(t) >= 60:
            sc += 1
        return sc

    paras.sort(key=score, reverse=True)
    return paras[:200]

def _closest_heading_for(p: Tag) -> Optional[str]:
    """Find the closest preceding <h2>/<h3> text for a paragraph."""
    # Fast path: nearest previous h2/h3 in document order
    h = p.find_previous(["h2", "h3"])
    if h and isinstance(h, Tag):
        txt = final_trim(h.get_text(" ", strip=True))
        if txt and not _JUNK_TEXT_PATTERNS.search(txt):
            return txt

    # If not found (rare), try climbing to parent containers and scanning
    cur = p.parent
    while cur and isinstance(cur, Tag):
        prev_heading = None
        # walk previous siblings backwards to find a heading
        sib = cur.previous_sibling
        while sib:
            if isinstance(sib, Tag):
                if sib.name in ("h2", "h3"):
                    prev_heading = sib
                    break
                # also scan inside sibling (some CMS wrap headings)
                inner = sib.find(["h2", "h3"])
                if inner:
                    prev_heading = inner
                    break
            sib = sib.previous_sibling
        if prev_heading:
            txt = final_trim(prev_heading.get_text(" ", strip=True))
            if txt and not _JUNK_TEXT_PATTERNS.search(txt):
                return txt
        cur = cur.parent

    return None

def extract_article_sections(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """
    NEW: Pair each kept article paragraph with its closest preceding H2/H3.
    Returns a list of { "heading": str, "paragraph": str }.
    """
    sections: List[Dict[str, str]] = []
    for p in _paragraph_candidates(soup):
        para_txt = final_trim(p.get_text(" ", strip=True))
        if not para_txt:
            continue
        heading_txt = _closest_heading_for(p) or ""
        sections.append({"heading": heading_txt, "paragraph": para_txt})
    return sections[:200]


# -------------------------------------------------
# JSON-LD Recipe extraction
# -------------------------------------------------
def first_or_none(x: Any) -> Any:
    if isinstance(x, list) and x:
        return x[0]
    return x

def extract_jsonld_recipe(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue

        candidates: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    candidates.append(item)
        elif isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                for item in data["@graph"]:
                    if isinstance(item, dict):
                        candidates.append(item)
            else:
                candidates.append(data)

        for item in candidates:
            types = item.get("@type")
            if isinstance(types, list):
                is_recipe = any(str(t).lower() == "recipe" for t in types)
            else:
                is_recipe = str(types).lower() == "recipe"
            if not is_recipe:
                continue

            recipe: Dict[str, Any] = {}
            recipe["name"] = item.get("name")
            recipe["description"] = item.get("description")

            author = item.get("author")
            if isinstance(author, list) and author:
                author = author[0]
            if isinstance(author, dict):
                recipe["author"] = author.get("name")
            else:
                recipe["author"] = author

            ing = item.get("recipeIngredient") or item.get("ingredients")
            if isinstance(ing, list):
                recipe["ingredients"] = [clean_text(str(i)) for i in ing if clean_text(str(i))]
            else:
                recipe["ingredients"] = []

            instr = item.get("recipeInstructions")
            steps: List[str] = []
            if isinstance(instr, list):
                for st in instr:
                    if isinstance(st, dict):
                        txt = st.get("text") or ""
                    else:
                        txt = str(st)
                    txt = clean_text(txt)
                    if txt:
                        steps.append(txt)
            elif isinstance(instr, str):
                for line in instr.splitlines():
                    line = clean_text(line)
                    if line:
                        steps.append(line)
            recipe["instructions"] = steps

            recipe["prepTime"]  = item.get("prepTime")
            recipe["cookTime"]  = item.get("cookTime")
            recipe["totalTime"] = item.get("totalTime")

            tags = item.get("keywords")
            if isinstance(tags, str):
                recipe["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
            elif isinstance(tags, list):
                recipe["tags"] = [clean_text(str(t)) for t in tags if clean_text(str(t))]
            else:
                recipe["tags"] = []

            nut = item.get("nutrition")
            recipe["nutrition"] = nut if isinstance(nut, dict) else {}

            return recipe
    return None


# -------------------------------------------------
# Fallback scraping for title/desc/ingredients/instructions
# -------------------------------------------------
def fallback_title(soup: BeautifulSoup) -> Optional[str]:
    h1 = soup.find("h1")
    if h1:
        return clean_text(h1.get_text(" ", strip=True))
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return clean_text(og["content"])
    return None

def fallback_description(soup: BeautifulSoup) -> Optional[str]:
    m = soup.find("meta", attrs={"name": "description"})
    if m and m.get("content"):
        return clean_text(m["content"])
    og = soup.find("meta", property="og:description")
    if og and og.get("content"):
        return clean_text(og["content"])
    return None

def collect_list_text(soup: BeautifulSoup, selectors: List[str]) -> List[str]:
    out: List[str] = []
    for sel in selectors:
        for li in soup.select(sel):
            txt = clean_text(li.get_text(" ", strip=True))
            if txt:
                out.append(txt)
        if out:
            break
    return dedupe_keep_order(out)

def fallback_ingredients(soup: BeautifulSoup) -> List[str]:
    selectors = [
        # Dotdash sites
        "ul.ingredients-section li, li.mntl-structured-ingredients__list-item",
        "[itemprop='recipeIngredient']",
        # Generic
        "ul.ingredients li",
        ".ingredients li",
    ]
    return collect_list_text(soup, selectors)

def fallback_instructions(soup: BeautifulSoup) -> List[str]:
    selectors = [
        # Dotdash sites
        "ol.instructions-section li, li.mntl-sc-block-group--LI",
        "[itemprop='recipeInstructions'] li",
        # Generic
        "ol li.instruction, ol li, .instructions li",
    ]
    return collect_list_text(soup, selectors)

def fallback_author(soup: BeautifulSoup) -> Optional[str]:
    cand = soup.select_one("[itemprop='author'], .byline__name, .mntl-attribution__item-name")
    if cand:
        return clean_text(cand.get_text(" ", strip=True))
    return None


# -------------------------------------------------
# Main extraction
# -------------------------------------------------
def parse_recipe(url: str, html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    # Prefer JSON-LD
    recipe = extract_jsonld_recipe(soup) or {}

    # Title/description/author fallbacks
    title = recipe.get("name") or fallback_title(soup) or ""
    description = recipe.get("description") or fallback_description(soup) or ""
    author = recipe.get("author") or fallback_author(soup) or ""

    # Ingredients/instructions fallbacks
    ingredients = recipe.get("ingredients") or fallback_ingredients(soup)
    instructions = recipe.get("instructions") or fallback_instructions(soup)

    # Ensure clean lines & no stuck numbers
    ingredients = [clean_text(i) for i in ingredients]
    instructions = [clean_text(s) for s in instructions]

    # Times (humanized)
    prep_iso  = recipe.get("prepTime")
    cook_iso  = recipe.get("cookTime")
    total_iso = recipe.get("totalTime")

    prep_h  = humanize_minutes(parse_iso8601_minutes(prep_iso))
    cook_h  = humanize_minutes(parse_iso8601_minutes(cook_iso))
    total_h = humanize_minutes(parse_iso8601_minutes(total_iso))

    times: Dict[str, Optional[str]] = {}
    if prep_h:  times["prepTime"]  = prep_h
    if cook_h:  times["cookTime"]  = cook_h
    if total_h: times["totalTime"] = total_h

    tags = recipe.get("tags") or []
    nutrition = recipe.get("nutrition") or {}

    # Article content
    article_paragraphs = extract_article_paragraphs(soup)      # existing flat list
    article_sections   = extract_article_sections(soup)        # NEW structured list

    result: Dict[str, Any] = {
        "schema_version": "1.2.0",
        "source_url": url,
        "title": title,
        "description": description,
        "ingredients": ingredients,
        "instructions": instructions,
        "times": times,
        "nutrition": nutrition if isinstance(nutrition, dict) else {},
        "tags": tags,
        "author": author,
        "article": article_paragraphs,     # unchanged
        "article_sections": article_sections,  # NEW
        "extracted_at": int(time.time()),
    }
    return result


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/", tags=["Health"])
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/extract", tags=["Extract"])
def extract(body: ExtractBody) -> Dict[str, Any]:
    url = body.url.strip()
    if not url.startswith("http"):
        raise HTTPException(status_code=422, detail="Invalid URL.")
    html = fetch_html(url)
    return parse_recipe(url, html)


# -------------------------------------------------
# Local dev entrypoint (optional)
# -------------------------------------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

