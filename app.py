import re
import json
import time
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup, Tag
from dateutil import parser as dateparser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Recipe Extractor", version="1.2.0")

# --- HTTP ---

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)

def fetch_html(url: str) -> str:
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
            },
            timeout=20,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {e}")
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch URL: {resp.status_code} Client Error:  for url: {url}",
        )
    return resp.text


# --- Helpers: time & text cleanup ---

ISO_DUR_RE = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$", re.I)

def humanize_iso8601_duration(value: Optional[str]) -> Optional[str]:
    """
    Turn ISO 8601 durations (e.g., PT6H20M, PT45M) into human strings ('6 hrs 20 mins', '45 mins').
    Returns None if input is None/empty/unparseable.
    """
    if not value or not isinstance(value, str):
        return None
    m = ISO_DUR_RE.match(value.strip())
    if not m:
        return value  # return original if already human or unexpected
    days, hours, mins, secs = m.groups(default="0")
    d, h, m_, s = int(days), int(hours), int(mins), int(secs)
    total_mins = d * 24 * 60 + h * 60 + m_ + (1 if s >= 30 else 0)

    if total_mins < 60:
        return f"{total_mins} mins"

    hrs = total_mins // 60
    mins_rem = total_mins % 60
    if mins_rem == 0:
        return f"{hrs} hrs"
    return f"{hrs} hrs {mins_rem} mins"


BAD_INLINE_PHRASES = re.compile(
    r"\b(read more|shop now|watch now|sponsored|subscribe|sign up)\b",
    re.I,
)

def clean_text(t: str) -> str:
    if not t:
        return ""
    # Collapse whitespace and strip
    txt = re.sub(r"\s+", " ", t).strip()
    # Remove tracking ellipses/boilerplate
    txt = re.sub(r"•", "-", txt)
    return txt


def node_is_ad_or_rail(node: Tag) -> bool:
    """
    Heuristics to skip ad/promotional/rail/related link blocks common on MNTL/Condé Nast/etc.
    """
    classes = " ".join((node.get("class") or []))
    id_ = node.get("id") or ""
    role = node.get("role") or ""
    data_component = node.get("data-component") or ""

    bad_fragments = (
        "ad", "promo", "rail", "social", "newsletter",
        "subscribe", "related-links", "toc", "jump", "recirc",
        "sponsor", "affiliate", "embed", "cookie", "footer",
        "sidebar", "gallery", "slideshow"
    )
    haystack = " ".join([classes, id_, role, data_component]).lower()
    return any(b in haystack for b in bad_fragments)


# --- JSON-LD (recipe) ---

def find_json_ld_blocks(soup: BeautifulSoup) -> List[dict]:
    blocks: List[dict] = []
    for node in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(node.string or "")
            if isinstance(data, dict):
                blocks.append(data)
            elif isinstance(data, list):
                blocks.extend([d for d in data if isinstance(d, dict)])
        except Exception:
            # Skip invalid JSON-LD blocks
            continue
    return blocks


def pick_best_recipe_block(blocks: List[dict]) -> Optional[dict]:
    """
    Choose a recipe-ish JSON-LD block (Recipe or with @graph containing Recipe).
    """
    for b in blocks:
        t = (b.get("@type") or b.get("@context") or "")
        if isinstance(t, list):
            t = " ".join(t)
        t = str(t).lower()
        if "recipe" in t:
            return b

    # @graph style
    for b in blocks:
        graph = b.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                t = str(item.get("@type", "")).lower()
                if "recipe" in t:
                    return item
    return None


def extract_from_recipe_ld(recipe_ld: dict) -> Dict[str, Any]:
    """
    Extract normalized fields from a Recipe JSON-LD object.
    """
    title = recipe_ld.get("name") or ""
    description = recipe_ld.get("description") or ""

    # Ingredients and instructions come in various shapes; normalize to lists of strings
    ingredients: List[str] = []
    raw_ing = recipe_ld.get("recipeIngredient")
    if isinstance(raw_ing, list):
        ingredients = [clean_text(x) for x in raw_ing if isinstance(x, str)]

    instructions: List[str] = []
    raw_inst = recipe_ld.get("recipeInstructions")
    if isinstance(raw_inst, list):
        for step in raw_inst:
            if isinstance(step, str):
                s = clean_text(step)
                if s:
                    instructions.append(s)
            elif isinstance(step, dict):
                s = clean_text(step.get("text", ""))
                if s:
                    instructions.append(s)
    elif isinstance(raw_inst, str):
        s = clean_text(raw_inst)
        if s:
            # Split on sentence-ish boundaries if it looks like a blob:
            parts = [clean_text(p) for p in re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9])", s) if clean_text(p)]
            instructions = parts or [s]

    # Times (human friendly)
    prep = humanize_iso8601_duration(recipe_ld.get("prepTime"))
    cook = humanize_iso8601_duration(recipe_ld.get("cookTime"))
    total = humanize_iso8601_duration(recipe_ld.get("totalTime"))

    # Nutrition (pass through if present)
    nutrition = recipe_ld.get("nutrition") or {}

    author = ""
    a = recipe_ld.get("author")
    if isinstance(a, dict):
        author = a.get("name", "") or ""
    elif isinstance(a, list) and a:
        if isinstance(a[0], dict):
            author = a[0].get("name", "") or ""
        elif isinstance(a[0], str):
            author = a[0]
    elif isinstance(a, str):
        author = a

    tags: List[str] = []
    kw = recipe_ld.get("keywords")
    if isinstance(kw, str):
        tags = [clean_text(k) for k in re.split(r"[;,]", kw) if clean_text(k)]
    elif isinstance(kw, list):
        tags = [clean_text(k) for k in kw if isinstance(k, str) and clean_text(k)]

    return {
        "title": clean_text(title),
        "description": clean_text(description),
        "ingredients": ingredients,
        "instructions": instructions,
        "times": {
            "prepTime": prep,
            "cookTime": cook,
            "totalTime": total,
        },
        "nutrition": nutrition if isinstance(nutrition, dict) else {},
        "tags": tags,
        "author": clean_text(author),
    }


# --- Article extraction (MNTL-first with generic fallbacks) ---

def select_article_stream(soup: BeautifulSoup) -> List[Tag]:
    """
    Return a DOM-ordered stream of heading & paragraph blocks that represent the *article body*
    (not the recipe card, not related links/ad slots).
    Priority:
      1) MNTL block classes (common to Allrecipes/SimplyRecipes/etc.)
      2) schema.org Article [itemprop=articleBody]
      3) Generic <article> container
    """
    # 1) MNTL block pattern
    mntl_candidates = soup.select(
        ".mntl-sc-block-heading, .mntl-sc-block-html, .mntl-sc-block-p, .mntl-sc-block-callout"
    )
    if mntl_candidates:
        # filter out junk by ancestor signals and inline “read more” etc.
        stream = []
        for node in mntl_candidates:
            if any(node_is_ad_or_rail(anc) for anc in node.parents if isinstance(anc, Tag)):
                continue
            # avoid obvious “related/read more” callouts
            txt = clean_text(node.get_text(" ", strip=True))
            if not txt:
                continue
            if BAD_INLINE_PHRASES.search(txt):
                continue
            stream.append(node)
        if stream:
            return stream

    # 2) schema.org article body wrapper
    article_body = soup.select_one('[itemprop="articleBody"]')
    if isinstance(article_body, Tag):
        els = article_body.find_all(["h2", "h3", "p"], recursive=True)
        stream = []
        for el in els:
            if any(node_is_ad_or_rail(anc) for anc in el.parents if isinstance(anc, Tag)):
                continue
            txt = clean_text(el.get_text(" ", strip=True))
            if not txt or BAD_INLINE_PHRASES.search(txt):
                continue
            stream.append(el)
        if stream:
            return stream

    # 3) Generic <article>
    art = soup.find("article")
    if isinstance(art, Tag):
        els = art.find_all(["h2", "h3", "p"], recursive=True)
        stream = []
        for el in els:
            if any(node_is_ad_or_rail(anc) for anc in el.parents if isinstance(anc, Tag)):
                continue
            txt = clean_text(el.get_text(" ", strip=True))
            if not txt or BAD_INLINE_PHRASES.search(txt):
                continue
            stream.append(el)
        if stream:
            return stream

    return []


def build_article_payload(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Produce:
      - article: flat paragraph list
      - article_sections: [{heading, paragraphs: []}, ...]
    Heading rules:
      * MNTL: any .mntl-sc-block-heading text (inner span text if present)
      * Generic: any H2/H3 text
      * Paragraphs (MNTL .mntl-sc-block-html/.mntl-sc-block-p or generic <p>)
      * Skip tiny boilerplate lines and link-only bits
    """
    stream = select_article_stream(soup)
    if not stream:
        return {"article": [], "article_sections": []}

    flat_paras: List[str] = []
    sections: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {"heading": None, "paragraphs": []}

    def push_current():
        nonlocal current
        if current["heading"] or current["paragraphs"]:
            sections.append(
                {"heading": current["heading"], "paragraphs": current["paragraphs"][:]}
            )
        current = {"heading": None, "paragraphs": []}

    for node in stream:
        # Identify type + extract text
        is_heading = False
        if "mntl-sc-block-heading" in (node.get("class") or []):
            is_heading = True
            # pull inner span if present
            span = node.select_one(".mntl-sc-block-heading__text")
            text = clean_text(span.get_text(" ", strip=True) if span else node.get_text(" ", strip=True))
        elif node.name in ("h2", "h3"):
            is_heading = True
            text = clean_text(node.get_text(" ", strip=True))
        else:
            # paragraph-like
            text = clean_text(node.get_text(" ", strip=True))

        # Drop very short / junk paragraphs
        if not text or BAD_INLINE_PHRASES.search(text):
            continue
        if not is_heading and len(text) < 25 and not text.endswith((".", "!", "?")):
            # likely a caption/crumb
            continue

        if is_heading:
            # start new section
            push_current()
            current["heading"] = text
        else:
            flat_paras.append(text)
            current["paragraphs"].append(text)

    push_current()
    return {"article": flat_paras, "article_sections": sections}


# --- Request/Response models ---

class ExtractRequest(BaseModel):
    url: str


# --- Routes ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
def extract(req: ExtractRequest) -> Dict[str, Any]:
    html = fetch_html(req.url)
    soup = BeautifulSoup(html, "lxml")

    # JSON-LD recipe extraction
    blocks = find_json_ld_blocks(soup)
    recipe_ld = pick_best_recipe_block(blocks)
    if not recipe_ld:
        raise HTTPException(status_code=422, detail="Could not find recipe schema on page.")

    recipe = extract_from_recipe_ld(recipe_ld)

    # Article extraction (non-breaking optional fields)
    art_payload = build_article_payload(soup)

    out: Dict[str, Any] = {
        "schema_version": "1.2.0",
        "source_url": req.url,
        **recipe,  # title, description, ingredients, instructions, times, nutrition, tags, author
        "article": art_payload["article"],
        "article_sections": art_payload["article_sections"],
        "extracted_at": int(time.time()),
    }

    return out
