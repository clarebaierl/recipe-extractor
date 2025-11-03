import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Recipe Extractor", version="1.3.0")

# A realistic desktop UA keeps some sites from blocking the request
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# -----------------------------
# Request model
# -----------------------------
class ExtractRequest(BaseModel):
    url: str


# -----------------------------
# Helpers
# -----------------------------
def fetch_html(url: str) -> str:
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {resp.status_code} for url: {url}")
    return resp.text


def load_json_ld(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Return list of JSON-LD dictionaries on the page."""
    out: List[Dict[str, Any]] = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            # Some pages embed multiple JSON blocks in one tag
            data = json.loads(tag.string or tag.text or "")
            if isinstance(data, list):
                out.extend(d for d in data if isinstance(d, dict))
            elif isinstance(data, dict):
                out.append(data)
        except Exception:
            continue
    return out


def find_first_recipe(json_ld_blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the first schema.org Recipe object in JSON-LD (handles @graph)."""
    for block in json_ld_blocks:
        # direct
        if str(block.get("@type")).lower() == "recipe":
            return block
        # @graph
        graph = block.get("@graph")
        if isinstance(graph, list):
            for node in graph:
                if isinstance(node, dict) and str(node.get("@type")).lower() == "recipe":
                    return node
    return None


# ---- text utilities ----
WS_RE = re.compile(r"\s+")
LEARN_MORE_RE = re.compile(r"^(learn more|read more|related|watch|see also)\b", re.I)


def clean_text(s: str) -> str:
    s = WS_RE.sub(" ", s or "").strip()
    return s


def strip_inline_refs(s: str) -> str:
    """Remove inline reference markers like 'Cabernet Sauvignon3' or 'carrots4'."""
    # Replace a letter-run followed by an inline digit with just the letters.
    # (avoid killing times like '6 hrs', fractions like '1/2', etc.)
    return re.sub(r"([A-Za-z])(\d+)(\b)", r"\1", s)


def list_safely(x: Any) -> List[Any]:
    return x if isinstance(x, list) else ([x] if x not in (None, "") else [])


# -----------------------------
# Recipe extraction (JSON-LD first)
# -----------------------------
def parse_times(recipe: Dict[str, Any]) -> Dict[str, str]:
    def pick(*keys: str) -> Optional[str]:
        for k in keys:
            v = recipe.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    # Prefer total, then compute if missing
    prep = pick("prepTime", "prep_time")
    cook = pick("cookTime", "cook_time")
    total = pick("totalTime", "total_time")
    return {"prepTime": prep or "", "cookTime": cook or "", "totalTime": total or ""}


def parse_nutrition(recipe: Dict[str, Any]) -> Dict[str, Any]:
    nutrition = recipe.get("nutrition") or {}
    if isinstance(nutrition, dict):
        return nutrition
    return {}


def parse_instructions(recipe: Dict[str, Any]) -> List[str]:
    instructions = recipe.get("recipeInstructions") or recipe.get("instructions")
    steps: List[str] = []

    if isinstance(instructions, str):
        # split on sentences / bullets
        for part in re.split(r"(?:\n+|\. (?=[A-Z(]))", instructions):
            part = clean_text(part)
            if part:
                steps.append(part if part.endswith(".") else f"{part}.")
        return steps

    if isinstance(instructions, list):
        for item in instructions:
            if isinstance(item, str):
                part = clean_text(item)
                if part:
                    steps.append(part)
            elif isinstance(item, dict):
                # HowToStep, HowToSection, etc.
                txt = item.get("text") or item.get("name") or ""
                txt = clean_text(txt)
                if txt:
                    steps.append(txt)
    return steps


def parse_ingredients(recipe: Dict[str, Any]) -> List[str]:
    ings = list_safely(recipe.get("recipeIngredient") or recipe.get("ingredients"))
    out: List[str] = []
    for ing in ings:
        if isinstance(ing, str):
            out.append(clean_text(ing))
    return out


# -----------------------------
# Article extraction (MNTL blocks + generic fallback)
# -----------------------------
def extract_mntl_article(soup: BeautifulSoup) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Extract paragraphs and {heading, paragraph} pairs from MNTL-based sites
    (Allrecipes, Simply Recipes, etc.). We respect document order and skip
    ads/related/read-more blocks.
    """
    article: List[str] = []
    sections: List[Dict[str, str]] = []

    # MNTL pages render story blocks as a linear stream of nodes with ids like
    # mntl-sc-block_XX-0. Paragraphs use <p class="... mntl-sc-block-html">,
    # headings use <h2 class="...-sc-block-heading"> or <h3 class="...-subheading">.
    # We'll iterate all such P/H2/H3 in DOM order.
    blocks = soup.select(
        'p.mntl-sc-block, h2.mntl-sc-block, h3.mntl-sc-block, '
        'p[class*="mntl-sc-block-html"], '
        'h2[class*="sc-block-heading"], h3[class*="sc-block-heading"], '
        'h2[class*="sc-block-subheading"], h3[class*="sc-block-subheading"]'
    )
    if not blocks:
        return ([], [])

    current_heading: Optional[str] = None

    def is_bad_container(tag: Tag) -> bool:
        clazz = " ".join(tag.get("class", [])).lower()
        if any(x in clazz for x in ["adslot", "ads", "promo", "related", "gallery", "newsletter"]):
            return True
        if tag.name in ("aside", "nav", "footer"):
            return True
        return False

    def in_bad_ancestor(tag: Tag) -> bool:
        p = tag.parent
        while isinstance(p, Tag):
            if is_bad_container(p):
                return True
            p = p.parent
        return False

    for el in blocks:
        if in_bad_ancestor(el):
            continue

        name = el.name.lower()

        # Grab the *visual* heading text (MNTL sometimes nests text in <span>)
        if name in ("h2", "h3"):
            htxt = clean_text(el.get_text(" ", strip=True))
            htxt = strip_inline_refs(htxt)
            if not htxt:
                continue
            current_heading = htxt
            # We do not append headings to `article`; they’ll appear in `article_sections`
            continue

        if name == "p":
            txt = clean_text(el.get_text(" ", strip=True))
            if not txt:
                continue
            # Filter "Learn more / Read more" and similar non-body lines
            if LEARN_MORE_RE.match(txt):
                continue
            # Kill inline numeric reference noise (e.g., "Cabernet Sauvignon3")
            txt = strip_inline_refs(txt)

            article.append(txt)
            sections.append({"heading": current_heading, "paragraph": txt})

    return (article, sections)


def extract_generic_article(soup: BeautifulSoup) -> Tuple[List[str], List[Dict[str, str]]]:
    """A safe generic fallback if the MNTL heuristic finds nothing."""
    article: List[str] = []
    sections: List[Dict[str, str]] = []

    main = soup.find("article") or soup.find("main") or soup
    current_heading: Optional[str] = None

    for node in main.descendants:
        if not isinstance(node, Tag):
            continue
        if node.name in ("script", "style", "noscript"):
            continue
        cls = " ".join(node.get("class", [])).lower()
        if any(bad in cls for bad in ["ad", "promo", "related", "newsletter", "social", "toc"]):
            continue

        if node.name in ("h2", "h3"):
            current_heading = clean_text(node.get_text(" ", strip=True))
            current_heading = strip_inline_refs(current_heading)
            continue

        if node.name == "p":
            txt = clean_text(node.get_text(" ", strip=True))
            if not txt or LEARN_MORE_RE.match(txt):
                continue
            txt = strip_inline_refs(txt)
            article.append(txt)
            sections.append({"heading": current_heading, "paragraph": txt})

    return (article, sections)


def extract_article_fields(soup: BeautifulSoup) -> Tuple[List[str], List[Dict[str, str]]]:
    # Try MNTL first
    article, sections = extract_mntl_article(soup)
    if article:
        return article, sections
    # Fallback
    return extract_generic_article(soup)


# -----------------------------
# Main extraction orchestrator
# -----------------------------
def build_payload(url: str, soup: BeautifulSoup) -> Dict[str, Any]:
    json_ld = load_json_ld(soup)
    recipe_ld = find_first_recipe(json_ld)

    if not recipe_ld:
        # If no recipe JSON-LD, return just article parts with minimal shell
        article, article_sections = extract_article_fields(soup)
        return {
            "schema_version": "1.3.0",
            "source_url": url,
            "title": soup.title.string.strip() if soup.title else "",
            "description": "",
            "ingredients": [],
            "instructions": [],
            "times": {"prepTime": "", "cookTime": "", "totalTime": ""},
            "nutrition": {},
            "tags": [],
            "author": "",
            "article": article,
            "article_sections": article_sections,
            "extracted_at": int(time.time()),
        }

    # --- Recipe fields ---
    title = recipe_ld.get("name") or recipe_ld.get("headline") or (soup.title.string.strip() if soup.title else "")
    description = clean_text(recipe_ld.get("description") or "")

    ingredients = parse_ingredients(recipe_ld)
    instructions = parse_instructions(recipe_ld)
    times = parse_times(recipe_ld)
    nutrition = parse_nutrition(recipe_ld)

    # Author
    author = ""
    author_field = recipe_ld.get("author")
    if isinstance(author_field, dict):
        author = author_field.get("name") or ""
    elif isinstance(author_field, list):
        for a in author_field:
            if isinstance(a, dict) and a.get("name"):
                author = a["name"]
                break
            if isinstance(a, str) and a.strip():
                author = a.strip()
                break
    elif isinstance(author_field, str):
        author = author_field

    # Tags
    keywords = list_safely(recipe_ld.get("keywords"))
    if len(keywords) == 1 and isinstance(keywords[0], str) and "," in keywords[0]:
        keywords = [clean_text(x) for x in keywords[0].split(",")]
    tags = [clean_text(k) for k in keywords if clean_text(k)]

    # --- Article fields (new) ---
    article, article_sections = extract_article_fields(soup)

    return {
        "schema_version": "1.3.0",
        "source_url": url,
        "title": clean_text(title),
        "description": clean_text(description),
        "ingredients": [strip_inline_refs(i) for i in ingredients],
        "instructions": [strip_inline_refs(s) for s in instructions],
        "times": times,
        "nutrition": nutrition,
        "tags": tags,
        "author": author,
        "article": article,  # ordered paragraphs
        "article_sections": article_sections,  # [{heading, paragraph}] in order
        "extracted_at": int(time.time()),
    }


# -----------------------------
# Route
# -----------------------------
@app.post("/extract")
def extract(req: ExtractRequest) -> Dict[str, Any]:
    html = fetch_html(req.url)
    soup = BeautifulSoup(html, "lxml")
    return build_payload(req.url, soup)

