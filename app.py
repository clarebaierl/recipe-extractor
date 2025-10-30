from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl, ValidationError

# BeautifulSoup with safe parser fallback
from bs4 import BeautifulSoup, FeatureNotFound

APP_TITLE = "Recipe Extractor"
APP_DESC = "Extract a normalized recipe JSON from a public recipe URL."
APP_VERSION = "1.0.0"

app = FastAPI(title=APP_TITLE, description=APP_DESC, version=APP_VERSION)

# CORS (adjust origins for production if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you like
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------

class ExtractRequest(BaseModel):
    url: HttpUrl


class RecipeTimes(BaseModel):
    prepTime: Optional[str] = None
    cookTime: Optional[str] = None
    totalTime: Optional[str] = None


class Recipe(BaseModel):
    schema_version: str = Field(default="1.0.0")
    source_url: HttpUrl
    title: Optional[str] = None
    description: Optional[str] = None
    ingredients: List[str] = Field(default_factory=list)
    instructions: List[str] = Field(default_factory=list)
    recipe_yield: Optional[str] = None
    times: RecipeTimes = Field(default_factory=RecipeTimes)
    nutrition: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    extracted_at: int = Field(default_factory=lambda: int(time.time()))

# ---------- Utilities ----------

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
)
COMMON_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def pick_parser() -> str:
    """
    Prefer lxml if installed; else fall back to html.parser (never throw).
    """
    try:
        # This only checks availability; BeautifulSoup will use it if present.
        import lxml  # noqa: F401
        return "lxml"
    except Exception:
        return "html.parser"

def bs(html: str) -> BeautifulSoup:
    parser = pick_parser()
    try:
        return BeautifulSoup(html, parser)
    except FeatureNotFound:
        # Absolute fallback if lxml was requested but missing at runtime
        return BeautifulSoup(html, "html.parser")


def _ensure_list(obj: Union[str, Dict[str, Any], List[Any], None]) -> List[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def _textify_instruction(step: Any) -> Optional[str]:
    """
    Normalize a recipe instruction step that may be:
    - string
    - dict {"@type":"HowToStep","text":"..."} or {"text":"..."}
    - dict {"@type":"HowToSection","name":"...","itemListElement":[...]}
    """
    if step is None:
        return None
    if isinstance(step, str):
        s = step.strip()
        return s or None
    if isinstance(step, dict):
        # HowToStep
        if "text" in step and isinstance(step["text"], str):
            s = step["text"].strip()
            return s or None
        # HowToSection (flatten its items)
        if "itemListElement" in step and isinstance(step["itemListElement"], list):
            parts: List[str] = []
            for sub in step["itemListElement"]:
                t = _textify_instruction(sub)
                if t:
                    parts.append(t)
            if parts:
                return " ".join(parts)
    return None


def _from_graph(obj: Any) -> List[Dict[str, Any]]:
    """
    Given a parsed JSON-LD object, return all candidate nodes (handles @graph).
    """
    candidates: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if "@graph" in obj and isinstance(obj["@graph"], list):
            candidates.extend([x for x in obj["@graph"] if isinstance(x, dict)])
        else:
            candidates.append(obj)
    elif isinstance(obj, list):
        candidates.extend([x for x in obj if isinstance(x, dict)])
    return candidates


def _is_recipe(node: Dict[str, Any]) -> bool:
    t = node.get("@type") or node.get("type")
    if t is None:
        return False
    if isinstance(t, str):
        return t.lower() == "recipe"
    if isinstance(t, list):
        return any(isinstance(x, str) and x.lower() == "recipe" for x in t)
    return False


def parse_json_ld(html: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON-LD blocks and return the first node that looks like a Recipe.
    """
    soup = bs(html)
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or tag.get_text() or "")
        except Exception:
            continue
        for node in _from_graph(data):
            if _is_recipe(node):
                return node
    return None


def normalize_recipe(node: Dict[str, Any], url: str) -> Recipe:
    # Title / description
    title = node.get("name") or node.get("headline")
    description = node.get("description")

    # Ingredients (usually a list of strings)
    raw_ingredients = node.get("recipeIngredient")
    ingredients: List[str] = []
    for ing in _ensure_list(raw_ingredients):
        if isinstance(ing, str):
            s = ing.strip()
            if s:
                ingredients.append(s)

    # Instructions: list of strings OR list of HowToStep/HowToSection
    raw_instructions = node.get("recipeInstructions")
    instructions: List[str] = []
    for step in _ensure_list(raw_instructions):
        t = _textify_instruction(step)
        if t:
            instructions.append(t)

    # Yield (string/number)
    ry = node.get("recipeYield")
    if isinstance(ry, (int, float)):
        recipe_yield = str(ry)
    elif isinstance(ry, list):
        recipe_yield = ", ".join(str(x) for x in ry if x)
    elif isinstance(ry, str):
        recipe_yield = ry.strip() or None
    else:
        recipe_yield = None

    # Times (ISO 8601 duration strings typically)
    times = RecipeTimes(
        prepTime=node.get("prepTime"),
        cookTime=node.get("cookTime"),
        totalTime=node.get("totalTime"),
    )

    # Nutrition (flatten if present)
    nutrition = {}
    if isinstance(node.get("nutrition"), dict):
        for k, v in node["nutrition"].items():
            if v not in (None, "", []):
                nutrition[k] = v

    # Tags / keywords / categories
    tags: List[str] = []
    for key in ("keywords", "recipeCategory", "recipeCuisine"):
        val = node.get(key)
        if isinstance(val, str):
            # keywords often comma-separated
            parts = [x.strip() for x in val.split(",") if x.strip()]
            tags.extend(parts)
        elif isinstance(val, list):
            tags.extend([str(x).strip() for x in val if str(x).strip()])

    # Author (string or dict or list)
    author = None
    auth = node.get("author")
    if isinstance(auth, str):
        author = auth.strip() or None
    elif isinstance(auth, dict):
        author = (auth.get("name") or "").strip() or None
    elif isinstance(auth, list):
        # take first non-empty name
        for a in auth:
            if isinstance(a, str) and a.strip():
                author = a.strip()
                break
            if isinstance(a, dict) and isinstance(a.get("name"), str) and a["name"].strip():
                author = a["name"].strip()
                break

    r = Recipe(
        source_url=url,
        title=title,
        description=description,
        ingredients=ingredients,
        instructions=instructions,
        recipe_yield=recipe_yield,
        times=times,
        nutrition=nutrition,
        tags=list(dict.fromkeys(tags)),  # de-dup preserving order
        author=author,
    )
    return r


def microdata_fallback(html: str, url: str) -> Optional[Recipe]:
    """
    Extremely light fallback: try to grab obvious ingredient/instruction lists
    when JSON-LD is missing (best-effort).
    """
    soup = bs(html)

    title = None
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)

    # ingredients
    ingredients: List[str] = []
    # try itemprop or class names
    for sel in [
        '[itemprop="recipeIngredient"]',
        ".ingredients li",
        ".ingredient, .ingredients__item, li.ingredient",
    ]:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) < 300:
                ingredients.append(txt)
        if ingredients:
            break

    # instructions
    instructions: List[str] = []
    for sel in [
        '[itemprop="recipeInstructions"] li',
        ".instructions li",
        ".direction, .directions__item, li.instruction",
        "ol li, .recipe-instructions li",
    ]:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) < 600:
                instructions.append(txt)
        if instructions:
            break

    if not title and not ingredients and not instructions:
        return None

    return Recipe(
        source_url=url,
        title=title,
        ingredients=ingredients,
        instructions=instructions,
    )

# ---------- Routes ----------

@app.get("/", tags=["Health"])
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": APP_TITLE, "version": APP_VERSION}


@app.post("/extract", response_model=Recipe, tags=["Extract"])
async def extract(req: ExtractRequest) -> Recipe:
    url = str(req.url)

    # Fetch page
    try:
        async with httpx.AsyncClient(
            headers=COMMON_HEADERS,
            follow_redirects=True,
            timeout=httpx.Timeout(15.0, connect=10.0, read=15.0),
        ) as client:
            resp = await client.get(url)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Network error: {e}") from e

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=f"Upstream returned {resp.status_code}")

    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type:
        raise HTTPException(status_code=415, detail=f"Unsupported content-type: {content_type}")

    html = resp.text

    # First, try JSON-LD
    node = parse_json_ld(html)
    recipe: Optional[Recipe] = None
    if node:
        try:
            recipe = normalize_recipe(node, url)
        except ValidationError as e:
            # If something in JSON-LD is malformed, we still try fallbacks
            recipe = None

    # Fallback: simple HTML/microdata scrape
    if not recipe:
        recipe = microdata_fallback(html, url)

    if not recipe:
        raise HTTPException(status_code=422, detail="Could not extract a recipe from this page")

    return recipe
