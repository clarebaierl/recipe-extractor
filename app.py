# app.py
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Union

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, field_validator

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    url: str


class AnalyzeRequest(BaseModel):
    """Optional second endpoint to run light analysis on an already extracted recipe."""
    ingredients: List[str]
    instructions: List[str]


class Recipe(BaseModel):
    schema_version: str = "1.0.0"
    source_url: str
    title: str
    description: Optional[str] = None
    ingredients: List[str]
    instructions: List[str]
    # Hardened: allow None and coerce lists/anything -> string (see validator)
    recipe_yield: Optional[str] = None
    times: Dict[str, Any] = {}
    nutrition: Dict[str, Any] = {}
    tags: Optional[List[str]] = None
    author: Optional[str] = None
    extracted_at: float

    # --- Hardening validators ---

    @field_validator("recipe_yield", mode="before")
    @classmethod
    def coerce_recipe_yield(cls, v):
        """
        Some sources serialize recipeYield as a list (e.g., ["4", "4 servings"]).
        Normalize to a single string; allow None.
        """
        if v is None:
            return None
        if isinstance(v, list):
            return " ".join(str(x) for x in v if x)
        return str(v)

    @field_validator("ingredients", "instructions", mode="before")
    @classmethod
    def ensure_list_of_strings(cls, v):
        """
        Normalize to list[str]. Accept a single string or a list of mixed types.
        """
        if v is None:
            return []
        if isinstance(v, (str, bytes)):
            return [str(v)]
        return [str(x) for x in v if x is not None]


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

app = FastAPI(title="Recipe Extractor", version="1.0.0")

# CORS for convenience (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

ISO8601_DURATION = re.compile(
    r"^P(?:(?P<years>\d+)Y)?(?:(?P<months>\d+)M)?(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$",
    re.IGNORECASE,
)


def parse_iso8601_duration_to_minutes(value: str) -> Optional[int]:
    """
    Convert ISO-8601 duration like 'PT1H30M' or 'P0DT45M' to total minutes.
    Returns None if it doesn't match.
    """
    if not value or not isinstance(value, str):
        return None
    m = ISO8601_DURATION.match(value.strip())
    if not m:
        return None

    parts = {k: int(v) if v else 0 for k, v in m.groupdict().items()}

    # Map years/months/weeks/days to minutes (rough approximations for recipes).
    minutes = (
        parts["years"] * 525600
        + parts["months"] * 43800
        + parts["weeks"] * 7 * 24 * 60
        + parts["days"] * 24 * 60
        + parts["hours"] * 60
        + parts["minutes"]
        + (parts["seconds"] // 60)
    )
    return minutes if minutes > 0 else None


def first_non_empty(*values: Any) -> Optional[str]:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value).strip()]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    # Fallback: coerce anything else to one string
    return [str(value).strip()]


def pick_recipe_from_jsonld(ld: Any) -> Optional[Dict[str, Any]]:
    """
    Given already loaded JSON-LD object (dict/list), try to find a Recipe node.
    """
    if isinstance(ld, dict):
        # @graph or a single node
        graph = ld.get("@graph")
        if isinstance(graph, list):
            for node in graph:
                if isinstance(node, dict) and node.get("@type") in ("Recipe", ["Recipe"]):
                    return node
        if ld.get("@type") in ("Recipe", ["Recipe"]):
            return ld
    elif isinstance(ld, list):
        for node in ld:
            if isinstance(node, dict) and node.get("@type") in ("Recipe", ["Recipe"]):
                return node
    return None


async def fetch_html(url: str, timeout: float = 20.0) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout, headers={
        "User-Agent": "Mozilla/5.0 (compatible; RecipeExtractor/1.0; +https://example.com)"
    }) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


def extract_from_html(html: str) -> Dict[str, Any]:
    """
    Minimal HTML fallback if JSON-LD is missing/partial.
    Best-effort: tries common classes/ids for ingredients & instructions.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title / description
    title = first_non_empty(
        getattr(soup.find("meta", property="og:title"), "get", lambda *_: None)("content", None),
        getattr(soup.find("title"), "text", None),
        getattr(soup.find("h1"), "text", None),
    )
    description = first_non_empty(
        getattr(soup.find("meta", property="og:description"), "get", lambda *_: None)("content", None),
        getattr(soup.find("meta", attrs={"name": "description"}), "get", lambda *_: None)("content", None),
    )

    # Very light-weight guesses
    ingredients = []
    for selector in ["li.ingredient", "li.ingredients-item", ".ingredients-section li", "li[itemprop='recipeIngredient']"]:
        for li in soup.select(selector):
            text = li.get_text(" ", strip=True)
            if text:
                ingredients.append(text)
        if ingredients:
            break

    instructions = []
    for selector in ["li.instruction", "li.instructions-section-item", ".instructions-section li", "li[itemprop='recipeInstructions']"]:
        for li in soup.select(selector):
            text = li.get_text(" ", strip=True)
            if text:
                instructions.append(text)
        if instructions:
            break

    return {
        "title": title or "",
        "description": description,
        "ingredients": ingredients,
        "instructions": instructions,
        "recipe_yield": None,
        "times": {},
        "nutrition": {},
        "tags": None,
        "author": None,
    }


def extract_from_jsonld(html: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON-LD and return a dict with normalized recipe fields if found.
    """
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", type=lambda t: t and "ld+json" in t)

    for s in scripts:
        try:
            data = json.loads(s.string or s.text or "")
        except Exception:
            continue

        recipe = pick_recipe_from_jsonld(data)
        if not recipe:
            continue

        # Title / description
        title = first_non_empty(
            recipe.get("name"),
            recipe.get("headline"),
        )
        description = recipe.get("description")

        # Ingredients (variously named)
        ingredients_raw = recipe.get("recipeIngredient") or recipe.get("ingredients")
        ingredients = normalize_list(ingredients_raw)

        # Instructions can be string, list of strings, or list of HowToStep dicts
        instructions_raw = recipe.get("recipeInstructions")
        instructions: List[str] = []
        if isinstance(instructions_raw, list):
            for item in instructions_raw:
                if isinstance(item, dict):
                    # HowToStep or HowToSection
                    txt = first_non_empty(item.get("text"), item.get("name"))
                    if txt:
                        instructions.append(txt)
                else:
                    txt = str(item).strip()
                    if txt:
                        instructions.append(txt)
        elif isinstance(instructions_raw, str):
            # split heuristically on newlines / periods if it looks like a paragraph
            parts = [p.strip() for p in re.split(r"[\n\r]+", instructions_raw) if p.strip()]
            instructions = parts if parts else [instructions_raw.strip()]

        # Yield
        recipe_yield = recipe.get("recipeYield")

        # Times
        times: Dict[str, Any] = {}
        # Many sites use ISO 8601 for totalTime, prepTime, cookTime, etc.
        for key, out in [
            ("totalTime", "total_minutes"),
            ("prepTime", "prep_minutes"),
            ("cookTime", "cook_minutes"),
            ("performTime", "perform_minutes"),
        ]:
            mins = parse_iso8601_duration_to_minutes(recipe.get(key))
            if mins is not None:
                times[out] = mins

        # Nutrition
        nutrition = {}
        if isinstance(recipe.get("nutrition"), dict):
            nutrition = {k: v for k, v in recipe["nutrition"].items() if v not in (None, "")}

        # Tags
        keywords = recipe.get("keywords")
        tags: Optional[List[str]] = None
        if isinstance(keywords, str):
            tags = [t.strip() for t in keywords.split(",") if t.strip()]
        elif isinstance(keywords, list):
            tags = [str(t).strip() for t in keywords if str(t).strip()]

        # Author
        author = None
        auth = recipe.get("author")
        if isinstance(auth, dict):
            author = first_non_empty(auth.get("name"))
        elif isinstance(auth, list) and auth:
            if isinstance(auth[0], dict):
                author = first_non_empty(auth[0].get("name"))
            else:
                author = str(auth[0]).strip()
        elif isinstance(auth, str):
            author = auth.strip()

        return {
            "title": title or "",
            "description": description,
            "ingredients": ingredients,
            "instructions": instructions,
            "recipe_yield": recipe_yield,
            "times": times,
            "nutrition": nutrition,
            "tags": tags,
            "author": author,
        }

    return None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "recipe-extractor"}


@app.post("/extract", response_model=Recipe)
async def extract(req: ExtractRequest) -> Recipe:
    """
    Extract a normalized recipe JSON from a public recipe URL.
    """
    try:
        html = await fetch_html(req.url)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {e}") from e

    # Prefer JSON-LD; fall back to light HTML heuristics
    data = extract_from_jsonld(html) or extract_from_html(html)

    # Build model with validation; convert validation errors to 422 for the client
    try:
        recipe = Recipe(
            schema_version="1.0.0",
            source_url=req.url,
            title=data["title"] or "",
            description=data.get("description"),
            ingredients=data.get("ingredients") or [],
            instructions=data.get("instructions") or [],
            recipe_yield=data.get("recipe_yield"),
            times=data.get("times") or {},
            nutrition=data.get("nutrition") or {},
            tags=data.get("tags"),
            author=data.get("author"),
            extracted_at=time.time(),
        )
    except ValidationError as e:
        # Expose which field failed (instead of a 500)
        raise HTTPException(status_code=422, detail=e.errors())

    return recipe


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    """
    Tiny example analyzer: counts ingredients, steps, and estimates complexity.
    """
    ing_count = len([i for i in req.ingredients if i.strip()])
    step_count = len([s for s in req.instructions if s.strip()])
    complexity = "easy"
    if ing_count >= 12 or step_count >= 10:
        complexity = "hard"
    elif ing_count >= 7 or step_count >= 6:
        complexity = "medium"

    return {
        "ingredients_count": ing_count,
        "steps_count": step_count,
        "estimated_complexity": complexity,
    }
