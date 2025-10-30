import os
import re
import time
from datetime import datetime, timezone
from typing import List, Optional, Union

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

APP_NAME = "Recipe Extractor"
APP_VERSION = "1.0.0"

# -----------------------------
# Models
# -----------------------------

class Recipe(BaseModel):
    schema_version: str = Field(default="1.0.0")
    source_url: str
    title: Optional[str] = None
    description: Optional[str] = None
    ingredients: List[str] = Field(default_factory=list)
    instructions: List[str] = Field(default_factory=list)
    recipe_yield: Optional[str] = None
    times: dict = Field(default_factory=dict)  # {"prepTime": "...", "cookTime": "...", "totalTime": "..."}
    nutrition: dict = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    extracted_at: int = Field(default_factory=lambda: int(time.time()))

    @validator("ingredients", "instructions", pre=True)
    def coerce_to_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            # If we ever get a single big string, split on lines that look like distinct items.
            tentative = [s.strip() for s in re.split(r"\n|\r|\t|\u2028|\u2029", v) if s.strip()]
            return tentative if len(tentative) > 1 else [v.strip()]
        if isinstance(v, list):
            # strip empties
            return [str(x).strip() for x in v if str(x).strip()]
        return []

class ExtractRequest(BaseModel):
    url: str

# -----------------------------
# App setup
# -----------------------------

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# CORS (relaxed for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple, in-memory rate limiter (per IP)
RATE_WINDOW_SEC = 60
RATE_MAX_REQUESTS = 60
_ip_hits = {}

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    try:
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        bucket = _ip_hits.setdefault(ip, [])
        # drop old
        while bucket and now - bucket[0] > RATE_WINDOW_SEC:
            bucket.pop(0)
        if len(bucket) >= RATE_MAX_REQUESTS:
            return fastapi_json({"detail": "Too Many Requests"}, 429)
        bucket.append(now)
        return await call_next(request)
    except Exception:
        # never block the request on limiter failure
        return await call_next(request)

def fastapi_json(payload: dict, status_code: int = 200):
    from fastapi.responses import JSONResponse
    return JSONResponse(content=payload, status_code=status_code)

# -----------------------------
# Utility helpers
# -----------------------------

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)

HTTP_TIMEOUT = 20.0

async def fetch_html(url: str) -> str:
    if not re.match(r"^https?://", url, flags=re.I):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml"}
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=HTTP_TIMEOUT, headers=headers) as client:
            r = await client.get(url)
            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"Upstream error {r.status_code}")
            ctype = r.headers.get("content-type", "")
            if "text/html" not in ctype and "application/json" not in ctype:
                # Some sites send JSON-LD with JSON content-type; allow both
                raise HTTPException(status_code=415, detail="Unsupported Content-Type from source")
            return r.text
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timed out fetching source page")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Network error: {e}")

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def iso8601_to_friendly(s: Optional[str]) -> Optional[str]:
    """
    Convert ISO 8601 durations like PT20M / PT6H / PT6H20M / P1DT30M to '20 min', '6 hr', '1 day 30 min', etc.
    If s not ISO-8601, return it unchanged.
    """
    if not s or not isinstance(s, str):
        return s
    m = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?",
        s.strip(),
        flags=re.I,
    )
    if not m:
        return s  # not ISO-8601, pass through

    days = int(m.group("days") or 0)
    hours = int(m.group("hours") or 0)
    minutes = int(m.group("minutes") or 0)
    seconds = int(m.group("seconds") or 0)

    parts = []
    if days:
        parts.append(f"{days} day" + ("s" if days != 1 else ""))
    if hours:
        parts.append(f"{hours} hr" + ("s" if hours != 1 else ""))
    if minutes:
        parts.append(f"{minutes} min" + ("s" if minutes != 1 else ""))
    if seconds and not (days or hours or minutes):
        parts.append(f"{seconds} sec" + ("s" if seconds != 1 else ""))

    return " ".join(parts) if parts else "0 min"

def ensure_list(obj) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [clean_text(str(x)) for x in obj if clean_text(str(x))]
    if isinstance(obj, str):
        return [clean_text(obj)]
    return []

def extract_json_ld(soup: BeautifulSoup) -> Optional[dict]:
    # Gather all application/ld+json blocks and look for @type Recipe
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            data = json.loads(tag.string or tag.text or "")
        except Exception:
            continue
        # JSON-LD can be a list or a single object
        candidates = data if isinstance(data, list) else [data]
        # Also can be wrapped in "@graph"
        if isinstance(data, dict) and "@graph" in data:
            candidates.extend(data["@graph"])
        for node in candidates:
            try:
                t = node.get("@type")
                if isinstance(t, list):
                    t = [x.lower() for x in t]
                    if "recipe" in t:
                        return node
                elif isinstance(t, str) and t.lower() == "recipe":
                    return node
            except Exception:
                continue
    return None

def select_list_items(soup: BeautifulSoup, selectors: List[str]) -> List[str]:
    """
    Try several CSS selectors; return the first non-empty list of <li> texts.
    Ensures each LI becomes a separate element, keeping spaces so we don't get 'flour1' joining.
    """
    for sel in selectors:
        nodes = soup.select(sel)
        # If the selector hit list containers, flatten their <li> children.
        items = []
        for node in nodes:
            if node.name in ("ul", "ol"):
                lis = node.find_all("li")
                items.extend([clean_text(li.get_text(" ")) for li in lis])
            elif node.name == "li":
                items.append(clean_text(node.get_text(" ")))
        items = [i for i in items if i]
        if items:
            return items
    return []

def extract_fallback(soup: BeautifulSoup) -> Recipe:
    # Title
    title = None
    if soup.title:
        title = clean_text(soup.title.get_text(" "))

    # Ingredients
    ingredient_selectors = [
        '[itemprop="recipeIngredient"] li',
        '[itemprop="ingredients"] li',
        'ul[class*="ingredient"] li',
        'ul.ingredients li',
        'li.ingredient',
        # as a container:
        '[itemprop="recipeIngredient"]',
        '.ingredients li',
    ]
    ingredients = select_list_items(soup, ingredient_selectors)

    # Instructions
    instruction_selectors = [
        '[itemprop="recipeInstructions"] li',
        'ol[class*="instruction"] li',
        'ol.instructions li',
        '.instructions li',
        'li.instruction',
        # Sometimes instructions are in paragraphs:
        '[itemprop="recipeInstructions"] p',
        '.instructions p',
    ]
    instructions = select_list_items(soup, instruction_selectors)
    if not instructions:
        # Some sites put HowToStep content as blocks; gather paragraphs within the overall container
        container = soup.select_one('[itemprop="recipeInstructions"], .instructions')
        if container:
            paras = [clean_text(p.get_text(" ")) for p in container.find_all("p")]
            instructions = [p for p in paras if p]

    return Recipe(
        source_url="",
        title=title,
        ingredients=ingredients,
        instructions=instructions,
    )

def normalize_from_jsonld(node: dict) -> Recipe:
    # Title / desc / author
    title = clean_text(node.get("name") or node.get("headline") or "")
    description = clean_text(node.get("description") or "")
    author = None
    if node.get("author"):
        a = node["author"]
        if isinstance(a, list) and a:
            a = a[0]
        if isinstance(a, dict) and a.get("name"):
            author = clean_text(a["name"])
        elif isinstance(a, str):
            author = clean_text(a)

    # Ingredients
    ing = node.get("recipeIngredient") or node.get("ingredients")
    ingredients = ensure_list(ing)

    # Instructions can be list of HowToStep dicts, list of strings, or one big string
    inst = node.get("recipeInstructions")
    instructions: List[str] = []
    if isinstance(inst, list):
        for step in inst:
            if isinstance(step, dict):
                txt = step.get("text") or step.get("name") or ""
                txt = clean_text(txt)
                if txt:
                    instructions.append(txt)
            elif isinstance(step, str):
                txt = clean_text(step)
                if txt:
                    instructions.append(txt)
    elif isinstance(inst, str):
        # Split on sentences or new lines
        bits = [clean_text(s) for s in re.split(r"\n+|\r+|(?<=\.)\s+(?=[A-Z])", inst) if clean_text(s)]
        instructions.extend(bits)

    # Times
    def friendly(key):
        raw = node.get(key)
        if isinstance(raw, list) and raw:
            raw = raw[0]
        raw = clean_text(raw) if isinstance(raw, str) else raw
        return iso8601_to_friendly(raw)  # convert if ISO-8601, otherwise pass through

    times = {}
    for key in ("prepTime", "cookTime", "totalTime"):
        val = friendly(key)
        if val:
            times[key] = val

    # Yield
    ry = node.get("recipeYield")
    if isinstance(ry, list):
        ry = ry[0] if ry else None
    recipe_yield = clean_text(str(ry)) if ry else None

    # Nutrition
    nutrition = {}
    if isinstance(node.get("nutrition"), dict):
        for k, v in node["nutrition"].items():
            nutrition[k] = clean_text(str(v))

    # Tags (keywords)
    tags = []
    kw = node.get("keywords")
    if isinstance(kw, str):
        tags = [clean_text(k) for k in kw.split(",") if clean_text(k)]
    elif isinstance(kw, list):
        tags = [clean_text(k) for k in kw if clean_text(k)]

    return Recipe(
        source_url="",
        title=title or None,
        description=description or None,
        ingredients=ingredients,
        instructions=instructions,
        recipe_yield=recipe_yield or None,
        times=times,
        nutrition=nutrition,
        tags=tags,
        author=author or None,
    )

# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def health():
    return {
        "ok": True,
        "name": APP_NAME,
        "version": APP_VERSION,
        "now": datetime.now(timezone.utc).isoformat(),
    }

@app.post("/extract")
async def extract(req: ExtractRequest):
    html = await fetch_html(req.url)
    soup = BeautifulSoup(html, "lxml")  # needs 'lxml' installed

    # Prefer JSON-LD Recipe if present
    node = extract_json_ld(soup)
    if node:
        recipe = normalize_from_jsonld(node)
    else:
        recipe = extract_fallback(soup)

    # Always ensure list items are one-per-element, not a single collapsed blob
    recipe.ingredients = [clean_text(x) for x in recipe.ingredients]
    recipe.instructions = [clean_text(x) for x in recipe.instructions]

    # Add source & timestamp last
    recipe.source_url = req.url
    recipe.extracted_at = int(time.time())

    # If we still have nothing meaningful, return 422 to signal “couldn’t parse”
    if not (recipe.title or recipe.ingredients or recipe.instructions):
        raise HTTPException(status_code=422, detail="Could not extract a recipe from this URL")

    return recipe.dict()

