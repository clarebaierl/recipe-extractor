import json
import re
import time
from typing import List, Optional, Dict, Any

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Recipe Extractor",
    version="1.0.0",
    description="Extract a normalized recipe JSON from a public recipe URL.",
)

# ---- Models -----------------------------------------------------------------

class Times(BaseModel):
    prepTime: Optional[str] = None   # human friendly e.g. "20 mins" or "6 hrs 20 mins"
    cookTime: Optional[str] = None
    totalTime: Optional[str] = None

class Recipe(BaseModel):
    schema_version: str = Field(default="1.0.0")
    source_url: str
    title: Optional[str] = None
    description: Optional[str] = None
    ingredients: List[str] = Field(default_factory=list)
    instructions: List[str] = Field(default_factory=list)
    times: Times = Field(default_factory=Times)
    nutrition: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    extracted_at: int

class ExtractRequest(BaseModel):
    url: str

# ---- Helpers ----------------------------------------------------------------

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Connection": "close"
}

SESSION = requests.Session()
SESSION.headers.update(BROWSER_HEADERS)
SESSION.timeout = 20  # default per-request timeout

ISO_DUR_RE = re.compile(
    r"^P(?:(?P<years>\d+)Y)?(?:(?P<months>\d+)M)?(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$",
    re.IGNORECASE,
)

def iso8601_to_human(iso: Optional[str]) -> Optional[str]:
    if not iso:
        return None
    m = ISO_DUR_RE.match(iso.strip())
    if not m:
        return None
    parts = {k: int(v) for k, v in m.groupdict(default="0").items()}
    minutes_total = (
        parts["years"] * 525600
        + parts["months"] * 43800
        + parts["weeks"] * 10080
        + parts["days"] * 1440
        + parts["hours"] * 60
        + parts["minutes"]
        + (1 if parts["seconds"] and int(parts["seconds"]) > 0 else 0)
    )
    if minutes_total <= 0:
        return None
    hours, mins = divmod(minutes_total, 60)
    if hours == 0:
        return f"{mins} mins"
    if mins == 0:
        return f"{hours} hr" if hours == 1 else f"{hours} hrs"
    return f"{hours} hr {mins} mins" if hours == 1 else f"{hours} hrs {mins} mins"

def fetch_html(url: str) -> str:
    # Small retry to dodge transient 403/460s
    last_exc = None
    for i in range(2):
        try:
            resp = SESSION.get(url, timeout=20)
            if resp.status_code in (403, 429, 460):
                # brief pause and retry once with slightly different headers
                time.sleep(0.8)
                SESSION.headers["Accept-Language"] = "en-US,en;q=0.8"
                resp = SESSION.get(url, timeout=20)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            last_exc = e
            time.sleep(0.5)
    raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {last_exc}")

def coerce_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        # flatten HowToStep-like objects or nested
        out = []
        for item in v:
            if isinstance(item, dict) and "text" in item:
                text = (item.get("text") or "").strip()
                if text:
                    out.append(text)
            elif isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
        return out
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    return []

def normalize_text_items(items: List[str]) -> List[str]:
    # Remove accidental numbering/gluing, trim, collapse whitespace
    out = []
    for raw in items:
        s = re.sub(r"\s+", " ", raw).strip()
        # Sometimes publishers prefix items with step numbers or bullets; drop “1)”, “1.”, “- ”
        s = re.sub(r"^\s*(?:[-•]\s*|\d+[\).\s]+)\s*", "", s)
        if s:
            out.append(s)
    return out

def parse_json_ld(soup: BeautifulSoup) -> Dict[str, Any]:
    # Find all <script type="application/ld+json"> blocks and merge/choose the Recipe one
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or tag.text or "")
        except Exception:
            continue
        # JSON-LD can be an array or a dict with @graph
        candidates = []
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                candidates = data["@graph"]
            else:
                candidates = [data]
        for node in candidates:
            t = node.get("@type")
            # Could be ["Recipe", ...] or a single string
            if (isinstance(t, str) and t.lower() == "recipe") or (
                isinstance(t, list) and any(str(x).lower() == "recipe" for x in t)
            ):
                return node
    return {}

def extract_from_html_fallback(soup: BeautifulSoup) -> Dict[str, Any]:
    # Best-effort fallback: lists under common containers
    ingredients = []
    instructions = []
    # Ingredients: look for UL/OL with ingredient-ish hints
    for ul in soup.find_all(["ul", "ol"]):
        hint = (ul.get("class") or []) + [ul.get("id") or ""]
        hint_text = " ".join(hint).lower()
        if "ingredient" in hint_text:
            for li in ul.find_all("li"):
                txt = li.get_text(" ", strip=True)
                if txt:
                    ingredients.append(txt)
    # Instructions (steps)
    for ol in soup.find_all(["ol", "ul"]):
        hint = (ol.get("class") or []) + [ol.get("id") or ""]
        hint_text = " ".join(hint).lower()
        if "instruction" in hint_text or "direction" in hint_text or "method" in hint_text or "step" in hint_text:
            for li in ol.find_all("li"):
                txt = li.get_text(" ", strip=True)
                if txt:
                    instructions.append(txt)

    # If still empty, be conservative rather than grabbing raw paragraphs (which can glue numbers)
    return {
        "ingredients": normalize_text_items(ingredients),
        "instructions": normalize_text_items(instructions),
    }

def build_recipe(url: str, html: str) -> Recipe:
    soup = BeautifulSoup(html, "html.parser")

    data = parse_json_ld(soup)

    # Title & description
    title = data.get("name") or soup.title.string.strip() if soup.title else None
    description = (data.get("description") or "").strip() or None

    # Ingredients & instructions (prefer JSON-LD)
    ingredients = coerce_list(data.get("recipeIngredient"))
    instructions_raw = data.get("recipeInstructions")
    instructions = coerce_list(instructions_raw)

    # If JSON-LD missing, fallback to HTML list extraction
    if not ingredients or not instructions:
        fb = extract_from_html_fallback(soup)
        if not ingredients:
            ingredients = fb["ingredients"]
        if not instructions:
            instructions = fb["instructions"]

    ingredients = normalize_text_items(ingredients)
    instructions = normalize_text_items(instructions)

    # Times
    times = Times(
        prepTime=iso8601_to_human(data.get("prepTime")),
        cookTime=iso8601_to_human(data.get("cookTime")),
        totalTime=iso8601_to_human(data.get("totalTime")),
    )

    # Nutrition (JSON-LD often has it as nested object)
    nutrition = {}
    if isinstance(data.get("nutrition"), dict):
        nutrition = {k: v for k, v in data["nutrition"].items() if v}

    # Tags and author if present
    tags = []
    if isinstance(data.get("recipeCategory"), list):
        tags += [str(x) for x in data["recipeCategory"] if x]
    elif isinstance(data.get("recipeCategory"), str):
        tags.append(data["recipeCategory"])
    if isinstance(data.get("recipeCuisine"), list):
        tags += [str(x) for x in data["recipeCuisine"] if x]
    elif isinstance(data.get("recipeCuisine"), str):
        tags.append(data["recipeCuisine"])

    author = None
    a = data.get("author")
    if isinstance(a, dict):
        author = a.get("name")
    elif isinstance(a, list) and a and isinstance(a[0], dict):
        author = a[0].get("name")
    elif isinstance(a, str):
        author = a

    return Recipe(
        source_url=url,
        title=title,
        description=description,
        ingredients=ingredients,
        instructions=instructions,
        times=times,
        nutrition=nutrition,
        tags=tags,
        author=author,
        extracted_at=int(time.time() * 1000),
    )

# ---- Routes -----------------------------------------------------------------

@app.get("/", tags=["Health"])
def health():
    return {"ok": True}

@app.post("/extract", response_model=Recipe, tags=["Extract"])
def extract(req: ExtractRequest = Body(...)):
    html = fetch_html(req.url)
    return build_recipe(req.url, html)

