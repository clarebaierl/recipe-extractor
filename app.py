# app.py — works with Python 3.9 + Pydantic v2

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import httpx, json, re, time
from typing import Any, List, Dict, Optional

app = FastAPI(title="Recipe Extractor", version="1.0.0")

# --------- Models ---------

class ExtractRequest(BaseModel):
    # Pydantic v2: use pattern= (not regex=)
    url: str = Field(..., pattern=r"^https?://")

class Recipe(BaseModel):
    schema_version: str = "1.0.0"
    source_url: str
    title: str
    description: Optional[str] = None
    ingredients: List[str]
    instructions: List[str]
    recipe_yield: Optional[str] = None
    times: Optional[Dict] = None          # e.g., {"prep_minutes": 10, "cook_minutes": 50}
    nutrition: Optional[Dict] = None
    tags: Optional[List[str]] = None
    author: Optional[str] = None
    extracted_at: float

# --------- Helpers ---------

ISO_DUR_RE = re.compile(r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?)?$", re.I)

def iso_to_minutes(val: Optional[str]) -> Optional[int]:
    """Convert ISO-8601 durations (e.g., PT15M, PT1H, P1DT30M) to total minutes."""
    if not val or not isinstance(val, str):
        return None
    m = ISO_DUR_RE.match(val.strip())
    if not m:
        return None
    days = int(m.group("days") or 0)
    hours = int(m.group("h") or 0)
    mins = int(m.group("m") or 0)
    return days * 24 * 60 + hours * 60 + mins

def flatten_instructions(instr: Any) -> List[str]:
    """
    Normalize recipeInstructions into a flat list[str].
    Handles:
      - list[str]
      - list[HowToStep/HowToSection]
      - single string
    """
    out: List[str] = []
    if isinstance(instr, list):
        for item in instr:
            if isinstance(item, dict):
                t = item.get("@type")
                if t in ("HowToStep", "HowToDirection") and "text" in item:
                    out.append(str(item["text"]))
                elif t == "HowToSection" and "itemListElement" in item:
                    out.extend(flatten_instructions(item["itemListElement"]))
            elif isinstance(item, str):
                out.append(item)
    elif isinstance(instr, dict):
        if instr.get("@type") == "HowToSection":
            out.extend(flatten_instructions(instr.get("itemListElement")))
        elif "text" in instr:
            out.append(str(instr["text"]))
    elif isinstance(instr, str):
        out.append(instr)

    return [re.sub(r"\s+", " ", s).strip() for s in out if s and s.strip()]

def pick_recipe_object(payload: Any) -> Optional[Dict]:
    """Among JSON-LD objects, return the one whose @type includes 'Recipe'."""
    def is_recipe(obj: Dict) -> bool:
        t = obj.get("@type")
        if isinstance(t, list):
            return any(str(x).lower() == "recipe" for x in t)
        return str(t).lower() == "recipe"

    if isinstance(payload, list):
        for obj in payload:
            if isinstance(obj, dict) and is_recipe(obj):
                return obj
    elif isinstance(payload, dict) and is_recipe(payload):
        return payload
    return None

def normalize_author(author_val: Any) -> Optional[str]:
    if author_val is None:
        return None
    if isinstance(author_val, dict):
        return author_val.get("name")
    if isinstance(author_val, list):
        names: List[str] = []
        for a in author_val:
            if isinstance(a, dict):
                n = a.get("name")
                if n:
                    names.append(str(n))
            else:
                names.append(str(a))
        return ", ".join([n for n in names if n]) or None
    return str(author_val)

# --------- Endpoints ---------

@app.post("/extract", response_model=Recipe)
async def extract(req: ExtractRequest):
    # 1) Fetch the HTML
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            r = await client.get(req.url, headers={"User-Agent": "RecipeExtractor/1.0"})
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch page: {e}")

    # 2) Find JSON-LD blocks
    soup = BeautifulSoup(r.text, "html.parser")
    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    recipe_obj: Optional[Dict] = None
    for tag in scripts:
        raw = tag.string
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        recipe_obj = pick_recipe_object(data)
        if recipe_obj:
            break

    if not recipe_obj:
        raise HTTPException(422, "No schema.org Recipe JSON-LD found on this page.")

    # 3) Normalize fields
    title = recipe_obj.get("name") or recipe_obj.get("headline") or ""
    description = recipe_obj.get("description")

    ingredients = recipe_obj.get("recipeIngredient") or []
    ingredients = [re.sub(r"\s+", " ", str(i)).strip() for i in ingredients]

    instructions = flatten_instructions(recipe_obj.get("recipeInstructions"))
    if not ingredients or not instructions:
        raise HTTPException(422, "Recipe JSON-LD missing ingredients or instructions.")

    times = {
        "prep_minutes": iso_to_minutes(recipe_obj.get("prepTime")),
        "cook_minutes": iso_to_minutes(recipe_obj.get("cookTime")),
        "total_minutes": iso_to_minutes(recipe_obj.get("totalTime")),
    }
    times = {k: v for k, v in times.items() if v is not None} or None

    nutrition = recipe_obj.get("nutrition")
    tags = recipe_obj.get("keywords")
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    author = normalize_author(recipe_obj.get("author"))

    # 4) Return canonical object
    return Recipe(
        source_url=req.url,
        title=title,
        description=description,
        ingredients=ingredients,
        instructions=instructions,
        recipe_yield=recipe_obj.get("recipeYield"),
        times=times,
        nutrition=nutrition,
        tags=tags,
        author=author,
        extracted_at=time.time(),
    )

class AnalyzeRequest(BaseModel):
    url: str
    task: str  # e.g., "seo" | "edits" | "summary"

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Optional wrapper to plug your LLM later.
    For now, it just returns the extracted recipe with the requested task.
    """
    recipe = await extract(ExtractRequest(url=req.url))
    return {"task": req.task, "source_url": recipe.source_url, "recipe": recipe.dict()}

