# app.py
import re
import json
import time
from typing import List, Optional, Any, Dict, Union

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, AnyHttpUrl, Field

# ---------------------------
# Models
# ---------------------------

class Recipe(BaseModel):
    schema_version: str = "1.0.0"
    source_url: AnyHttpUrl
    title: Optional[str] = None
    description: Optional[str] = None
    ingredients: List[str] = []
    instructions: List[str] = []
    recipe_yield: Optional[str] = None
    times: Dict[str, Union[str, int]] = {}
    nutrition: Dict[str, Union[str, float, int]] = {}
    tags: Optional[List[str]] = None
    author: Optional[str] = None
    extracted_at: float = Field(default_factory=lambda: time.time())

class ExtractRequest(BaseModel):
    url: AnyHttpUrl

class AnalyzeRequest(BaseModel):
    text: str

# ---------------------------
# FastAPI
# ---------------------------

app = FastAPI(
    title="Recipe Extractor",
    version="1.0.0",
    description="Extract a normalized recipe JSON from a public recipe URL."
)

ALLOWED_SCHEMES = {"http", "https"}
DEFAULT_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.9",
}

# ---------------------------
# Helpers
# ---------------------------

def _clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    # strip unicode bullets and odd whitespace
    t = t.lstrip("•·-–—").strip()
    return t

def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _first(*vals):
    for v in vals:
        if v:
            return v
    return None

async def fetch_html(url: str, timeout: float = 12.0) -> str:
    # Scheme guard
    if url.split(":", 1)[0].lower() not in ALLOWED_SCHEMES:
        raise HTTPException(status_code=400, detail="Only http/https URLs are allowed.")
    try:
        async with httpx.AsyncClient(follow_redirects=True, headers=DEFAULT_HEADERS, timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            # Some CDNs serve weird encodings; httpx will decode, but be defensive:
            content = resp.text or resp.content.decode(resp.encoding or "utf-8", errors="ignore")
            return content
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream fetch failed: {e!s}")

def _pick_recipe_from_ld(ld: Any) -> Optional[Dict[str, Any]]:
    """
    Given a decoded JSON-LD payload (dict/list), find a dict whose @type includes 'Recipe'.
    Supports items inside @graph, arrays, nested objects.
    """
    def is_recipe_obj(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        t = obj.get("@type")
        if isinstance(t, list):
            return any(str(x).lower() == "recipe" for x in t)
        if isinstance(t, str):
            return t.lower() == "recipe"
        return False

    # Direct dict
    if isinstance(ld, dict):
        if is_recipe_obj(ld):
            return ld
        # @graph case
        graph = ld.get("@graph")
        if isinstance(graph, list):
            for node in graph:
                if is_recipe_obj(node):
                    return node
        # Thing with mainEntity
        me = ld.get("mainEntity")
        if is_recipe_obj(me):
            return me
        if isinstance(me, list):
            for node in me:
                if is_recipe_obj(node):
                    return node

    # Array of things
    if isinstance(ld, list):
        for item in ld:
            r = _pick_recipe_from_ld(item)
            if r:
                return r
    return None

def _normalize_yield(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, list):
        raw = " / ".join([_clean_text(str(x)) for x in raw if x])
    return _clean_text(str(raw))

def _normalize_ingredients(raw: Any) -> List[str]:
    out: List[str] = []
    for item in _as_list(raw):
        if isinstance(item, str):
            out.append(_clean_text(item))
        elif isinstance(item, dict):
            # Some sites wrap ingredients as objects with 'text'
            out.append(_clean_text(item.get("text") or item.get("name") or ""))
    return [x for x in out if x]

def _flatten_instruction_node(node: Any) -> List[str]:
    """
    Handles strings, HowToStep dicts with 'text', HowToSection with 'itemListElement', or lists.
    """
    steps: List[str] = []
    if node is None:
        return steps
    if isinstance(node, str):
        t = _clean_text(node)
        if t:
            steps.append(t)
        return steps
    if isinstance(node, list):
        for n in node:
            steps.extend(_flatten_instruction_node(n))
        return steps
    if isinstance(node, dict):
        node_type = node.get("@type") or ""
        # HowToSection: recurse into itemListElement
        if isinstance(node.get("itemListElement"), list):
            for n in node["itemListElement"]:
                steps.extend(_flatten_instruction_node(n))
        # HowToStep or generic object with 'text'
        t = _clean_text(node.get("text") or node.get("name") or "")
        if t:
            steps.append(t)
        return steps
    return steps

def _extract_json_ld(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    """
    Iterate all JSON-LD scripts; choose the most complete Recipe object.
    """
    best: Optional[Dict[str, Any]] = None
    best_score = -1

    for script in soup.find_all("script", type=lambda v: v and "ld+json" in v):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            # Sometimes there's invalid JSON; try to fix common trailing commas
            try:
                payload = json.loads(re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", raw)))
            except Exception:
                continue
        recipe_obj = _pick_recipe_from_ld(payload)
        if not recipe_obj:
            continue

        # Quick “completeness” score to prefer objects with ingredients & instructions
        ingr = _normalize_ingredients(recipe_obj.get("recipeIngredient"))
        instr = _flatten_instruction_node(recipe_obj.get("recipeInstructions"))
        score = (len(ingr) > 0) + (len(instr) > 0)

        if score > best_score:
            best = recipe_obj
            best_score = score

    return best

def _collect_text_from_list(root: Tag) -> List[str]:
    items: List[str] = []
    for li in root.find_all(["li", "p"], recursive=True):
        # Only take direct text, not nested section headers etc.
        txt = " ".join(li.stripped_strings)
        txt = _clean_text(txt)
        if txt:
            items.append(txt)
    return items

def _fallback_from_html(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """
    Broad HTML fallback for ingredients and instructions.
    Targets Simply Recipes and many other sites.
    """
    ingredients: List[str] = []
    instructions: List[str] = []

    # --- INGREDIENTS ---
    # Common Simply Recipes / Dotdash patterns:
    ingr_candidates = [
        # Simply Recipes & Dotdash
        {"css": "ul.structured-ingredients__list"},
        {"css": "ul.mntl-structured-ingredients__list"},
        {"css": "[data-ingredient-list], ul[data-ingredient-list]"},
        # microdata
        {"css": "[itemprop='recipeIngredient']"},
        # generic
        {"css": "ul[class*='ingredient']"},
        {"css": "div[class*='ingredient'] ul"},
    ]

    for cand in ingr_candidates:
        for node in soup.select(cand["css"]):
            ingredients.extend(_collect_text_from_list(node))
        if ingredients:
            break

    # A more generic catch:
    if not ingredients:
        for ul in soup.find_all("ul"):
            cls = " ".join(ul.get("class", []))
            if re.search(r"ingredient", cls, re.I):
                ingredients.extend(_collect_text_from_list(ul))
        ingredients = [x for x in ingredients if x]

    # --- INSTRUCTIONS ---
    instr_candidates = [
        {"css": "section.instructions-section"},
        {"css": "ol.mntl-structured-instructions__list"},      # Dotdash
        {"css": "ol[class*='instruction']"},
        {"css": "ul[class*='instruction']"},
        {"css": "[itemprop='recipeInstructions']"},
        {"css": "div[class*='direction'] ol"},
        {"css": "div[class*='method'] ol"},
    ]

    for cand in instr_candidates:
        for node in soup.select(cand["css"]):
            instructions.extend(_collect_text_from_list(node))
        if instructions:
            break

    if not instructions:
        # Very generic fallback: look for ordered lists that look like steps
        for ol in soup.find_all("ol"):
            cls = " ".join(ol.get("class", []))
            if re.search(r"(instruction|direction|method|step)", cls, re.I):
                instructions.extend(_collect_text_from_list(ol))
        instructions = [x for x in instructions if x]

    return {"ingredients": ingredients, "instructions": instructions}

def _extract_fields_from_recipe_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    title = _first(obj.get("name"), obj.get("headline"))
    description = _clean_text(_first(obj.get("description"), obj.get("summary")) or "")
    author = None
    auth = obj.get("author")
    if isinstance(auth, dict):
        author = _first(auth.get("name"), auth.get("@name"))
    elif isinstance(auth, list) and auth:
        if isinstance(auth[0], dict):
            author = auth[0].get("name")
        else:
            author = str(auth[0])

    ingredients = _normalize_ingredients(obj.get("recipeIngredient"))
    instructions = _flatten_instruction_node(obj.get("recipeInstructions"))

    times = {}
    for key in ["prepTime", "cookTime", "totalTime"]:
        if obj.get(key):
            times[key] = _clean_text(obj[key])

    nutrition = {}
    nut = obj.get("nutrition")
    if isinstance(nut, dict):
        for k, v in nut.items():
            if v is None:
                continue
            if isinstance(v, (int, float)):
                nutrition[k] = v
            else:
                nutrition[k] = _clean_text(str(v))

    tags = None
    keyw = obj.get("keywords")
    if isinstance(keyw, str):
        tags = [x.strip() for x in keyw.split(",") if x.strip()]
    elif isinstance(keyw, list):
        tags = [str(x).strip() for x in keyw if str(x).strip()]

    return dict(
        title=_clean_text(title or ""),
        description=description or None,
        author=_clean_text(author or "") or None,
        ingredients=ingredients,
        instructions=instructions,
        recipe_yield=_normalize_yield(obj.get("recipeYield")),
        times=times,
        nutrition=nutrition,
        tags=tags or None,
    )

# ---------------------------
# Endpoints
# ---------------------------

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/extract", response_model=Recipe)
async def extract(req: ExtractRequest):
    html = await fetch_html(str(req.url))
    soup = BeautifulSoup(html, "lxml")

    # 1) Try JSON-LD (robust)
    obj = _extract_json_ld(soup)

    data: Dict[str, Any] = {
        "title": None,
        "description": None,
        "author": None,
        "ingredients": [],
        "instructions": [],
        "recipe_yield": None,
        "times": {},
        "nutrition": {},
        "tags": None,
    }

    if obj:
        data.update(_extract_fields_from_recipe_obj(obj))

    # 2) Fallback to HTML if needed or if JSON-LD is incomplete
    if not data["ingredients"] or not data["instructions"]:
        fallback = _fallback_from_html(soup)
        if not data["ingredients"]:
            data["ingredients"] = fallback["ingredients"]
        if not data["instructions"]:
            data["instructions"] = fallback["instructions"]

    # 3) Title/description fallback from DOM
    if not data["title"]:
        ttl = soup.find("meta", property="og:title") or soup.find("title")
        data["title"] = _clean_text(ttl.get("content") if ttl and ttl.has_attr("content") else (ttl.get_text() if ttl else "")) or None
    if not data["description"]:
        md = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
        if md and md.has_attr("content"):
            data["description"] = _clean_text(md["content"]) or None

    # Build and return
    recipe = Recipe(
        source_url=str(req.url),
        title=data["title"],
        description=data["description"],
        ingredients=data["ingredients"],
        instructions=data["instructions"],
        recipe_yield=data["recipe_yield"],
        times=data["times"],
        nutrition=data["nutrition"],
        tags=data["tags"],
        author=data["author"],
    )
    return recipe

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    text = re.sub(r"\s+", " ", req.text or "").strip()
    return {"length": len(text), "preview": text[:160]}

