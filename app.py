import json
import os
import re
import time
from typing import List, Optional, Union

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------
# Utilities
# ---------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def clean_text(s: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", (s or "").strip())

def ensure_list(x: Union[str, dict, List[Union[str, dict]], None]) -> List[str]:
    """
    Normalize recipeIngredient / recipeInstructions into a list of strings.
    Handles:
      - list[str]
      - list[dict] with 'text'
      - dict with 'text'
      - single string
      - None
    """
    if x is None:
        return []
    if isinstance(x, str):
        return [clean_text(x)]
    if isinstance(x, dict):
        # Some pages use HowToSection/HowToStep objects
        if "text" in x and isinstance(x["text"], str):
            return [clean_text(x["text"])]
        # If it’s a HowToSection with 'itemListElement'
        if "itemListElement" in x:
            return ensure_list(x["itemListElement"])
        return []
    items: List[str] = []
    for item in x:
        if isinstance(item, str):
            items.append(clean_text(item))
        elif isinstance(item, dict):
            if "text" in item and isinstance(item["text"], str):
                items.append(clean_text(item["text"]))
            elif "itemListElement" in item:
                items.extend(ensure_list(item["itemListElement"]))
    return [t for t in items if t]

def pick_first_text(obj: Union[str, dict, List[Union[str, dict]], None]) -> Optional[str]:
    """Get a reasonable string from JSON-LD title/description/author shapes."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return clean_text(obj)
    if isinstance(obj, dict):
        for k in ("name", "text", "title"):
            if k in obj and isinstance(obj[k], str):
                return clean_text(obj[k])
        return None
    # list case
    for item in obj:
        val = pick_first_text(item)
        if val:
            return val
    return None

def parse_iso8601_duration(dur: Optional[str]) -> Optional[int]:
    """
    Parse ISO 8601 duration like 'PT20M', 'PT6H', 'PT6H20M', 'P0DT40M', etc.
    Return total minutes, or None.
    """
    if not dur or not isinstance(dur, str):
        return None
    # tolerance for lower/uppercase
    m = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<mins>\d+)M)?(?:(?P<secs>\d+)S)?)?",
        dur.strip(),
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    days = int(m.group("days") or 0)
    hours = int(m.group("hours") or 0)
    mins = int(m.group("mins") or 0)
    secs = int(m.group("secs") or 0)
    total = days * 24 * 60 + hours * 60 + mins + (secs // 60)
    return total if total > 0 else None

def format_minutes_human(total_mins: Optional[int]) -> Optional[str]:
    """'380' -> '6 hrs 20 mins', '60' -> '1 hr', '45' -> '45 mins'."""
    if total_mins is None:
        return None
    h, m = divmod(int(total_mins), 60)
    parts = []
    if h > 0:
        parts.append(f"{h} hr" + ("" if h == 1 else "s"))
    if m > 0 or not parts:
        parts.append(f"{m} min" + ("" if m == 1 else "s"))
    return " ".join(parts)

def walk_for_recipes(node) -> List[dict]:
    """
    Walk any JSON structure and collect objects whose @type is/contains 'Recipe'.
    Handles dict, list, and @graph shapes.
    """
    found = []
    if isinstance(node, dict):
        t = node.get("@type")
        if isinstance(t, str) and t.lower() == "recipe":
            found.append(node)
        elif isinstance(t, list) and any((isinstance(x, str) and x.lower() == "recipe") for x in t):
            found.append(node)
        # Recurse into possible fields
        for k, v in node.items():
            if k in ("@graph", "graph", "itemListElement", "mainEntity", "relatedLink"):
                found.extend(walk_for_recipes(v))
    elif isinstance(node, list):
        for item in node:
            found.extend(walk_for_recipes(item))
    return found

def extract_via_jsonld(soup: BeautifulSoup) -> Optional[dict]:
    """
    Use JSON-LD to extract recipe fields (most reliable on Simply Recipes,
    Allrecipes, NYT, etc.). Return dict or None if not found.
    """
    scripts = soup.find_all("script", {"type": "application/ld+json"})
    for tag in scripts:
        try:
            data = json.loads(tag.string or tag.text or "")
        except Exception:
            continue

        # Collect all Recipe objects anywhere in this blob
        recipes = walk_for_recipes(data if isinstance(data, (dict, list)) else [])
        if not recipes:
            # Some sites dump a list at the top level
            if isinstance(data, list):
                for item in data:
                    recipes.extend(walk_for_recipes(item))
        if not recipes:
            continue

        # Prefer the first one that actually has ingredients/instructions
        for r in recipes:
            ingredients = ensure_list(r.get("recipeIngredient"))
            instructions = ensure_list(r.get("recipeInstructions"))
            title = pick_first_text(r.get("name"))
            description = pick_first_text(r.get("description"))
            author = pick_first_text(r.get("author"))

            prep = format_minutes_human(parse_iso8601_duration(r.get("prepTime")))
            cook = format_minutes_human(parse_iso8601_duration(r.get("cookTime")))
            total = format_minutes_human(parse_iso8601_duration(r.get("totalTime")))

            # If both ingredient/instructions are present as lists, we’re good.
            if ingredients or instructions:
                return {
                    "title": title,
                    "description": description,
                    "author": author,
                    "ingredients": ingredients,
                    "instructions": instructions,
                    "times": {
                        "prepTime": prep,
                        "cookTime": cook,
                        "totalTime": total,
                    },
                }
    return None

# Fallback CSS selectors for common sites
INGREDIENT_SELECTORS = [
    "[itemprop='recipeIngredient'] li",
    ".ingredient, .ingredients li, .ingredients__item, .mntl-structured-ingredients__item",
    "[data-ingredient], .recipe-ingredients li",
]
INSTRUCTION_SELECTORS = [
    "[itemprop='recipeInstructions'] li",
    ".instruction, .instructions li, .steps li, .direction, .directions li, "
    ".mntl-sc-block-group--LI, .mntl-sc-block-step",
]

def extract_via_html(soup: BeautifulSoup) -> Optional[dict]:
    """Fallback extraction from visible HTML lists (each LI becomes an item)."""
    ingredients: List[str] = []
    for sel in INGREDIENT_SELECTORS:
        for li in soup.select(sel):
            txt = clean_text(li.get_text(" "))
            if txt and len(txt) > 1:
                ingredients.append(txt)
        if ingredients:
            break  # stop at first selector that yields content

    instructions: List[str] = []
    for sel in INSTRUCTION_SELECTORS:
        for li in soup.select(sel):
            txt = clean_text(li.get_text(" "))
            if txt and len(txt) > 1:
                instructions.append(txt)
        if instructions:
            break

    # Try title/description from the page if available
    title = None
    if soup.title and soup.title.string:
        title = clean_text(soup.title.string)

    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = clean_text(meta_desc["content"]) if meta_desc and meta_desc.get("content") else None

    if ingredients or instructions:
        return {
            "title": title,
            "description": description,
            "author": None,
            "ingredients": ingredients,
            "instructions": instructions,
            "times": {"prepTime": None, "cookTime": None, "totalTime": None},
        }
    return None

# ---------------------------
# API model & app
# ---------------------------

class ExtractRequest(BaseModel):
    url: str

class Recipe(BaseModel):
    schema_version: str = "1.0.0"
    source_url: str
    title: Optional[str]
    description: Optional[str]
    ingredients: List[str]
    instructions: List[str]
    times: dict
    nutrition: Optional[dict] = {}
    tags: Optional[List[str]] = []
    author: Optional[str]
    extracted_at: int

app = FastAPI(title="Recipe Extractor", version="1.0.0", description="Extract a normalized recipe JSON from a public recipe URL.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
def health():
    return {"ok": True, "ts": int(time.time())}

@app.post("/extract", response_model=Recipe, tags=["Extract"])
def extract(payload: ExtractRequest):
    url = payload.url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="Provide a valid http(s) URL.")

    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {e}")

    soup = BeautifulSoup(r.text, "lxml")

    # 1) Prefer JSON-LD
    data = extract_via_jsonld(soup)

    # 2) Fallback to visible HTML lists
    if not data:
        data = extract_via_html(soup)

    if not data:
        raise HTTPException(status_code=500, detail="Could not find recipe data on the page.")

    # Final cleanups (defensive)
    ingredients = [clean_text(x) for x in data["ingredients"] if clean_text(x)]
    instructions = [clean_text(x) for x in data["instructions"] if clean_text(x)]

    recipe = Recipe(
        source_url=url,
        title=data.get("title"),
        description=data.get("description"),
        author=data.get("author"),
        ingredients=ingredients,
        instructions=instructions,
        times=data.get("times", {"prepTime": None, "cookTime": None, "totalTime": None}),
        extracted_at=int(time.time()),
    )
    return recipe

if __name__ == "__main__":
    # For local runs: uvicorn app:app --reload
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
