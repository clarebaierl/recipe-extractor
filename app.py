import os
import json
import re
from typing import Any, Dict, List, Optional, Union

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# ---------- FastAPI setup ----------
app = FastAPI(
    title="Recipe Extractor",
    version="1.0.0",
    description="Extract a normalized recipe JSON from a public recipe URL.",
)

# CORS: keep permissive for now; lock down later if you need
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class ExtractRequest(BaseModel):
    url: str

# ---------- Helpers ----------

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_html(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {e}")


def clean_str(s: str) -> str:
    # collapse whitespace and strip
    return re.sub(r"\s+", " ", s).strip()


def to_human_time(iso: Optional[str]) -> Optional[str]:
    """Convert ISO-8601 durations (e.g., PT6H20M) to '6 hrs 20 mins'."""
    if not iso:
        return None
    m = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<mins>\d+)M)?(?:(?P<secs>\d+)S)?)?",
        iso.strip().upper(),
    )
    if not m:
        # some sites put just minutes like 'PT360M' or 'PT20M'
        return iso
    days = int(m.group("days") or 0)
    hours = int(m.group("hours") or 0)
    mins = int(m.group("mins") or 0)
    secs = int(m.group("secs") or 0)

    total_mins = days * 24 * 60 + hours * 60 + mins
    h, r = divmod(total_mins, 60)
    parts = []
    if h:
        parts.append(f"{h} hr" + ("s" if h != 1 else ""))
    if r:
        parts.append(f"{r} min" + ("s" if r != 1 else ""))
    if not parts and secs:
        parts.append(f"{secs} sec" + ("s" if secs != 1 else ""))
    if not parts:
        parts.append("0 mins")
    return " ".join(parts)


# --- Run-on item splitter ---
_RUNON_SPLIT_PATTERN = re.compile(
    # Insert a split *before* a number that immediately follows a letter or ')'
    r"(?<=[A-Za-z\)])(?=\d)"
)

def split_runons(items: List[str]) -> List[str]:
    """
    Fix ingredients that accidentally merged, e.g.:
      '... Cabernet Sauvignon3 ribs celery, chopped2 medium parsnips ...'
    by splitting into separate items where a number follows a letter with no space.
    """
    fixed: List[str] = []
    for raw in items:
        # First normalize whitespace
        piece = clean_str(raw)

        # If there's an obvious run-on boundary, split there
        parts = _RUNON_SPLIT_PATTERN.split(piece)

        # The split keeps numbers at the start of new parts; now each part is a candidate item.
        for p in parts:
            p = clean_str(p)
            if not p:
                continue
            fixed.append(p)
    return fixed


def ensure_list(x: Union[str, List[Any], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        # Extract text if elements are dicts like {"@type":"HowToStep","text":"..."}
        out: List[str] = []
        for el in x:
            if isinstance(el, dict) and "text" in el:
                out.append(clean_str(str(el["text"])))
            else:
                out.append(clean_str(str(el)))
        return out
    return [clean_str(str(x))]


def extract_from_jsonld(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    scripts = soup.find_all("script", {"type": "application/ld+json"})
    candidates: List[Dict[str, Any]] = []
    for sc in scripts:
        try:
            data = json.loads(sc.string or sc.get_text() or "")
        except Exception:
            continue

        def push(obj: Any):
            if isinstance(obj, dict):
                t = obj.get("@type")
                if isinstance(t, list):
                    tset = {str(v).lower() for v in t}
                else:
                    tset = {str(t).lower()} if t else set()
                if "recipe" in tset:
                    candidates.append(obj)
            elif isinstance(obj, list):
                for o in obj:
                    push(o)

        push(data)

    if not candidates:
        return None

    # Heuristic: prefer the one that has most recipe fields
    best = max(
        candidates,
        key=lambda d: sum(k in d for k in ("name", "recipeIngredient", "recipeInstructions")),
    )

    # Normalize out
    title = clean_str(best.get("name") or "")
    description = clean_str(best.get("description") or "")
    author = ""
    au = best.get("author")
    if isinstance(au, dict):
        author = clean_str(au.get("name") or "")
    elif isinstance(au, list) and au:
        if isinstance(au[0], dict):
            author = clean_str(au[0].get("name") or "")
        else:
            author = clean_str(str(au[0]))
    elif isinstance(au, str):
        author = clean_str(au)

    ingredients = ensure_list(best.get("recipeIngredient"))
    ingredients = split_runons(ingredients)  # <- fix run-ons here

    instructions = ensure_list(best.get("recipeInstructions"))

    times = {
        "prepTime": to_human_time(best.get("prepTime")),
        "cookTime": to_human_time(best.get("cookTime")),
        "totalTime": to_human_time(best.get("totalTime")),
    }

    # Optional: nutrition object sometimes present
    nutrition = best.get("nutrition") if isinstance(best.get("nutrition"), dict) else {}

    # Tags (keywords) can be comma string or list
    tags: List[str] = []
    kw = best.get("keywords")
    if isinstance(kw, str):
        tags = [clean_str(k) for k in kw.split(",") if clean_str(k)]
    elif isinstance(kw, list):
        tags = [clean_str(k) for k in kw]

    return {
        "title": title,
        "description": description,
        "author": author,
        "ingredients": ingredients,
        "instructions": instructions,
        "times": {k: v for k, v in times.items() if v},
        "nutrition": nutrition or {},
        "tags": tags,
    }


def extract_from_html(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Fallback when JSON-LD isn’t usable: try to pick <li> items for ingredients
    and numbered/step lists for instructions.
    """
    # Ingredients candidates
    ing_candidates = []
    for sel in [
        # common patterns
        "[class*='ingredient'] li",
        "[id*='ingredient'] li",
        "li[itemprop='recipeIngredient']",
    ]:
        ing_candidates = [clean_str(li.get_text(" ")) for li in soup.select(sel)]
        if len(ing_candidates) >= 2:
            break
    ing_candidates = split_runons(ing_candidates)

    # Instructions candidates
    inst_candidates = []
    for sel in [
        "[class*='instruction'] li",
        "[id*='instruction'] li",
        "li[itemprop='recipeInstructions']",
        "[class*='instruction'] p",
    ]:
        inst_candidates = [clean_str(x.get_text(" ")) for x in soup.select(sel)]
        if len(inst_candidates) >= 2:
            break

    return {
        "title": clean_str(soup.title.get_text()) if soup.title else "",
        "description": "",
        "author": "",
        "ingredients": [i for i in ing_candidates if i],
        "instructions": [i for i in inst_candidates if i],
        "times": {},
        "nutrition": {},
        "tags": [],
    }


# ---------- Routes ----------
@app.get("/")
def health():
    return {"ok": True}


@app.post("/extract")
def extract(req: ExtractRequest):
    html = fetch_html(req.url)
    soup = BeautifulSoup(html, "lxml")

    payload = extract_from_jsonld(soup) or extract_from_html(soup)

    # Final schema wrapper
    return {
        "schema_version": "1.0.0",
        "source_url": req.url,
        **payload,
        "extracted_at": int(__import__("time").time() * 1000),
    }

