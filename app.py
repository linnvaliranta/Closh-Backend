# app.py (Closh backend v0.3.0)
import os
import io
import uuid
import base64
import json
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import requests
from dotenv import load_dotenv

# optional S3 helper (local fallback)
try:
    from s3_helpers import upload_bytes_to_s3, get_s3_public_url
except Exception:
    upload_bytes_to_s3 = None
    get_s3_public_url = None

load_dotenv()

# -------- Config --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REMOVEBG_API_KEY = os.getenv("REMOVEBG_API_KEY")
USE_S3 = os.getenv("USE_S3", "false").lower() in ("1", "true", "yes")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Image size (512 for speed/quality tradeoff)
IMAGE_SIZE = "512x512"
IMG_W, IMG_H = 512, 512

# -------- FastAPI --------
app = FastAPI(title="Closh API", version="0.3.0")

# CORS - set origins in .env via CORS_ORIGINS, default '*' for dev (change for production)
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Helpers --------
def _save_bytes_local(data: bytes, subdir: str) -> str:
    folder = os.path.join(STORAGE_DIR, subdir)
    os.makedirs(folder, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.png"
    path = os.path.join(folder, fname)
    with open(path, "wb") as f:
        f.write(data)
    return path

def _save_bytes(data: bytes, subdir: str) -> str:
    """Save to S3 if enabled, otherwise local path. Return public URL if S3 else local file path."""
    if USE_S3 and S3_BUCKET and upload_bytes_to_s3 and get_s3_public_url:
        key = f"{subdir}/{uuid.uuid4().hex}.png"
        upload_bytes_to_s3(data, S3_BUCKET, key)
        return get_s3_public_url(S3_BUCKET, key)
    else:
        return _save_bytes_local(data, subdir)

def _download_to_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def get_openai_client():
    """Return a configured OpenAI client instance (using OpenAI Python lib)."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
    import openai
    return openai.OpenAI(api_key=OPENAI_API_KEY)

def _openai_images_generate(prompt: str) -> bytes:
    """Call OpenAI Images (gpt-image-1) and return PNG bytes."""
    client = get_openai_client()
    try:
        resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=IMAGE_SIZE,
            n=1
        )
        b64 = resp.data[0].b64_json
        return base64.b64decode(b64)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI image error: {e}")

def _openai_text_generate(prompt: str, max_tokens: int = 300) -> str:
    """Call OpenAI text responses and return text output."""
    client = get_openai_client()
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=max_tokens
        )
        # Extract text safely
        out_text = ""
        if hasattr(response, "output") and response.output:
            try:
                contents = response.output[0].content
                for c in contents:
                    if hasattr(c, "text"):
                        out_text += c.text
                    elif isinstance(c, dict) and "text" in c:
                        out_text += c["text"]
            except Exception:
                out_text = getattr(response, "output_text", "")
        else:
            out_text = getattr(response, "output_text", "")
        return (out_text or "").strip()
    except Exception as e:
        # Propagate HTTPException so callers can fallback
        raise HTTPException(status_code=502, detail=f"OpenAI text error: {e}")

def _remove_bg(image_bytes: bytes) -> bytes:
    if not REMOVEBG_API_KEY:
        return image_bytes
    try:
        r = requests.post(
            "https://api.remove.bg/v1.0/removebg",
            files={"image_file": ("upload.png", image_bytes)},
            data={"size": "auto"},
            headers={"X-Api-Key": REMOVEBG_API_KEY},
            timeout=120,
        )
        if r.status_code != 200:
            return image_bytes
        return r.content
    except Exception:
        return image_bytes

def _compose_layers(base_png: bytes, layer_pngs: List[bytes], positions: Optional[List[tuple]] = None) -> bytes:
    """Composite multiple transparent PNG layers over a base doll PNG."""
    base = Image.open(io.BytesIO(base_png)).convert("RGBA").resize((IMG_W, IMG_H), Image.LANCZOS)
    canvas = Image.new("RGBA", (IMG_W, IMG_H), (255, 255, 255, 0))
    canvas.alpha_composite(base)

    if positions is None:
        positions = [(128, 120, 0.9), (120, 260, 0.9), (120, 380, 0.6), (120, 220, 1.0)]

    for idx, layer_bytes in enumerate(layer_pngs):
        try:
            layer = Image.open(io.BytesIO(layer_bytes)).convert("RGBA")
        except Exception:
            continue
        lw, lh = layer.size
        max_side = max(lw, lh)
        square = Image.new("RGBA", (max_side, max_side), (255, 255, 255, 0))
        square.paste(layer, ((max_side - lw) // 2, (max_side - lh) // 2))
        x, y, scale = positions[min(idx, len(positions) - 1)]
        size = int(IMG_W * scale)
        layer_rs = square.resize((size, size), Image.LANCZOS)
        canvas.alpha_composite(layer_rs, dest=(int(x), int(y)))

    out = io.BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()

# -------- Schemas --------
class TextItemIn(BaseModel):
    description: str
    illustration_style: str
    category: str

class TextItemOut(BaseModel):
    image_url: str

class ComposeIn(BaseModel):
    doll_image_path: str
    item_image_paths: List[str]
    positions: Optional[List[List[float]]] = None

class ComposeOut(BaseModel):
    composed_url: str

class WardrobeItem(BaseModel):
    id: str
    category: str   # e.g. "tops", "bottoms", "coats", "shoes", "accessories"
    description: str
    image_url: Optional[str] = None

class OutfitRequest(BaseModel):
    doll: dict                 # { "image_url": "...", "gender": "...", etc. } preferred to include image_url
    art_style: str             # "watercolor", "bold_pop", "runway_sketch"
    muses: List[str] = []      # ["Kate Moss", "Bella Hadid"]
    style_keywords: List[str] = [] # ["Minimalist", "Streetwear"]
    wardrobe: List[WardrobeItem]
    occasion: Optional[str] = None  # "Relaxed", "Everyday", "Professional", "Going Out"
    weather: Optional[str] = None   # "Warm/Hot", "Mild/Spring", "Cold/Winter", "Rainy"

# -------- Static lists (muses + keywords) --------
CLOSH_MUSES = [
    "Kate Moss", "Jane Birkin", "Bella Hadid", "Rihanna", "Jennie Kim",
    "Princess Diana", "Carrie Bradshaw", "Kim Kardashian", "Olsen Twins",
    "Blair Waldorf", "Tyler, The Creator", "JFK Jr.", "Mick Jagger",
    "A$AP Rocky", "Harry Styles", "Al Pacino", "Alain Delon",
    "Bad Bunny", "Devon Aoki"
]

CLOSH_KEYWORDS = [
    "Minimalist", "Scandinavian", "Parisian Chic", "Preppy", "Grunge",
    "Y2K", "Vintage / Retro", "Boho Chic", "Streetwear", "Feminine",
    "Edgy / Rocker", "Old Money / Classic", "High Glam"
]

# -------- Routes --------
@app.get("/")
def root():
    return {"status": "ok", "version": "0.3.0"}

@app.get("/muses")
def get_muses():
    return {"muses": CLOSH_MUSES}

@app.get("/style_keywords")
def get_style_keywords():
    return {"style_keywords": CLOSH_KEYWORDS}

# ---------- item endpoints (text / photo) ----------
@app.post("/items/text", response_model=TextItemOut)
@app.post("/doll")
async def create_doll(
    gender: str = Form(...),
    size: str = Form(...),
    skin_tone: str = Form(...),
    style: str = Form(...)
):
    """
    Generate a customizable doll illustration for the user.
    """
    try:
        gender_map = {
            "female": "a feminine figure",
            "male": "a masculine figure",
            "neutral": "a gender-neutral figure"
        }
        size_map = {
            "slim": "slim body type",
            "average": "average build",
            "curvy": "curvy body shape",
            "plus": "plus-size figure"
        }
        skin_map = {
            "light": "light skin tone",
            "medium": "medium skin tone",
            "dark": "dark brown skin tone",
            "deep": "deep ebony skin tone"
        }
        style_map = {
            "watercolor": "soft pastel watercolor fashion sketch",
            "bold_pop": "bold cartoon pop-art style",
            "runway_sketch": "high-fashion runway designer sketch"
        }

        prompt = f"A fashion doll illustration of {gender_map.get(gender, gender)}, {size_map.get(size, size)}, {skin_map.get(skin_tone, skin_tone)}, drawn in {style_map.get(style, style)}."

        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="512x512"
        )

        image_url = response.data[0].url
        return {"doll_url": image_url, "prompt": prompt}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def generate_item_from_text(payload: TextItemIn):
    """
    Create an illustrated wardrobe item from a text description.
    (Either use this OR /items/photo for a photo upload — not both.)
    """
    style_map = {
        "watercolor": "soft pastel watercolor, delicate brush strokes",
        "bold pop": "bold colorful cartoon pop, thick black outlines, vivid palette",
        "runway sketch": "high-fashion runway sketch, loose ink lines and minimal washes",
        "runway_sketch": "high-fashion runway sketch, loose ink lines and minimal washes",
        "bold_pop": "bold colorful cartoon pop, thick black outlines, vivid palette"
    }
    style_phrase = style_map.get(payload.illustration_style.lower(), "soft pastel watercolor")
    prompt = (
        f"Standalone fashion illustration of {payload.description}. "
        f"Transparent background, front-facing flat item for a digital closet, {style_phrase}. "
        "High resolution and clean edges, no text, no logos."
    )
    png_bytes = _openai_images_generate(prompt)
    url = _save_bytes(png_bytes, subdir="items")
    return TextItemOut(image_url=url)

@app.post("/items/photo", response_model=TextItemOut)
async def generate_item_from_photo(
    illustration_style: str = Form(...),
    category: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Upload a photo of a garment — remove background and redraw in chosen art style.
    """
    raw = await file.read()
    cutout = _remove_bg(raw)
    style_map = {
        "watercolor": "soft pastel watercolor, delicate brush strokes",
        "bold pop": "bold colorful cartoon pop, thick black outlines, vivid palette",
        "runway sketch": "high-fashion runway sketch, loose ink lines and minimal washes",
        "runway_sketch": "high-fashion runway sketch, loose ink lines and minimal washes",
        "bold_pop": "bold colorful cartoon pop, thick black outlines, vivid palette"
    }
    style_phrase = style_map.get(illustration_style.lower(), "soft pastel watercolor")
    prompt = (
        f"Redraw the uploaded garment as a standalone fashion illustration. Keep silhouette and pattern faithful. "
        f"Transparent background, front-facing flat item for a digital closet, {style_phrase}. No text or logos."
    )
    _save_bytes(raw, subdir="uploads")
    png_bytes = _openai_images_generate(prompt)
    url = _save_bytes(png_bytes, subdir="items")
    return TextItemOut(image_url=url)

# ---------- compose outfit (layers) ----------
@app.post("/outfits/compose", response_model=ComposeOut)
def compose_outfit(payload: ComposeIn):
    # Load doll image (could be S3 URL or local path)
    layer_bytes = []
    try:
        doll_bytes = _download_to_bytes(payload.doll_image_path) if payload.doll_image_path.startswith("http") else open(payload.doll_image_path, "rb").read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not load doll image")

    for p in payload.item_image_paths:
        try:
            if p.startswith("http"):
                layer_bytes.append(_download_to_bytes(p))
            else:
                with open(p, "rb") as f:
                    layer_bytes.append(f.read())
        except Exception:
            continue

    positions = None
    if payload.positions:
        positions = [tuple(map(float, t)) for t in payload.positions]

    composed = _compose_layers(doll_bytes, layer_bytes, positions)
    url = _save_bytes(composed, subdir="outfits")
    return ComposeOut(composed_url=url)

# ---------- doll endpoints ----------
@app.post("/doll")
async def create_doll(
    gender: str = Form(...),
    size: str = Form(...),
    skin_tone: str = Form(...),
    style: str = Form(...),
    muses: Optional[str] = Form(None),           # comma-separated optional list
    style_keywords: Optional[str] = Form(None)   # comma-separated optional list
):
    """
    Generate a customizable doll illustration for the user.
    muses and style_keywords can be provided at profile creation time (comma separated).
    """
    try:
        # Map input values into a nice natural prompt
        gender_map = {
            "female": "a feminine figure",
            "male": "a masculine figure",
            "neutral": "a gender-neutral figure",
        }
        size_map = {
            "slim": "slim body type",
            "average": "average build",
            "curvy": "curvy body shape",
            "plus": "plus-size figure"
        }
        skin_map = {
            "light": "light skin tone",
            "medium": "medium skin tone",
            "dark": "dark brown skin tone",
            "deep": "deep ebony skin tone"
        }
        style_map = {
            "watercolor": "soft pastel watercolor fashion sketch",
            "bold pop": "bold cartoon pop-art style",
            "runway sketch": "high-fashion runway designer sketch",
            "runway_sketch": "high-fashion runway designer sketch",
            "bold_pop": "bold cartoon pop-art style"
        }

        muse_text = ""
        keyword_text = ""
        if muses:
            muse_list = [m.strip() for m in muses.split(",") if m.strip()]
            muse_text = "Inspired by muses: " + ", ".join(muse_list) + ". "
        if style_keywords:
            kw_list = [k.strip() for k in style_keywords.split(",") if k.strip()]
            keyword_text = "Style keywords: " + ", ".join(kw_list) + ". "

        prompt = (
            f"A fashion doll illustration of {gender_map.get(gender, gender)}, "
            f"{size_map.get(size, size)}, {skin_map.get(skin_tone, skin_tone)}, "
            f"drawn in {style_map.get(style, style)}. "
            f"{muse_text}{keyword_text}Front-facing, neutral pose, full body, transparent background."
        )

        client = get_openai_client()
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=IMAGE_SIZE,
            n=1
        )
        b64 = response.data[0].b64_json
        png_bytes = base64.b64decode(b64)
        url = _save_bytes(png_bytes, subdir="dolls")
        return {"doll_url": url, "prompt": prompt}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/doll/restyle")
async def restyle_doll(
    gender: str = Form(...),
    size: str = Form(...),
    skin_tone: str = Form(...),
    new_style: str = Form(...)
):
    """
    Redraw an existing doll with a new art style while keeping gender, size, and skin tone the same.
    """
    try:
        gender_map = {
            "female": "a feminine figure",
            "male": "a masculine figure",
            "neutral": "a gender-neutral figure"
        }
        size_map = {
            "slim": "slim body type",
            "average": "average build",
            "curvy": "curvy body shape",
            "plus": "plus-size figure"
        }
        skin_map = {
            "light": "light skin tone",
            "medium": "medium skin tone",
            "dark": "dark brown skin tone",
            "deep": "deep ebony skin tone"
        }
        style_map = {
            "watercolor": "soft pastel watercolor fashion sketch",
            "bold_pop": "bold cartoon pop-art style",
            "runway_sketch": "high-fashion runway designer sketch"
        }

        prompt = f"A fashion doll illustration of {gender_map.get(gender, gender)}, {size_map.get(size, size)}, {skin_map.get(skin_tone, skin_tone)}, drawn in {style_map.get(new_style, new_style)}."

        client = get_openai_client()
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=IMAGE_SIZE,
            n=1
        )
        b64 = response.data[0].b64_json
        png_bytes = base64.b64decode(b64)
        url = _save_bytes(png_bytes, subdir="dolls")
        return {"restyled_doll_url": url, "prompt": prompt}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/doll/update")
async def update_doll(
    gender: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    skin_tone: Optional[str] = Form(None),
    style: Optional[str] = Form(None),
    muses: Optional[str] = Form(None),
    style_keywords: Optional[str] = Form(None)
):
    """
    Update a doll's details (gender, size, skin tone, style) - regenerates the doll.
    Optional fields only - pass whatever you want to change.
    """
    try:
        gender_val = gender or "neutral"
        size_val = size or "average"
        skin_val = skin_tone or "medium"
        style_val = style or "watercolor"

        muse_text = ""
        keyword_text = ""
        if muses:
            muse_list = [m.strip() for m in muses.split(",") if m.strip()]
            muse_text = "Inspired by muses: " + ", ".join(muse_list) + ". "
        if style_keywords:
            kw_list = [k.strip() for k in style_keywords.split(",") if k.strip()]
            keyword_text = "Style keywords: " + ", ".join(kw_list) + ". "

        style_map = {
            "watercolor": "soft pastel watercolor fashion sketch",
            "bold_pop": "bold cartoon pop-art style",
            "runway_sketch": "high-fashion runway designer sketch",
            "bold pop": "bold cartoon pop-art style",
            "runway sketch": "high-fashion runway designer sketch"
        }

        prompt = (
            f"A fashion doll illustration of {gender_val}, {size_val}, {skin_val}, "
            f"drawn in {style_map.get(style_val, style_val)}. {muse_text}{keyword_text} "
            "Front-facing, neutral pose, full body, transparent background."
        )

        client = get_openai_client()
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=IMAGE_SIZE,
            n=1
        )
        b64 = response.data[0].b64_json
        png_bytes = base64.b64decode(b64)
        url = _save_bytes(png_bytes, subdir="dolls")
        return {"updated_doll_url": url, "prompt": prompt}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- items restyle/update/bulk ----------
@app.post("/items/restyle")
async def restyle_item(
    description: str = Form(...),
    new_style: str = Form(...)
):
    """
    Redraw an existing clothing item in a new art style.
    """
    try:
        style_map = {
            "watercolor": "soft pastel watercolor fashion sketch",
            "bold_pop": "bold colorful cartoon pop, thick black outlines, vivid palette",
            "runway_sketch": "high-fashion runway designer sketch"
        }

        prompt = f"A clothing illustration of {description}, drawn in {style_map.get(new_style, new_style)}."

        client = get_openai_client()
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=IMAGE_SIZE,
            n=1
        )
        b64 = response.data[0].b64_json
        png_bytes = base64.b64decode(b64)
        url = _save_bytes(png_bytes, subdir="items")
        return {"restyled_item_url": url, "prompt": prompt}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/items/update")
async def update_item(
    new_description: str = Form(...),
    style: str = Form(...)
):
    """
    Update a clothing item by re-describing it (e.g. fixing details) 
    and regenerating the illustration in the chosen style.
    """
    try:
        style_map = {
            "watercolor": "soft pastel watercolor fashion sketch",
            "bold_pop": "bold colorful cartoon pop, thick black outlines, vivid palette",
            "runway_sketch": "high-fashion runway designer sketch"
        }

        prompt = f"A clothing illustration of {new_description}, drawn in {style_map.get(style, style)}."

        client = get_openai_client()
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=IMAGE_SIZE,
            n=1
        )
        b64 = response.data[0].b64_json
        png_bytes = base64.b64decode(b64)
        url = _save_bytes(png_bytes, subdir="items")
        return {"updated_item_url": url, "prompt": prompt}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/items/restyle/bulk")
async def bulk_restyle_items(
    items: List[str] = Form(...),  # list of descriptions of all clothing items
    new_style: str = Form(...)
):
    """
    Redraw multiple clothing items in a new art style.
    Returns a list of URLs for all restyled items.
    """
    try:
        style_map = {
            "watercolor": "soft pastel watercolor fashion sketch",
            "bold_pop": "bold colorful cartoon pop, thick black outlines, vivid palette",
            "runway_sketch": "high-fashion runway designer sketch"
        }

        restyled_urls = []
        client = get_openai_client()
        for description in items:
            prompt = f"A clothing illustration of {description}, drawn in {style_map.get(new_style, new_style)}."
            response = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size=IMAGE_SIZE,
                n=1
            )
            b64 = response.data[0].b64_json
            restyled_urls.append(_save_bytes(base64.b64decode(b64), subdir="items"))

        return {"restyled_items_urls": restyled_urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- NEW: generate outfit (fashion-aware) ----------
@app.post("/generate_outfit")
def generate_outfit(request: OutfitRequest):
    """
    Fashion-aware outfit generator.
    Inputs: wardrobe (items), doll (preferably includes image_url), art_style,
            muses (list), style_keywords (list), occasion, weather.
    Steps:
      1) Build a prompt for the text model asking it to pick top/bottom/shoes/accessories.
      2) Try to parse the model's JSON response into chosen items.
      3) If the model fails, fall back to deterministic selection.
      4) Ensure each chosen item has an illustration (generate from description if missing).
      5) Compose items on the doll (use existing compose function) and return final URL + selection.
    """
    # Validate wardrobe
    if not request.wardrobe:
        raise HTTPException(status_code=400, detail="Wardrobe cannot be empty")

    # Build selection prompt
    lines = [
        "You are a professional fashion stylist. Given the user's wardrobe and style preferences,",
        "select a cohesive outfit that matches the requested occasion and weather.",
    ]
    if request.muses:
        lines.append("Muses: " + ", ".join(request.muses) + ".")
    if request.style_keywords:
        lines.append("Style keywords: " + ", ".join(request.style_keywords) + ".")
    if request.occasion:
        lines.append("Occasion: " + request.occasion + ".")
    if request.weather:
        lines.append("Weather: " + request.weather + ".")
    lines.append("Available wardrobe items (category: description):")
    for it in request.wardrobe:
        lines.append(f"- {it.category}: {it.description}")
    lines.append("Choose a top (if applicable), bottom (if applicable), shoes, and up to two accessories.")
    lines.append("Return the result as strict JSON with keys: top, bottom, shoes, accessories (array).")
    lines.append("Each value should be the exact wardrobe item description as given above.")

    selection_prompt = "\n".join(lines)

    # Attempt text model selection
    chosen = None
    try:
        selection_text = _openai_text_generate(selection_prompt, max_tokens=300)
        # attempt to extract JSON
        json_start = selection_text.find("{")
        if json_start != -1:
            json_part = selection_text[json_start:]
            chosen = json.loads(json_part)
        else:
            # if the model returned only JSON-like structure or plain JSON
            chosen = json.loads(selection_text)
    except HTTPException:
        # OpenAI text failed (or returned error). We'll fallback downstream.
        chosen = None
    except Exception:
        chosen = None

    # Deterministic fallback selector if chosen is None or incomplete
    def deterministic_selector(wardrobe_list):
        sel = {"top": None, "bottom": None, "shoes": None, "accessories": []}
        for it in wardrobe_list:
            cat = it.category.lower()
            desc = it.description
            if not sel["top"] and ("top" in cat or "blouse" in cat or "shirt" in cat or "tee" in cat):
                sel["top"] = desc
            if not sel["bottom"] and ("bottom" in cat or "skirt" in cat or "jean" in cat or "trouser" in cat or "dress" in cat):
                # treat a dress as bottom if no separate top exists (fallback)
                sel["bottom"] = desc
            if not sel["shoes"] and ("shoe" in cat or "boot" in cat or "sandal" in cat):
                sel["shoes"] = desc
        # accessories: first up to 2 items with accessory in category or small bags/jewelry
        for it in wardrobe_list:
            if len(sel["accessories"]) >= 2:
                break
            cat = it.category.lower()
            if "accessor" in cat or "bag" in cat or "jewel" in cat or "earring" in cat or "scarf" in cat:
                sel["accessories"].append(it.description)
        return sel

    if not chosen or not isinstance(chosen, dict):
        chosen = deterministic_selector(request.wardrobe)

    # Helper to find the wardrobe item by description fuzzy-match
    def find_item_by_description(desc: Optional[str]):
        if not desc:
            return None
        desc_norm = desc.strip().lower()
        # direct exact match first
        for it in request.wardrobe:
            if it.description.strip().lower() == desc_norm:
                return it
        # substring match
        for it in request.wardrobe:
            if desc_norm in it.description.strip().lower():
                return it
        # keyword match: first token
        first = desc_norm.split()[0]
        for it in request.wardrobe:
            if first in it.description.strip().lower():
                return it
        return None

    selected_items = {}
    for key in ("top", "bottom", "shoes"):
        desc = chosen.get(key) if isinstance(chosen, dict) else None
        selected_items[key] = find_item_by_description(desc)

    accessories_items = []
    if isinstance(chosen, dict) and chosen.get("accessories"):
        for acc_desc in chosen.get("accessories"):
            found = find_item_by_description(acc_desc)
            if found:
                accessories_items.append(found)
    selected_items["accessories"] = accessories_items

    # Ensure each selected item has an image_url (if not, generate using item description)
    generated_item_urls = []

    style_map = {
        "watercolor": "soft pastel watercolor fashion sketch",
        "bold_pop": "bold colorful cartoon pop, thick black outlines, vivid palette",
        "runway_sketch": "high-fashion runway sketch, loose ink lines and minimal washes",
        "bold pop": "bold colorful cartoon pop, thick black outlines, vivid palette",
        "runway sketch": "high-fashion runway sketch, loose ink lines and minimal washes"
    }

    art_phrase = style_map.get(request.art_style.lower(), "soft pastel watercolor")

    for key in ("top", "bottom", "shoes"):
        item = selected_items.get(key)
        if item:
            if item.image_url:
                generated_item_urls.append(item.image_url)
            else:
                prompt = f"Standalone fashion illustration of {item.description}. Transparent background, front-facing flat item for a digital closet, {art_phrase}. High resolution and clean edges, no text, no logos."
                try:
                    png_bytes = _openai_images_generate(prompt)
                    url = _save_bytes(png_bytes, subdir="items")
                    generated_item_urls.append(url)
                    # update the item object's image_url for returning to client
                    item.image_url = url
                except Exception:
                    # ignore generation failure for that item
                    pass

    # accessories
    for acc in accessories_items:
        if acc.image_url:
            generated_item_urls.append(acc.image_url)
        else:
            prompt = f"Standalone fashion illustration of {acc.description}. Transparent background, front-facing flat item for a digital closet, {art_phrase}. High resolution and clean edges, no text, no logos."
            try:
                png_bytes = _openai_images_generate(prompt)
                url = _save_bytes(png_bytes, subdir="items")
                generated_item_urls.append(url)
                acc.image_url = url
            except Exception:
                pass

    # Prepare doll image (prefer image_url in request.doll)
    doll_image_url = request.doll.get("image_url") if isinstance(request.doll, dict) else None
    if not doll_image_url:
        # generate a neutral doll if client didn't provide one
        doll_prompt = f"A neutral fashion doll, front-facing, full-body, transparent background, drawn in {request.art_style} style."
        try:
            png_bytes = _openai_images_generate(doll_prompt)
            doll_image_url = _save_bytes(png_bytes, subdir="dolls")
        except Exception:
            raise HTTPException(status_code=500, detail="Could not obtain or generate doll image")

    # Download doll bytes
    try:
        doll_bytes = _download_to_bytes(doll_image_url) if doll_image_url.startswith("http") else open(doll_image_url, "rb").read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not load doll image for composition")

    # Download layer images for composition
    layer_bytes = []
    for url in generated_item_urls:
        try:
            if url.startswith("http"):
                layer_bytes.append(_download_to_bytes(url))
            else:
                with open(url, "rb") as f:
                    layer_bytes.append(f.read())
        except Exception:
            continue

    # Compose final outfit image
    try:
        composed_png = _compose_layers(doll_bytes, layer_bytes)
        composed_url = _save_bytes(composed_png, subdir="outfits")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Composition failed: {e}")

    # Build response selected item dicts (convert Pydantic objects to plain dicts if present)
    def item_to_dict(it):
        if not it:
            return None
        if isinstance(it, dict):
            return it
        # WardrobeItem pydantic => .dict()
        try:
            return it.dict()
        except Exception:
            return {"id": getattr(it, "id", None), "category": getattr(it, "category", None), "description": getattr(it, "description", None), "image_url": getattr(it, "image_url", None)}

    resp_selection = {
        "top": item_to_dict(selected_items.get("top")),
        "bottom": item_to_dict(selected_items.get("bottom")),
        "shoes": item_to_dict(selected_items.get("shoes")),
        "accessories": [item_to_dict(a) for a in accessories_items] if accessories_items else []
    }

    return {
        "selected": resp_selection,
        "composed_outfit_url": composed_url,
        "prompt_sent_to_ai": selection_prompt
    }

# ---------- health ping ----------
@app.get("/ping")
async def ping():
    return {"status": "Closh backend is alive!", "version": "0.3.0"}
