# app.py
import os
import io
import uuid
import base64
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
app = FastAPI(title="Closh API", version="0.2.0")

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

def _openai_images_generate(prompt: str) -> bytes:
    """Call OpenAI Images (gpt-image-1) and return PNG bytes."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

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

# -------- Routes --------
@app.get("/")
def root():
    return {"status": "ok", "version": "0.2.0"}

@app.post("/items/text", response_model=TextItemOut)
def generate_item_from_text(payload: TextItemIn):
    style_map = {
        "watercolor": "soft pastel watercolor, delicate brush strokes",
        "bold pop": "bold colorful cartoon pop, thick black outlines, vivid palette",
        "runway sketch": "high-fashion runway sketch, loose ink lines and minimal washes",
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
    raw = await file.read()
    cutout = _remove_bg(raw)
    style_map = {
        "watercolor": "soft pastel watercolor, delicate brush strokes",
        "bold pop": "bold colorful cartoon pop, thick black outlines, vivid palette",
        "runway sketch": "high-fashion runway sketch, loose ink lines and minimal washes",
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
        # Map input values into a nice natural prompt
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

        # Call OpenAI image generation
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="512x512"
        )

        # Upload to S3 (optional, if you’ve set it up)
        image_url = response.data[0].url
        return {"doll_url": image_url, "prompt": prompt}

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

        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="512x512"
        )

        image_url = response.data[0].url
        return {"restyled_doll_url": image_url, "prompt": prompt}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        @app.post("/doll/update")
async def update_doll(
    gender: str = Form(...),
    size: str = Form(...),
    skin_tone: str = Form(...),
    style: str = Form(...)
):
    """
    Update a doll's details (gender, size, skin tone, and style).
    Essentially the same as creating a new doll, but meant for editing an existing profile doll.
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

        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="512x512"
        )

        image_url = response.data[0].url
        return {"updated_doll_url": image_url, "prompt": prompt}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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
            "bold_pop": "bold cartoon pop-art style",
            "runway_sketch": "high-fashion runway designer sketch"
        }

        prompt = f"A clothing illustration of {description}, drawn in {style_map.get(new_style, new_style)}."

        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="512x512"
        )

        image_url = response.data[0].url
        return {"restyled_item_url": image_url, "prompt": prompt}

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
            "bold_pop": "bold cartoon pop-art style",
            "runway_sketch": "high-fashion runway designer sketch"
        }

        prompt = f"A clothing illustration of {new_description}, drawn in {style_map.get(style, style)}."

        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="512x512"
        )

        image_url = response.data[0].url
        return {"updated_item_url": image_url, "prompt": prompt}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        from typing import List

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
            "bold_pop": "bold cartoon pop-art style",
            "runway_sketch": "high-fashion runway designer sketch"
        }

        restyled_urls = []
        for description in items:
            prompt = f"A clothing illustration of {description}, drawn in {style_map.get(new_style, new_style)}."

            response = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="512x512"
            )
            restyled_urls.append(response.data[0].url)

        return {"restyled_items_urls": restyled_urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
