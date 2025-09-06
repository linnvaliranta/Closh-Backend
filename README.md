# Closh Backend (FastAPI)

Quickstart (local):
1. Create and activate venv:
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate

2. Install:
   pip install -r requirements.txt

3. Copy .env.example -> .env and fill keys (OPENAI_API_KEY ...)

4. Run:
   uvicorn app:app --reload --port 8000

Endpoints:
- POST /items/text  {description, illustration_style, category} -> {image_url}
- POST /items/photo (form-data: file, illustration_style, category) -> {image_url}
- POST /outfits/compose {doll_image_path, item_image_paths} -> {composed_url}
