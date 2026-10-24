"""
MonuScript - Monument Service
Manages validated monument metadata loaded from backend/data/monuments.json
"""

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "monuments.json"

_monuments_cache = None

def load_monuments():
    global _monuments_cache
    if _monuments_cache is None:
        if not DATA_PATH.exists():
            return {}
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            _monuments_cache = json.load(f)
    return _monuments_cache

def get_all_monuments():
    return load_monuments()

def get_monument_by_id(monument_id):
    monuments = load_monuments()
    return monuments.get(monument_id, {
        "id": monument_id,
        "name": monument_id.replace("_", " ").title(),
        "location": "India",
        "description": "Historical Indian monument.",
        "historical_details": [],
        "key_features": [],
        "architectural_style": "Historical Indian Architecture",
        "built_in": "Historical Era",
        "builder": "Historical Records",
        "model_url": None,
        "supported": True
    })
