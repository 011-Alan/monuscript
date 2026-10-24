"""
MonuScript - Backend Configuration
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "monuscript-academic-secret-key-2026")
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
    MAX_IMAGE_DIMENSION = 4096             # Max width or height in pixels
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"}
    MODEL_ARTIFACTS_DIR = BASE_DIR / "ml" / "model_artifacts"
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
    PORT = int(os.getenv("PORT", 5000))
    HOST = os.getenv("HOST", "0.0.0.0")

Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
