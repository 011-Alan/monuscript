"""
MonuScript - Recognition Service
Orchestrates file upload validation, safety checks, inference execution,
and database persistence.
"""

import os
import uuid
from pathlib import Path
from PIL import Image

from config import Config
from ml.inference import MonumentPredictor
from database.db import add_message, add_recognition_result

def is_allowed_file(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in Config.ALLOWED_EXTENSIONS

def validate_and_save_upload(file_storage):
    """
    Validates uploaded file:
    - Extension check
    - File size check
    - Image decode & integrity check
    - Dimension sanity check
    Saves safely with UUID-based name in Config.UPLOAD_FOLDER.
    Returns:
        saved_path, (width, height), mime_type
    Raises:
        ValueError for invalid/corrupted inputs
    """
    if not file_storage or not file_storage.filename:
        raise ValueError("No file provided in the request.")

    filename = file_storage.filename
    if not is_allowed_file(filename):
        raise ValueError("That file format isn't supported yet. Please upload a valid image in a supported format (JPEG, PNG, WEBP, BMP, TIFF).")

    ext = filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    target_path = Config.UPLOAD_FOLDER / unique_name

    file_storage.save(target_path)

    # Validate image content with Pillow
    try:
        with Image.open(target_path) as img:
            img.verify()
        with Image.open(target_path) as img:
            w, h = img.size
            if w > Config.MAX_IMAGE_DIMENSION or h > Config.MAX_IMAGE_DIMENSION:
                target_path.unlink(missing_ok=True)
                raise ValueError(f"Image dimensions ({w}x{h}) exceed maximum allowed dimension ({Config.MAX_IMAGE_DIMENSION}px).")
            if w < 32 or h < 32:
                target_path.unlink(missing_ok=True)
                raise ValueError("Image is too small to extract reliable architectural features.")
            return target_path, (w, h), ext
    except Exception as e:
        if target_path.exists():
            target_path.unlink(missing_ok=True)
        if "Image dimensions" in str(e) or "too small" in str(e):
            raise
        raise ValueError("I couldn't open this image. Please try another copy of the file.")

def process_recognition_request(file_storage, conversation_id=None, mode="single"):
    """
    Handles end-to-end recognition flow for an uploaded file.
    If conversation_id is provided, persists user message and assistant response in DB.
    """
    saved_path, (w, h), ext = validate_and_save_upload(file_storage)
    image_web_url = f"/uploads/{saved_path.name}"

    try:
        predictor = MonumentPredictor.get_instance()
    except FileNotFoundError as e:
        return {
            "success": False,
            "status_code": 503,
            "error": {
                "code": "MODEL_UNAVAILABLE",
                "message": "The recognition model hasn't been prepared yet. Please complete the model setup described in the project documentation."
            }
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 500,
            "error": {
                "code": "INFERENCE_ERROR",
                "message": f"An error occurred while loading the recognition pipeline: {str(e)}"
            }
        }

    # Run single or multiple candidate inference
    if mode == "multiple":
        rec_data = predictor.predict_multiple_candidates(str(saved_path))
    else:
        rec_data = predictor.predict_image(str(saved_path))

    rec_data["image"]["filename"] = saved_path.name
    rec_data["image"]["url"] = image_web_url

    # Persist in DB if conversation_id provided
    if conversation_id:
        # Add user message
        user_msg_id = add_message(
            conv_id=conversation_id,
            role="user",
            content="Uploaded monument photograph for architectural analysis.",
            image_url=image_web_url
        )
        # Add assistant response message
        asst_msg_id = add_message(
            conv_id=conversation_id,
            role="assistant",
            content=rec_data["message"],
            image_url=None
        )

        # Store recognition results
        for item in rec_data.get("results", []):
            add_recognition_result(
                message_id=asst_msg_id,
                monument_id=item["monument_id"],
                score=item["score"],
                score_type=item.get("score_type", "recognition_score"),
                accepted=1,
                metadata_snapshot=item
            )

        # If rejected, persist rejection record as well
        if not rec_data.get("accepted"):
            top_cls = rec_data.get("top_class", "unknown")
            top_sc = rec_data.get("top_score", 0.0)
            add_recognition_result(
                message_id=asst_msg_id,
                monument_id=top_cls,
                score=top_sc,
                score_type="rejected_candidate",
                accepted=0,
                metadata_snapshot={"reason": rec_data.get("message")}
            )

        rec_data["message_ids"] = {
            "user_message_id": user_msg_id,
            "assistant_message_id": asst_msg_id
        }

    return rec_data
