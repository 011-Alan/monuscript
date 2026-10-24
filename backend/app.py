"""
MonuScript - Flask Backend Application
Versioned RESTful API under /api/v1 for Indian Monument Recognition & Exploration.
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import Config
from database.db import (
    init_db,
    create_conversation,
    get_conversations,
    get_conversation,
    rename_conversation,
    delete_conversation,
    get_conversation_history
)
from services.monument_service import get_all_monuments, get_monument_by_id
from services.recognition_service import process_recognition_request
from ml.inference import MonumentPredictor

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Enable CORS for frontend integration
    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/uploads/*": {"origins": "*"}})

    # Initialize local SQLite database
    init_db()

    # Pre-warm model predictor if artifact exists
    try:
        MonumentPredictor.get_instance()
    except Exception as e:
        app.logger.warning(f"Model not loaded at startup: {e}")

    # ==================== Health & Info Endpoints ====================

    @app.route("/api/v1/health", methods=["GET"])
    def health_check():
        model_loaded = False
        supported_classes = []
        try:
            predictor = MonumentPredictor.get_instance()
            model_loaded = True
            supported_classes = list(predictor.classes)
        except Exception:
            pass

        return jsonify({
            "status": "ok",
            "model_loaded": model_loaded,
            "supported_classes": len(supported_classes),
            "classes": supported_classes,
            "service": "MonuScript Digital Image Processing Engine"
        }), 200

    # ==================== Monument Metadata Endpoints ====================

    @app.route("/api/v1/monuments", methods=["GET"])
    def list_monuments():
        monuments = get_all_monuments()
        clean_list = []
        for mon_id, data in monuments.items():
            clean_item = dict(data)
            clean_item.pop("reference_images", None)
            clean_list.append(clean_item)
        return jsonify({
            "success": True,
            "count": len(clean_list),
            "monuments": clean_list
        }), 200

    @app.route("/api/v1/monuments/<monument_id>", methods=["GET"])
    def get_monument(monument_id):
        mon = get_monument_by_id(monument_id)
        if not mon:
            return jsonify({
                "success": False,
                "error": {"code": "NOT_FOUND", "message": f"Monument '{monument_id}' not found."}
            }), 404
        return jsonify({"success": True, "monument": mon}), 200

    # ==================== Recognition Endpoints ====================

    @app.route("/api/v1/recognize", methods=["POST"])
    def recognize_single():
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": {"code": "MISSING_IMAGE", "message": "No image file provided in request."}
            }), 400

        file_storage = request.files["image"]
        conversation_id = request.form.get("conversation_id")

        try:
            res = process_recognition_request(file_storage, conversation_id=conversation_id, mode="single")
            status_code = res.pop("status_code", 200)
            return jsonify(res), status_code
        except ValueError as ve:
            return jsonify({
                "success": False,
                "error": {"code": "INVALID_IMAGE", "message": str(ve)}
            }), 422
        except Exception as e:
            return jsonify({
                "success": False,
                "error": {"code": "PROCESSING_ERROR", "message": f"Server error: {str(e)}"}
            }), 500

    @app.route("/api/v1/recognize/multiple", methods=["POST"])
    def recognize_multiple():
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": {"code": "MISSING_IMAGE", "message": "No image file provided in request."}
            }), 400

        file_storage = request.files["image"]
        conversation_id = request.form.get("conversation_id")

        try:
            res = process_recognition_request(file_storage, conversation_id=conversation_id, mode="multiple")
            status_code = res.pop("status_code", 200)
            return jsonify(res), status_code
        except ValueError as ve:
            return jsonify({
                "success": False,
                "error": {"code": "INVALID_IMAGE", "message": str(ve)}
            }), 422
        except Exception as e:
            return jsonify({
                "success": False,
                "error": {"code": "PROCESSING_ERROR", "message": f"Server error: {str(e)}"}
            }), 500

    # ==================== Conversation & Chat Endpoints ====================

    @app.route("/api/v1/conversations", methods=["GET"])
    def list_conversations():
        convs = get_conversations()
        return jsonify({"success": True, "conversations": convs}), 200

    @app.route("/api/v1/conversations", methods=["POST"])
    def create_new_conversation():
        data = request.get_json(silent=True) or {}
        title = data.get("title", "New Exploration")
        conv = create_conversation(title=title)
        return jsonify({"success": True, "conversation": conv}), 201

    @app.route("/api/v1/conversations/<conv_id>", methods=["GET"])
    def get_conversation_details(conv_id):
        conv = get_conversation(conv_id)
        if not conv:
            return jsonify({
                "success": False,
                "error": {"code": "NOT_FOUND", "message": "Conversation not found."}
            }), 404

        history = get_conversation_history(conv_id)
        return jsonify({
            "success": True,
            "conversation": conv,
            "messages": history
        }), 200

    @app.route("/api/v1/conversations/<conv_id>", methods=["PATCH"])
    def update_conversation(conv_id):
        conv = get_conversation(conv_id)
        if not conv:
            return jsonify({
                "success": False,
                "error": {"code": "NOT_FOUND", "message": "Conversation not found."}
            }), 404

        data = request.get_json(silent=True) or {}
        new_title = data.get("title")
        if not new_title or not new_title.strip():
            return jsonify({
                "success": False,
                "error": {"code": "BAD_REQUEST", "message": "Title cannot be empty."}
            }), 400

        updated = rename_conversation(conv_id, new_title.strip())
        return jsonify({"success": True, "conversation": updated}), 200

    @app.route("/api/v1/conversations/<conv_id>", methods=["DELETE"])
    def remove_conversation(conv_id):
        conv = get_conversation(conv_id)
        if not conv:
            return jsonify({
                "success": False,
                "error": {"code": "NOT_FOUND", "message": "Conversation not found."}
            }), 404

        delete_conversation(conv_id)
        return jsonify({"success": True, "message": "Conversation deleted successfully."}), 200

    # ==================== Upload Static Serving ====================

    @app.route("/uploads/<filename>", methods=["GET"])
    def serve_upload(filename):
        return send_from_directory(Config.UPLOAD_FOLDER, filename)

    # ==================== Error Handlers ====================

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return jsonify({
            "success": False,
            "error": {
                "code": "FILE_TOO_LARGE",
                "message": f"Uploaded file exceeds maximum limit ({Config.MAX_CONTENT_LENGTH // (1024*1024)} MB)."
            }
        }), 413

    @app.errorhandler(404)
    def resource_not_found(e):
        return jsonify({
            "success": False,
            "error": {"code": "NOT_FOUND", "message": "Endpoint not found."}
        }), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify({
            "success": False,
            "error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred."}
        }), 500

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=True)
