"""
MonuScript - Inference & Recognition Service
Loads trained SVM pipeline and performs single-monument and multi-monument candidate recognition.
"""

import sys
import time
from pathlib import Path
import numpy as np
import joblib
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from image_processing.feature_extractor import extract_features, extract_features_from_preprocessed, FEATURE_GROUPS
from image_processing.preprocessing import preprocess_image_for_features, open_and_normalize_image
from ml.rejection import RejectionJudge
from services.monument_service import get_monument_by_id

ARTIFACTS_DIR = BACKEND_DIR / "ml" / "model_artifacts"
MODEL_PIPELINE_PATH = ARTIFACTS_DIR / "svm_pipeline.joblib"

class MonumentPredictor:
    _instance = None

    def __init__(self, model_path=MODEL_PIPELINE_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. "
                "Please run python backend/ml/train_model.py first."
            )
        self.pipeline = joblib.load(model_path)
        self.scaler = self.pipeline["scaler"]
        self.classifier = self.pipeline["classifier"]
        self.classes = self.pipeline["classes"]
        self.use_weights = self.pipeline.get("use_feature_weights", True)
        self.class_centroids = self.pipeline.get("class_centroids")
        self.feature_groups = self.pipeline.get("feature_groups", FEATURE_GROUPS)

        self.use_clahe = self.pipeline.get("use_clahe", False)
        self.clahe_clip_limit = self.pipeline.get("clahe_clip_limit", 2.0)
        self.clahe_grid_size = tuple(self.pipeline.get("clahe_grid_size", (8, 8)))
        self.include_wavelet = self.pipeline.get("include_wavelet", False)
        self.include_shape = self.pipeline.get("include_shape", False)
        self.include_architectural = self.pipeline.get("include_architectural", False)

        rejection_cfg = self.pipeline.get("rejection_config", {})
        self.judge = RejectionJudge(
            accept_threshold=rejection_cfg.get("accept_threshold", 0.40),
            margin_threshold=rejection_cfg.get("margin_threshold", 0.12),
            uncertain_threshold=rejection_cfg.get("uncertain_threshold", 0.28),
            class_centroids=self.class_centroids
        )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _apply_weights(self, X):
        if not self.use_weights:
            return X
        X_weighted = X.copy()
        for group_name, info in self.feature_groups.items():
            s = info["start"]
            e = info["end"]
            w = info["default_weight"]
            scale_factor = np.sqrt(w)
            X_weighted[:, s:e] = X_weighted[:, s:e] * scale_factor
        return X_weighted

    def predict_image(self, image_input):
        """
        Runs complete inference pipeline on an uploaded image.
        Returns:
            dict containing:
                success: bool
                accepted: bool
                is_uncertain: bool
                message: str
                results: list of monument result objects
                top_class: str
                top_score: float
                processing_time_ms: float
                image_info: dict
                feature_summary: dict (for educational explanation of feature groups)
                diagnostics: dict (rich debug information for developer feature inspection)
        """
        t0 = time.time()

        # 1. Feature extraction & Preprocessing
        feats, prep = extract_features(
            image_input,
            use_clahe=self.use_clahe,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_grid_size=self.clahe_grid_size,
            include_wavelet=self.include_wavelet,
            include_shape=self.include_shape,
            include_architectural=self.include_architectural
        )
        orig_w, orig_h = prep["original_size"]

        # 2. Scaling & Weighting with clipping to [-5.0, 5.0] to prevent outlier centroid explosions
        scaled_feats = np.clip(self.scaler.transform(feats.reshape(1, -1)), -5.0, 5.0)
        proc_feats = self._apply_weights(scaled_feats)

        # 3. Model Probabilities & Classification
        probs = self.classifier.predict_proba(proc_feats)[0]

        # 4. Confidence & Rejection Assessment
        decision = self.judge.evaluate(probs, self.classes, scaled_features=proc_feats[0])

        elapsed_ms = round((time.time() - t0) * 1000.0, 1)

        # Prepare monument results
        results = []
        if decision["accepted"]:
            mon_id = decision["top_class"]
            meta = get_monument_by_id(mon_id)
            results.append({
                "monument_id": mon_id,
                "name": meta.get("name", mon_id.replace("_", " ").title()),
                "score": round(float(decision["top_score"]), 3),
                "score_type": "calibrated_score",
                "location": meta.get("location", ""),
                "description": meta.get("description", ""),
                "historical_details": meta.get("historical_details", []),
                "key_features": meta.get("key_features", []),
                "architectural_style": meta.get("architectural_style", ""),
                "built_in": meta.get("built_in", ""),
                "builder": meta.get("builder", ""),
                "model_url": meta.get("model_url")
            })

        # Summary of feature contribution for transparent academic inspection
        feature_summary = {
            "total_features": len(feats),
            "groups": {
                g: {
                    "count": info["length"],
                    "description": info["description"],
                    "weight_applied": info["default_weight"] if self.use_weights else 1.0
                }
                for g, info in self.feature_groups.items()
            }
        }

        # Developer / debug diagnostic inspection metadata (Requirement 8)
        diagnostics = {
            "model_version": "2.0-dip-svm-architectural",
            "model_type": f"Support Vector Machine ({getattr(self.classifier, 'kernel', 'rbf').upper()} kernel, C={getattr(self.classifier, 'C', 0.2)})",
            "probability_calibration_method": "Platt Scaling (logistic sigmoid on SVM decision margins)",
            "predicted_class": decision["top_class"],
            "top_score": round(float(decision["top_score"]), 3),
            "competing_candidates": [
                {"class": self.classes[i], "score": round(float(probs[i]), 3)}
                for i in np.argsort(probs)[::-1][:3]
            ],
            "confidence_margin": round(float(decision["score_margin"]), 3),
            "rejection_thresholds": {
                "accept_threshold": self.judge.accept_threshold,
                "margin_threshold": self.judge.margin_threshold,
                "uncertain_threshold": self.judge.uncertain_threshold,
                "max_centroid_distance": self.judge.max_centroid_distance
            },
            "centroid_distance": round(decision["centroid_distance"], 2) if decision["centroid_distance"] is not None else None,
            "rejection_reason": decision["rejection_reason"],
            "preprocessing_pipeline": {
                "clahe_applied": self.use_clahe,
                "clahe_clip_limit": self.clahe_clip_limit,
                "color_space": "CIE LAB Luminance + Syllabus ITU-R BT.601 Grayscale",
                "target_resolution": "256x256"
            },
            "feature_space": {
                "total_dimensions": len(feats),
                "z_score_clipped": True,
                "clip_range": "[-5.0, 5.0]"
            }
        }

        return {
            "success": True,
            "accepted": decision["accepted"],
            "is_uncertain": decision["is_uncertain"],
            "message": decision["message"],
            "results": results,
            "top_class": decision["top_class"],
            "top_score": round(float(decision["top_score"]), 3),
            "runner_up_class": decision["runner_up_class"],
            "runner_up_score": round(float(decision["runner_up_score"]), 3) if decision["runner_up_score"] else None,
            "processing_time_ms": elapsed_ms,
            "image": {
                "width": orig_w,
                "height": orig_h
            },
            "feature_summary": feature_summary,
            "diagnostics": diagnostics
        }

    def predict_multiple_candidates(self, image_input, max_regions=16):
        """
        Sliding-window / sub-region multi-candidate detection for composite and collage photographs.
        Evaluates candidate spatial sub-regions:
        - Full image
        - Left / Right vertical split halves
        - Top / Bottom horizontal split halves
        - 2x2 Quadrants (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
        - 3-Column panels (Left, Center, Right)
        - Central focal window
        Merges and deduplicates distinct accepted monument classes.
        """
        t0 = time.time()
        pil_img = open_and_normalize_image(image_input)
        w, h = pil_img.size

        regions = [
            ("Whole Image", (0, 0, w, h)),
            ("Left Half", (0, 0, int(w * 0.52), h)),
            ("Right Half", (int(w * 0.48), 0, w, h)),
            ("Top Half", (0, 0, w, int(h * 0.52))),
            ("Bottom Half", (0, int(h * 0.48), w, h)),
            ("Top-Left Quadrant", (0, 0, int(w * 0.52), int(h * 0.52))),
            ("Top-Right Quadrant", (int(w * 0.48), 0, w, int(h * 0.52))),
            ("Bottom-Left Quadrant", (0, int(h * 0.48), int(w * 0.52), h)),
            ("Bottom-Right Quadrant", (int(w * 0.48), int(h * 0.48), w, h)),
            ("Center Region", (int(w * 0.18), int(h * 0.18), int(w * 0.82), int(h * 0.82))),
            ("Left Third", (0, 0, int(w * 0.38), h)),
            ("Mid Column", (int(w * 0.31), 0, int(w * 0.69), h)),
            ("Right Third", (int(w * 0.62), 0, w, h)),
            ("Top Third", (0, 0, w, int(h * 0.38))),
            ("Bottom Third", (0, int(h * 0.62), w, h)),
        ]

        seen_monuments = {}

        for region_name, box in regions[:max_regions]:
            crop_img = pil_img.crop(box)
            crop_w, crop_h = crop_img.size
            if crop_w < 60 or crop_h < 60:
                continue

            feats, _ = extract_features(
                crop_img,
                use_clahe=self.use_clahe,
                clahe_clip_limit=self.clahe_clip_limit,
                clahe_grid_size=self.clahe_grid_size,
                include_wavelet=self.include_wavelet,
                include_shape=self.include_shape,
                include_architectural=self.include_architectural
            )
            scaled_feats = np.clip(self.scaler.transform(feats.reshape(1, -1)), -5.0, 5.0)
            proc_feats = self._apply_weights(scaled_feats)
            probs = self.classifier.predict_proba(proc_feats)[0]
            decision = self.judge.evaluate(probs, self.classes, scaled_features=proc_feats[0])

            if decision["accepted"]:
                mon_id = decision["top_class"]
                score = round(float(decision["top_score"]), 3)
                if mon_id not in seen_monuments or score > seen_monuments[mon_id]["score"]:
                    meta = get_monument_by_id(mon_id)
                    seen_monuments[mon_id] = {
                        "monument_id": mon_id,
                        "name": meta.get("name", mon_id.replace("_", " ").title()),
                        "score": score,
                        "score_type": "region_score",
                        "location": meta.get("location", ""),
                        "description": meta.get("description", ""),
                        "historical_details": meta.get("historical_details", []),
                        "model_url": meta.get("model_url"),
                        "detected_region": region_name,
                        "bounding_box": [box[0], box[1], box[2], box[3]]
                    }

        elapsed_ms = round((time.time() - t0) * 1000.0, 1)
        results = sorted(seen_monuments.values(), key=lambda r: r["score"], reverse=True)

        accepted = len(results) > 0
        message = (
            f"Detected {len(results)} distinct monument candidate(s)."
            if accepted else
            "No supported monument candidates met the acceptance threshold in candidate regions."
        )

        return {
            "success": True,
            "accepted": accepted,
            "message": message,
            "results": results,
            "distinct_count": len(results),
            "processing_time_ms": elapsed_ms,
            "image": {
                "width": w,
                "height": h
            }
        }
