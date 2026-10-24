"""
Unit Tests for Machine Learning Inference & Rejection
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.inference import MonumentPredictor
from ml.rejection import RejectionJudge

def test_model_artifact_loading():
    """Verify MonumentPredictor successfully loads saved SVM artifact."""
    predictor = MonumentPredictor.get_instance()
    assert predictor.classifier is not None
    assert len(predictor.classes) == 6
    assert "taj_mahal" in predictor.classes

def test_rejection_low_confidence():
    """Verify rejection judge rejects predictions with low confidence."""
    judge = RejectionJudge(accept_threshold=0.50, margin_threshold=0.15)
    # Uniform / flat probabilities (1/6 each)
    flat_probs = np.full(6, 1.0 / 6.0)
    classes = ["a", "b", "c", "d", "e", "f"]
    decision = judge.evaluate(flat_probs, classes)

    assert not decision["accepted"]
    assert "Oh my! Haven't seen this one before!" in decision["message"]

def test_rejection_ambiguous_margin():
    """Verify rejection judge marks close top-1 and top-2 scores as uncertain."""
    judge = RejectionJudge(accept_threshold=0.40, margin_threshold=0.15)
    # Top 1 is 0.42, Top 2 is 0.40 (margin is only 0.02 < 0.15)
    ambiguous_probs = np.array([0.42, 0.40, 0.05, 0.05, 0.04, 0.04])
    classes = ["taj_mahal", "qutub_minar", "red_fort", "india_gate", "charminar", "hawa_mahal"]
    decision = judge.evaluate(ambiguous_probs, classes)

    assert not decision["accepted"]
    assert decision["is_uncertain"]

def test_end_to_end_inference():
    """Verify complete end-to-end inference returns expected structure."""
    predictor = MonumentPredictor.get_instance()
    # Synthetic image for structural test
    synthetic = Image.new("RGB", (200, 200), (220, 220, 240))
    res = predictor.predict_image(synthetic)

    assert res["success"] is True
    assert "accepted" in res
    assert "message" in res
    assert "processing_time_ms" in res
    assert "image" in res
    assert res["image"]["width"] == 200
    assert res["image"]["height"] == 200
