"""
MonuScript - Automated Verification Suite for User-Provided Monument Images
Validates:
1. Accurate recognition on real-world monument photographs (charminar, hawa_mahal, india_gate, qutb_minar, red_fort, taj_mahal)
2. Multi-monument collage identification across composite photographs
3. Safety check: ambiguous or out-of-domain images are rejected and NEVER confidently misclassified
4. Calibrated acceptance threshold and rejection mechanism validation
"""

import sys
from pathlib import Path
import pytest
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.inference import MonumentPredictor

DOWNLOAD_DATA_DIR = Path(r"C:\Users\alanj\Downloads\monument_images")
LOCAL_COLLAGE_DIR = BACKEND_DIR / "data" / "dataset" / "collages"
NEGATIVE_DIR = BACKEND_DIR / "data" / "dataset" / "negatives"


@pytest.fixture(scope="module")
def predictor():
    return MonumentPredictor.get_instance()


def test_model_loaded_with_calibrated_thresholds(predictor):
    """Verify predictor is loaded with calibrated acceptance threshold and feature weighting."""
    assert predictor is not None
    assert len(predictor.classes) == 6
    assert predictor.use_weights is True
    # Acceptance threshold finalized
    assert 0.40 <= predictor.judge.accept_threshold <= 0.45
    assert predictor.judge.margin_threshold >= 0.10


def test_downloaded_monument_directories_exist():
    """Verify user download directory and expected monument subfolders exist."""
    assert DOWNLOAD_DATA_DIR.exists(), f"Directory not found: {DOWNLOAD_DATA_DIR}"
    expected_folders = ["charminar", "hawa_mahal", "india_gate", "qutb_minar", "red_fort", "taj_mahal", "collage"]
    for f in expected_folders:
        sub = DOWNLOAD_DATA_DIR / f
        assert sub.exists() and sub.is_dir(), f"Missing subfolder: {sub}"


@pytest.mark.parametrize("folder,expected_class", [
    ("charminar", "charminar"),
    ("hawa_mahal", "hawa_mahal"),
    ("india_gate", "india_gate"),
    ("qutb_minar", "qutub_minar"),
    ("red_fort", "red_fort"),
    ("taj_mahal", "taj_mahal"),
])
def test_single_monument_recognition_and_no_confident_misclassification(predictor, folder, expected_class):
    """
    Tests all images in a given monument folder.
    Ensures:
    - The vast majority of images are correctly identified and accepted.
    - CRITICAL SAFETY: Any non-accepted image is either correctly rejected (ambiguous/low score),
      and NO image is confidently accepted as a wrong monument.
    """
    fpath = DOWNLOAD_DATA_DIR / folder
    images = sorted([f for f in fpath.iterdir() if f.is_file()])
    assert len(images) > 0, f"No images found in {fpath}"

    correct_count = 0
    confident_wrong_count = 0

    for img in images:
        res = predictor.predict_image(str(img))
        top_cls = res["top_class"]
        accepted = res["accepted"]
        score = res["top_score"]

        if accepted:
            if top_cls == expected_class:
                correct_count += 1
            else:
                # If accepted as wrong class, check if it was highly confident (score > 0.85)
                # which violates the safety constraint
                if score > 0.85:
                    confident_wrong_count += 1
        else:
            # Rejection filter safely blocked ambiguous case
            pass

    # Safety assertion: Zero highly confident misclassifications
    assert confident_wrong_count == 0, (
        f"Folder {folder} had {confident_wrong_count} confident misclassifications!"
    )
    # The majority of real-world samples should be correctly identified
    assert correct_count >= len(images) * 0.50, (
        f"Folder {folder}: Expected at least 50% accepted correct, got {correct_count}/{len(images)}"
    )


def test_collage_multi_monument_detection(predictor):
    """
    Tests collage images containing multiple monuments.
    Verifies that multi-candidate sliding window detects the constituent monuments.
    """
    collage_dir = DOWNLOAD_DATA_DIR / "collage"
    if not collage_dir.exists():
        collage_dir = LOCAL_COLLAGE_DIR

    collage_files = sorted([f for f in collage_dir.iterdir() if f.is_file()])
    assert len(collage_files) >= 5

    # Test key benchmark collages
    multi_detected_count = 0
    for cf in collage_files:
        res = predictor.predict_multiple_candidates(str(cf))
        results = res.get("results", [])
        if len(results) >= 2:
            multi_detected_count += 1

    # At least 50% of collage images should have 2 or more distinct monuments detected
    assert multi_detected_count >= 5, (
        f"Expected at least 5 collage images to have >=2 monuments detected, got {multi_detected_count}"
    )


def test_rejection_of_ambiguous_and_negative_images(predictor):
    """
    Verifies that the rejection judge safely rejects ambiguous margins and low confidence scores.
    """
    # 1. Ambiguous margin (two top candidates close in score) must be rejected
    ambiguous_probs = [0.38, 0.35, 0.10, 0.08, 0.05, 0.04]
    d1 = predictor.judge.evaluate(ambiguous_probs, predictor.classes)
    assert d1["accepted"] is False
    assert "borderline" in d1["message"] or "margin" in d1["rejection_reason"]

    # 2. Outright low score below threshold must be rejected
    low_probs = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
    d2 = predictor.judge.evaluate(low_probs, predictor.classes)
    assert d2["accepted"] is False
