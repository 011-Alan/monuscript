"""
Unit Tests for Digital Image Preprocessing & Transformations
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from image_processing.preprocessing import (
    rgb_to_grayscale_syllabus,
    open_and_normalize_image,
    preprocess_image_for_features,
    STANDARD_WIDTH,
    STANDARD_HEIGHT
)
from image_processing.transformations import (
    apply_brightness_contrast,
    apply_gamma_transformation
)

def test_grayscale_formula_exact_weights():
    """Verify that rgb_to_grayscale_syllabus matches I = 0.299R + 0.587G + 0.114B exactly."""
    # Pure Red: 0.299 * 255 = 76.245 -> 76
    pure_red = np.zeros((2, 2, 3), dtype=np.uint8)
    pure_red[:, :, 0] = 255
    gray_red = rgb_to_grayscale_syllabus(pure_red)
    assert np.all(gray_red == 76)

    # Pure Green: 0.587 * 255 = 149.685 -> 150
    pure_green = np.zeros((2, 2, 3), dtype=np.uint8)
    pure_green[:, :, 1] = 255
    gray_green = rgb_to_grayscale_syllabus(pure_green)
    assert np.all(gray_green == 150)

    # Pure Blue: 0.114 * 255 = 29.07 -> 29
    pure_blue = np.zeros((2, 2, 3), dtype=np.uint8)
    pure_blue[:, :, 2] = 255
    gray_blue = rgb_to_grayscale_syllabus(pure_blue)
    assert np.all(gray_blue == 29)

    # Pure White: 255
    pure_white = np.full((2, 2, 3), 255, dtype=np.uint8)
    gray_white = rgb_to_grayscale_syllabus(pure_white)
    assert np.all(gray_white == 255)

def test_rgba_handling():
    """Ensure RGBA input is composited cleanly without transparency artifacts."""
    rgba_img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    norm_img = open_and_normalize_image(rgba_img)
    assert norm_img.mode == "RGB"
    assert norm_img.size == (100, 100)

def test_preprocessing_pipeline_dimensions():
    """Verify preprocessed dictionary contains standard dimensions and required keys."""
    test_img = Image.new("RGB", (320, 240), (120, 150, 180))
    prep = preprocess_image_for_features(test_img)
    assert prep["rgb"].shape == (STANDARD_HEIGHT, STANDARD_WIDTH, 3)
    assert prep["gray"].shape == (STANDARD_HEIGHT, STANDARD_WIDTH)
    assert prep["hsv"].shape == (STANDARD_HEIGHT, STANDARD_WIDTH, 3)
    assert prep["original_size"] == (320, 240)

def test_brightness_contrast_transformation():
    """Verify s = clip(alpha * r + beta, 0, L - 1)."""
    base = np.array([50, 100, 200], dtype=np.uint8)
    # Brightness offset +20
    brightened = apply_brightness_contrast(base, alpha=1.0, beta=20)
    assert np.array_equal(brightened, [70, 120, 220])

    # Clipping upper bound
    overflow = apply_brightness_contrast(base, alpha=1.0, beta=100)
    assert np.array_equal(overflow, [150, 200, 255])

    # Contrast factor 2.0
    contrast = apply_brightness_contrast(base, alpha=2.0, beta=0)
    assert np.array_equal(contrast, [100, 200, 255])

def test_gamma_transformation():
    """Verify s = (L - 1) * (r / (L - 1))^gamma."""
    base = np.array([0, 255], dtype=np.uint8)
    # Extremes 0 and 255 must map to 0 and 255
    res = apply_gamma_transformation(base, gamma=1.5)
    assert res[0] == 0
    assert res[1] == 255

    mid = np.array([128], dtype=np.uint8)
    # gamma > 1 darkens midtones
    darkened_mid = apply_gamma_transformation(mid, gamma=2.0)
    assert darkened_mid[0] < 128
