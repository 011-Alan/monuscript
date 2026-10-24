"""
Unit Tests for Handcrafted Feature Extraction
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from image_processing.histogram_features import (
    compute_normalized_histogram,
    compute_intensity_statistics,
    extract_histogram_features,
    HISTOGRAM_BINS
)
from image_processing.edge_features import (
    compute_sobel_gradients,
    compute_edge_density,
    extract_edge_features
)
from image_processing.texture_features import (
    compute_local_std_map,
    compute_spatial_grid_statistics,
    extract_texture_features
)
from image_processing.feature_extractor import (
    extract_features,
    TOTAL_FEATURE_DIM,
    FEATURE_GROUPS
)

def test_normalized_histogram_sum():
    """Verify that normalized histogram sums to 1.0 within floating point precision."""
    gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    hist = compute_normalized_histogram(gray, num_bins=HISTOGRAM_BINS)
    assert len(hist) == HISTOGRAM_BINS
    assert pytest.approx(float(np.sum(hist)), 1e-5) == 1.0

def test_intensity_statistics_moments():
    """Verify mean, variance, and standard deviation calculations."""
    const_gray = np.full((50, 50), 100, dtype=np.uint8)
    stats = compute_intensity_statistics(const_gray)
    # Mean should be 100/255
    assert pytest.approx(stats[0], 1e-4) == 100.0 / 255.0
    # Variance should be 0 for constant image
    assert pytest.approx(stats[1], 1e-4) == 0.0

def test_sobel_gradients_edge_detection():
    """Verify Sobel gradient detects a sharp step edge."""
    # Image with sharp vertical edge: left half 0, right half 255
    step_edge = np.zeros((64, 64), dtype=np.uint8)
    step_edge[:, 32:] = 255

    Gx, Gy, G, theta = compute_sobel_gradients(step_edge)
    # Gx should have high positive response at edge x=32
    assert np.max(Gx) > 500.0
    # Gy along flat vertical axis should be near 0
    assert np.max(np.abs(Gy)) < 1e-3
    # Orientation for vertical edge gradient vector pointing right should be 0 deg
    assert np.any(G > 100)

def test_total_feature_vector_dimension():
    """Verify feature extractor produces strictly 139 features and is deterministic."""
    img = Image.new("RGB", (128, 128), (200, 150, 100))
    feats1, _ = extract_features(img)
    feats2, _ = extract_features(img)

    assert len(feats1) == TOTAL_FEATURE_DIM
    assert np.array_equal(feats1, feats2)
    assert not np.any(np.isnan(feats1))
    assert not np.any(np.isinf(feats1))
