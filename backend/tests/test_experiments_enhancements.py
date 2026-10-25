"""
MonuScript - Unit & Integration Tests for Experimental DIP Enhancements (E1-E7)
Tests:
1. CLAHE in LAB color space (preprocessing)
2. 2D Haar Wavelet Feature Extraction (Unit III)
3. Morphological Silhouette Shape Descriptors (Unit I & IV)
4. Master Feature Extractor Configurations & Backward Compatibility
5. PCA Dimensionality Reduction consistency (Unit IV)
6. Multi-evidence Rejection Mechanism calibration (Unit V)
7. Monument-Specific Architectural Descriptors (Unit I, II & IV)
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
    apply_clahe_lab,
    preprocess_image_for_features,
    STANDARD_WIDTH,
    STANDARD_HEIGHT
)
from image_processing.wavelet_features import (
    extract_wavelet_features,
    WAVELET_FEATURE_DIM,
    WAVELET_FEATURE_NAMES
)
from image_processing.shape_features import (
    extract_shape_features,
    SHAPE_FEATURE_DIM,
    SHAPE_FEATURE_NAMES
)
from image_processing.architectural_features import (
    extract_architectural_features,
    ARCHITECTURAL_FEATURE_DIM,
    ARCHITECTURAL_FEATURE_NAMES
)
from image_processing.feature_extractor import (
    extract_features,
    get_feature_groups_for_config,
    TOTAL_FEATURE_DIM
)
from ml.rejection import RejectionJudge
from sklearn.decomposition import PCA


# =====================================================================
# 1. CLAHE in LAB Color Space Tests
# =====================================================================

def test_clahe_output_shape_and_dtype():
    """Verify apply_clahe_lab preserves spatial dimensions and uint8 dtype."""
    img_rgb = np.random.randint(50, 200, (120, 160, 3), dtype=np.uint8)
    enhanced = apply_clahe_lab(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8))

    assert enhanced.shape == img_rgb.shape
    assert enhanced.dtype == np.uint8
    assert not np.any(np.isnan(enhanced))


def test_clahe_contrast_enhancement():
    """Verify CLAHE broadens standard deviation of low-contrast luminance."""
    # Create low-contrast grayscale-like RGB image
    low_contrast = np.full((100, 100, 3), 128, dtype=np.uint8)
    low_contrast[:50, :50] = 132
    low_contrast[50:, 50:] = 124

    std_before = np.std(low_contrast[:, :, 0])
    enhanced = apply_clahe_lab(low_contrast, clip_limit=3.0, tile_grid_size=(4, 4))
    std_after = np.std(enhanced[:, :, 0])

    # CLAHE should increase luminance spread
    assert std_after >= std_before


def test_preprocess_image_with_clahe_toggle():
    """Verify preprocess_image_for_features respects use_clahe toggle."""
    synthetic_pil = Image.new("RGB", (200, 200), (140, 120, 100))
    prep_no_clahe = preprocess_image_for_features(synthetic_pil, use_clahe=False)
    prep_clahe = preprocess_image_for_features(synthetic_pil, use_clahe=True, clahe_clip_limit=2.0)

    assert prep_no_clahe["rgb"].shape == (STANDARD_HEIGHT, STANDARD_WIDTH, 3)
    assert prep_clahe["rgb"].shape == (STANDARD_HEIGHT, STANDARD_WIDTH, 3)
    assert prep_clahe["gray"].shape == (STANDARD_HEIGHT, STANDARD_WIDTH)


# =====================================================================
# 2. 2D Haar Wavelet Feature Extraction Tests
# =====================================================================

def test_wavelet_feature_dimensions():
    """Verify wavelet extraction yields strictly 16 compact statistics."""
    gray = np.random.randint(0, 256, (STANDARD_HEIGHT, STANDARD_WIDTH), dtype=np.uint8)
    wav_feats = extract_wavelet_features(gray)

    assert len(wav_feats) == WAVELET_FEATURE_DIM
    assert wav_feats.dtype == np.float32
    assert len(WAVELET_FEATURE_NAMES) == 16
    assert not np.any(np.isnan(wav_feats))
    assert not np.any(np.isinf(wav_feats))


def test_wavelet_energy_non_negative():
    """Verify wavelet subband energies and mean magnitudes are non-negative."""
    gray = np.zeros((128, 128), dtype=np.uint8)
    gray[32:96, 32:96] = 255  # High-contrast square

    wav_feats = extract_wavelet_features(gray)
    energies = [wav_feats[1], wav_feats[5], wav_feats[9], wav_feats[13]]
    for e in energies:
        assert e >= 0.0


# =====================================================================
# 3. Morphological Silhouette Shape Descriptors Tests
# =====================================================================

def test_shape_feature_dimensions():
    """Verify morphological shape extraction returns strictly 9 descriptors."""
    gray = np.random.randint(0, 256, (STANDARD_HEIGHT, STANDARD_WIDTH), dtype=np.uint8)
    shape_feats = extract_shape_features(gray)

    assert len(shape_feats) == SHAPE_FEATURE_DIM
    assert shape_feats.dtype == np.float32
    assert len(SHAPE_FEATURE_NAMES) == 9
    assert not np.any(np.isnan(shape_feats))


def test_shape_degenerate_mask_handling():
    """Verify shape extractor handles uniform blank image safely without division by zero."""
    blank_gray = np.zeros((STANDARD_HEIGHT, STANDARD_WIDTH), dtype=np.uint8)
    shape_feats = extract_shape_features(blank_gray)

    assert len(shape_feats) == 9
    assert np.all(shape_feats == 0.0)


def test_shape_geometric_properties():
    """Verify shape descriptors capture aspect ratio on tall vs wide objects."""
    # Tall synthetic object (mimicking Qutub Minar silhouette)
    tall_img = np.full((STANDARD_HEIGHT, STANDARD_WIDTH), 255, dtype=np.uint8)
    tall_img[20:280, 130:170] = 30
    tall_feats = extract_shape_features(tall_img)

    # Wide synthetic object
    wide_img = np.full((STANDARD_HEIGHT, STANDARD_WIDTH), 255, dtype=np.uint8)
    wide_img[180:260, 40:260] = 30
    wide_feats = extract_shape_features(wide_img)

    # Aspect ratio is index 1 (width / height)
    assert tall_feats[1] < wide_feats[1]


# =====================================================================
# 4. Master Feature Extractor Configurations & Compatibility
# =====================================================================

def test_extractor_backward_compatibility():
    """Verify default extract_features call preserves baseline 139 dimensions."""
    img = Image.new("RGB", (128, 128), (150, 120, 90))
    feats, prep = extract_features(img)

    assert len(feats) == TOTAL_FEATURE_DIM
    assert feats.dtype == np.float32


def test_extractor_enhanced_configurations():
    """Verify extract_features with wavelet, shape, and architectural flags expands vector cleanly."""
    img = Image.new("RGB", (128, 128), (150, 120, 90))

    # Architectural only (139 + 16 = 155)
    f_arch, _ = extract_features(img, include_architectural=True)
    assert len(f_arch) == 139 + ARCHITECTURAL_FEATURE_DIM

    # Wavelet only (139 + 16 = 155)
    f_wav, _ = extract_features(img, include_wavelet=True)
    assert len(f_wav) == 139 + WAVELET_FEATURE_DIM

    # Shape only (139 + 9 = 148)
    f_shp, _ = extract_features(img, include_shape=True)
    assert len(f_shp) == 139 + SHAPE_FEATURE_DIM

    # All enhancements (139 + 16 + 9 + 16 = 180)
    f_all, _ = extract_features(img, include_wavelet=True, include_shape=True, include_architectural=True)
    assert len(f_all) == 139 + WAVELET_FEATURE_DIM + SHAPE_FEATURE_DIM + ARCHITECTURAL_FEATURE_DIM

    # Verify feature group mapping matches
    groups, total_dim = get_feature_groups_for_config(include_architectural=True)
    assert total_dim == 155
    assert "architectural" in groups
    assert groups["architectural"]["length"] == 16


# =====================================================================
# 5. PCA Dimensionality Reduction Consistency Tests
# =====================================================================

def test_pca_variance_retention():
    """Verify PCA fitted on training data preserves specified variance ratio."""
    np.random.seed(42)
    synthetic_train = np.random.randn(72, 139).astype(np.float32)
    synthetic_test = np.random.randn(24, 139).astype(np.float32)

    pca = PCA(n_components=0.95, random_state=42)
    train_pca = pca.fit_transform(synthetic_train)
    test_pca = pca.transform(synthetic_test)

    assert train_pca.shape[1] < 139
    assert test_pca.shape[1] == train_pca.shape[1]
    assert np.sum(pca.explained_variance_ratio_) >= 0.95


# =====================================================================
# 6. Calibrated Rejection Mechanism Tests
# =====================================================================

def test_calibrated_rejection_decisions():
    """Verify calibrated rejection judge properly separates confident from uncertain and negative predictions."""
    judge = RejectionJudge(
        accept_threshold=0.48,
        margin_threshold=0.18,
        uncertain_threshold=0.32
    )
    classes = ["taj_mahal", "qutub_minar", "red_fort", "india_gate", "charminar", "hawa_mahal"]

    # Case 1: Confident and decisive prediction -> ACCEPTED
    confident_probs = np.array([0.70, 0.10, 0.05, 0.05, 0.05, 0.05])
    d1 = judge.evaluate(confident_probs, classes)
    assert d1["accepted"] is True
    assert d1["top_class"] == "taj_mahal"
    assert d1["score_margin"] >= 0.18

    # Case 2: Ambiguous margin (0.49 vs 0.45, margin=0.04 < 0.18) -> UNCERTAIN / NOT ACCEPTED
    ambiguous_probs = np.array([0.49, 0.45, 0.02, 0.02, 0.01, 0.01])
    d2 = judge.evaluate(ambiguous_probs, classes)
    assert d2["accepted"] is False
    assert d2["is_uncertain"] is True

    # Case 3: Very low score (< 0.32) -> OUTRIGHT REJECTED
    flat_probs = np.full(6, 1.0 / 6.0)
    d3 = judge.evaluate(flat_probs, classes)
    assert d3["accepted"] is False
    assert d3["is_uncertain"] is False
    assert "Haven't seen this one before" in d3["message"]


# =====================================================================
# 7. Monument-Specific Architectural Feature Extraction Tests
# =====================================================================

def test_architectural_feature_dimensions():
    """Verify architectural extraction yields strictly 16 finite descriptors."""
    gray = np.random.randint(0, 256, (STANDARD_HEIGHT, STANDARD_WIDTH), dtype=np.uint8)
    rgb = np.random.randint(0, 256, (STANDARD_HEIGHT, STANDARD_WIDTH, 3), dtype=np.uint8)
    arch_feats = extract_architectural_features(gray, rgb_image=rgb)

    assert len(arch_feats) == ARCHITECTURAL_FEATURE_DIM
    assert arch_feats.dtype == np.float32
    assert len(ARCHITECTURAL_FEATURE_NAMES) == 16
    assert not np.any(np.isnan(arch_feats))
    assert not np.any(np.isinf(arch_feats))


def test_architectural_feature_degenerate_handling():
    """Verify architectural extractor handles blank image without crashing or dividing by zero."""
    blank_gray = np.zeros((STANDARD_HEIGHT, STANDARD_WIDTH), dtype=np.uint8)
    arch_feats = extract_architectural_features(blank_gray)

    assert len(arch_feats) == ARCHITECTURAL_FEATURE_DIM
    assert np.all(arch_feats == 0.0)


def test_architectural_feature_cornice_and_window_distinction():
    """Verify architectural features capture upper horizontal edges vs repetitive window patterns."""
    # Synthetic image with strong upper horizontal line (India Gate cornice)
    cornice_img = np.full((STANDARD_HEIGHT, STANDARD_WIDTH), 128, dtype=np.uint8)
    cornice_img[40:50, :] = 255
    feats_cornice = extract_architectural_features(cornice_img)

    # Synthetic image with vertical line (mimicking column shaft)
    column_img = np.full((STANDARD_HEIGHT, STANDARD_WIDTH), 128, dtype=np.uint8)
    column_img[:, 120:136] = 255
    feats_column = extract_architectural_features(column_img)

    # Feature index 1 is india_gate_cornice_dominance (gy - gx)/(gx + gy)
    # The horizontal cornice image has higher cornice dominance than vertical column
    assert feats_cornice[1] > feats_column[1]
