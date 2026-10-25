"""
MonuScript - Master Feature Extractor Pipeline
Combines syllabus-aligned traditional digital image processing feature extractors:
1. Grayscale Histogram & Intensity Statistics (40 features) - Unit I & II
2. Sobel Edge Gradient Magnitude, Orientation & Density (36 features) - Unit IV
3. Texture Coarseness Moments & 3x3 Spatial Grid Structure (31 features) - Unit IV
4. HSV Color Space Normalized Histograms (32 features: H:16, S:8, V:8) - Unit I
Optional syllabus enhancement groups:
5. 2D Discrete Wavelet Transform (DWT) Multiresolution Statistics (16 features) - Unit III
6. Morphological Shape Descriptors (9 features) - Unit I & IV
7. Monument-Specific Architectural Descriptors (16 features) - Unit I, II & IV

Strictly deterministic, identical feature ordering and sizing for training and inference.
"""

import numpy as np
from .preprocessing import preprocess_image_for_features
from .histogram_features import extract_histogram_features
from .edge_features import extract_edge_features
from .texture_features import extract_texture_features
from .wavelet_features import extract_wavelet_features, WAVELET_FEATURE_DIM
from .shape_features import extract_shape_features, SHAPE_FEATURE_DIM
from .architectural_features import (
    extract_architectural_features,
    ARCHITECTURAL_FEATURE_DIM,
    ARCHITECTURAL_FEATURE_NAMES
)

BASE_FEATURE_GROUPS = {
    "histogram": {
        "start": 0,
        "end": 40,
        "length": 40,
        "description": "Grayscale normalized histogram (32 bins) and intensity statistical moments (8 values)",
        "default_weight": 0.20
    },
    "edge": {
        "start": 40,
        "end": 76,
        "length": 36,
        "description": "Sobel gradient magnitude histogram (16 bins), weighted orientation histogram (16 bins), edge density and directional anisotropy (4 values)",
        "default_weight": 0.45
    },
    "texture": {
        "start": 76,
        "end": 107,
        "length": 31,
        "description": "Local standard deviation coarseness moments (4 values) and 3x3 spatial grid architectural layout descriptors (27 values)",
        "default_weight": 0.30
    },
    "color": {
        "start": 107,
        "end": 139,
        "length": 32,
        "description": "Normalized HSV color channel histograms (Hue: 16 bins, Saturation: 8 bins, Value: 8 bins)",
        "default_weight": 0.05
    }
}

FEATURE_GROUPS = BASE_FEATURE_GROUPS
BASE_FEATURE_DIM = 139
TOTAL_FEATURE_DIM = 139  # Baseline feature dimension without enhancements (40 + 36 + 31 + 32)
PRODUCTION_FEATURE_DIM = 155  # Production model feature dimension (139 + 16 architectural descriptors)


def get_feature_groups_for_config(include_wavelet=False, include_shape=False, include_architectural=False):
    """
    Constructs the feature group boundary dictionary for a specified feature configuration.
    """
    groups = {k: dict(v) for k, v in BASE_FEATURE_GROUPS.items()}
    current_idx = 139

    if include_wavelet:
        groups["wavelet"] = {
            "start": current_idx,
            "end": current_idx + WAVELET_FEATURE_DIM,
            "length": WAVELET_FEATURE_DIM,
            "description": "2D Haar Wavelet transform subband energy, mean magnitude, and variance statistics (16 values)",
            "default_weight": 0.25
        }
        current_idx += WAVELET_FEATURE_DIM

    if include_shape:
        groups["shape"] = {
            "start": current_idx,
            "end": current_idx + SHAPE_FEATURE_DIM,
            "length": SHAPE_FEATURE_DIM,
            "description": "Morphological silhouette shape descriptors: aspect ratio, compactness, convexity, symmetry (9 values)",
            "default_weight": 0.15
        }
        current_idx += SHAPE_FEATURE_DIM

    if include_architectural:
        groups["architectural"] = {
            "start": current_idx,
            "end": current_idx + ARCHITECTURAL_FEATURE_DIM,
            "length": ARCHITECTURAL_FEATURE_DIM,
            "description": "Monument-specific architectural descriptors: central arch void, upper cornice dominance, tapering ratio, window opening density, dome circularity (16 values)",
            "default_weight": 0.40
        }
        current_idx += ARCHITECTURAL_FEATURE_DIM

    return groups, current_idx


def extract_hsv_color_features(hsv_image):
    """
    Computes normalized histograms for Hue, Saturation, and Value channels:
    - Hue: 16 bins, range [0, 180) in OpenCV
    - Saturation: 8 bins, range [0, 256)
    - Value: 8 bins, range [0, 256)
    Total: 32 color features.
    """
    if hsv_image.ndim != 3 or hsv_image.shape[2] != 3:
        raise ValueError("Expected 3-channel HSV image")

    H = hsv_image[:, :, 0]
    S = hsv_image[:, :, 1]
    V = hsv_image[:, :, 2]

    # Hue histogram (16 bins)
    h_hist, _ = np.histogram(H, bins=16, range=(0, 180))
    h_norm = (h_hist.astype(np.float32) / (np.sum(h_hist) + 1e-8))

    # Saturation histogram (8 bins)
    s_hist, _ = np.histogram(S, bins=8, range=(0, 256))
    s_norm = (s_hist.astype(np.float32) / (np.sum(s_hist) + 1e-8))

    # Value histogram (8 bins)
    v_hist, _ = np.histogram(V, bins=8, range=(0, 256))
    v_norm = (v_hist.astype(np.float32) / (np.sum(v_hist) + 1e-8))

    return np.concatenate([h_norm, s_norm, v_norm]).astype(np.float32)


def extract_features_from_preprocessed(
    preprocessed_dict,
    include_wavelet=False,
    include_shape=False,
    include_architectural=False
):
    """
    Extracts the feature vector from preprocessed image dictionary according to configuration.
    """
    gray = preprocessed_dict["gray"]
    hsv = preprocessed_dict["hsv"]
    rgb = preprocessed_dict.get("rgb")

    hist_feats = extract_histogram_features(gray)
    edge_feats = extract_edge_features(gray)
    texture_feats = extract_texture_features(gray)
    color_feats = extract_hsv_color_features(hsv)

    parts = [hist_feats, edge_feats, texture_feats, color_feats]

    if include_wavelet:
        wav_feats = extract_wavelet_features(gray)
        parts.append(wav_feats)

    if include_shape:
        shape_feats = extract_shape_features(gray)
        parts.append(shape_feats)

    if include_architectural:
        arch_feats = extract_architectural_features(gray, rgb_image=rgb)
        parts.append(arch_feats)

    feature_vec = np.concatenate(parts).astype(np.float32)

    expected_dim = (
        139
        + (WAVELET_FEATURE_DIM if include_wavelet else 0)
        + (SHAPE_FEATURE_DIM if include_shape else 0)
        + (ARCHITECTURAL_FEATURE_DIM if include_architectural else 0)
    )
    assert len(feature_vec) == expected_dim, f"Expected {expected_dim} features, got {len(feature_vec)}"
    return feature_vec


def extract_features(
    image_input,
    use_clahe=False,
    clahe_clip_limit=2.0,
    clahe_grid_size=(8, 8),
    include_wavelet=False,
    include_shape=False,
    include_architectural=False
):
    """
    End-to-end feature extraction from raw image input.
    Parameters:
        image_input: file path, bytes, or PIL.Image.
        use_clahe: bool, whether to apply LAB CLAHE preprocessing.
        clahe_clip_limit: float, CLAHE clip limit.
        clahe_grid_size: tuple, CLAHE grid size.
        include_wavelet: bool, whether to append 16 DWT features.
        include_shape: bool, whether to append 9 morphological shape features.
        include_architectural: bool, whether to append 16 architectural features.
    Returns:
        feature_vector: np.ndarray of shape (N,), dtype float32
        preprocessed_dict: dictionary with intermediate representations
    """
    prep = preprocess_image_for_features(
        image_input,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_grid_size=clahe_grid_size
    )
    feats = extract_features_from_preprocessed(
        prep,
        include_wavelet=include_wavelet,
        include_shape=include_shape,
        include_architectural=include_architectural
    )
    return feats, prep
