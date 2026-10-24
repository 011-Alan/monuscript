"""
MonuScript - Master Feature Extractor Pipeline
Combines:
1. Grayscale Histogram & Intensity Statistics (40 features)
2. Sobel Edge Gradient Magnitude, Orientation & Density (36 features)
3. Texture Coarseness Moments & 3x3 Spatial Grid Structure (31 features)
4. HSV Color Space Normalized Histograms (32 features: H:16, S:8, V:8)

Total Deterministic Feature Vector Dimension: 139 features.
Strictly deterministic, identical feature order and sizing for training and inference.
"""

import numpy as np
from .preprocessing import preprocess_image_for_features
from .histogram_features import extract_histogram_features
from .edge_features import extract_edge_features
from .texture_features import extract_texture_features

FEATURE_GROUPS = {
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

TOTAL_FEATURE_DIM = 139

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

def extract_features_from_preprocessed(preprocessed_dict):
    """
    Extracts the full 139-dimensional feature vector from preprocessed image dictionary.
    """
    gray = preprocessed_dict["gray"]
    hsv = preprocessed_dict["hsv"]

    hist_feats = extract_histogram_features(gray)
    edge_feats = extract_edge_features(gray)
    texture_feats = extract_texture_features(gray)
    color_feats = extract_hsv_color_features(hsv)

    feature_vec = np.concatenate([hist_feats, edge_feats, texture_feats, color_feats]).astype(np.float32)

    assert len(feature_vec) == TOTAL_FEATURE_DIM, f"Expected {TOTAL_FEATURE_DIM} features, got {len(feature_vec)}"
    return feature_vec

def extract_features(image_input):
    """
    End-to-end feature extraction from raw image input.
    Returns:
        feature_vector: np.ndarray of shape (139,), dtype float32
        preprocessed_dict: dictionary with intermediate representations (for previews/visualizations)
    """
    prep = preprocess_image_for_features(image_input)
    feats = extract_features_from_preprocessed(prep)
    return feats, prep
