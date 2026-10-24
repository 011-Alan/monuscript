"""
MonuScript - Gray-Level Transformations Module
Implements standard digital image processing gray-level transformations:
1. Linear contrast and brightness adjustment:
   s = clip(alpha * r + beta, 0, L - 1)
2. Power-Law (Gamma) transformation:
   s = (L - 1) * (r / (L - 1)) ^ gamma
Used for robustness evaluation, illumination invariance testing, and data augmentation.
"""

import numpy as np

def apply_brightness_contrast(image_arr, alpha=1.0, beta=0.0, L=256):
    """
    Linear intensity transformation:
    s = clip(alpha * r + beta, 0, L - 1)

    Parameters:
        image_arr: np.ndarray (grayscale or multichannel) uint8 or float
        alpha: float, contrast gain (alpha > 1 increases contrast, alpha < 1 decreases)
        beta: float, brightness offset (positive brightens, negative darkens)
        L: int, intensity dynamic range (default 256 for 8-bit images)
    Returns:
        transformed image of same shape, uint8
    """
    r = image_arr.astype(np.float32)
    s = alpha * r + beta
    s_clipped = np.clip(s, 0, L - 1)
    return np.round(s_clipped).astype(np.uint8)

def apply_gamma_transformation(image_arr, gamma=1.0, L=256):
    """
    Power-Law (Gamma) transformation:
    s = (L - 1) * (r / (L - 1)) ^ gamma

    Parameters:
        image_arr: np.ndarray uint8 or float
        gamma: float, gamma exponent:
            gamma < 1 expands dark regions (brightens shadow detail)
            gamma > 1 compresses dark regions (enhances highlights/contrast)
        L: int, default 256
    Returns:
        transformed image of same shape, uint8
    """
    if gamma <= 0:
        raise ValueError(f"Gamma must be positive, got {gamma}")

    r = image_arr.astype(np.float32)
    normalized = r / float(L - 1)
    # Power-law mapping
    s = (float(L - 1)) * np.power(normalized, gamma)
    s_clipped = np.clip(s, 0, L - 1)
    return np.round(s_clipped).astype(np.uint8)
