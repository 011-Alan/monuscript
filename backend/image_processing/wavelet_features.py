"""
MonuScript - Discrete Wavelet Transform (DWT) Feature Extraction Module
Adheres to Syllabus Unit III: Multi-Resolution Analysis & Wavelet Transforms.

Decomposes a 2D grayscale image into four subbands using single-level 2D-DWT:
1. LL: Approximation coefficients (low-frequency context and global luminance)
2. LH: Horizontal detail coefficients (horizontal frequency variations / vertical structures)
3. HL: Vertical detail coefficients (vertical frequency variations / horizontal structures)
4. HH: Diagonal detail coefficients (high-frequency corners, lattice stone patterns, textures)

Extracts a compact, scale-consistent vector of numerical statistics per subband:
- Mean absolute coefficient magnitude: (1 / N) * sum(|W|)
- Coefficient energy: (1 / N) * sum(W^2)
- Standard deviation of coefficients: sqrt((1 / N) * sum((W - mu)^2))
- Relative energy ratio: Subband_Energy / (Total_Energy + eps)

Total features: 4 subbands * 4 statistics = 16 features.
"""

import numpy as np
import pywt

# Default wavelet family: Haar (db1) is mathematically orthogonal, compact, and syllabus-standard
DEFAULT_WAVELET = "haar"
DEFAULT_MODE = "symmetric"
WAVELET_FEATURE_DIM = 16
WAVELET_FEATURE_NAMES = [
    f"{sb}_{stat}"
    for sb in ["LL", "LH", "HL", "HH"]
    for stat in ["mean_abs", "energy", "std", "rel_energy"]
]


def compute_2d_dwt(gray_image, wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE):
    """
    Computes single-level 2D Discrete Wavelet Transform on a 2D grayscale image.

    Parameters:
        gray_image: np.ndarray of shape (H, W), dtype uint8 or float32.
        wavelet: str, wavelet family name (default 'haar').
        mode: str, signal extension mode at boundaries (default 'symmetric').
    Returns:
        LL: 2D array of approximation coefficients.
        (LH, HL, HH): tuple of 2D arrays containing horizontal, vertical, and diagonal detail coefficients.
    """
    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray_image.shape}")

    # Normalize image to float32 [0, 1] for stable energy computation
    img_f = gray_image.astype(np.float32) / 255.0
    LL, (LH, HL, HH) = pywt.dwt2(img_f, wavelet=wavelet, mode=mode)
    return LL, (LH, HL, HH)


def compute_subband_statistics(subband):
    """
    Computes numerical statistics for a single DWT coefficient subband:
    1. Mean absolute value
    2. Energy (mean of squares)
    3. Standard deviation

    Parameters:
        subband: 2D np.ndarray of wavelet coefficients.
    Returns:
        tuple (mean_abs, energy, std)
    """
    N = float(subband.size)
    if N == 0:
        return 0.0, 0.0, 0.0

    mean_abs = float(np.mean(np.abs(subband)))
    energy = float(np.mean(subband ** 2))
    std = float(np.std(subband))

    return mean_abs, energy, std


def extract_wavelet_features(gray_image, wavelet=DEFAULT_WAVELET, mode=DEFAULT_MODE):
    """
    Extracts a fixed 16-dimensional wavelet feature vector from a grayscale image:
    Subbands evaluated: LL, LH, HL, HH.
    For each subband:
      - Mean absolute coefficient
      - Subband energy
      - Subband standard deviation
      - Relative energy ratio (Subband_Energy / Total_Energy)

    Total features: 16 values.
    """
    LL, (LH, HL, HH) = compute_2d_dwt(gray_image, wavelet=wavelet, mode=mode)
    subbands = {"LL": LL, "LH": LH, "HL": HL, "HH": HH}

    # Compute individual subband energy first to calculate relative energy ratios
    energies = {}
    stats_map = {}
    for name, sb in subbands.items():
        m_abs, en, s_dev = compute_subband_statistics(sb)
        stats_map[name] = (m_abs, en, s_dev)
        energies[name] = en

    total_energy = sum(energies.values()) + 1e-10

    features = []
    # Deterministic order: LL -> LH -> HL -> HH
    for name in ["LL", "LH", "HL", "HH"]:
        m_abs, en, s_dev = stats_map[name]
        rel_energy = en / total_energy
        features.extend([m_abs, en, s_dev, rel_energy])

    feat_arr = np.array(features, dtype=np.float32)
    assert len(feat_arr) == WAVELET_FEATURE_DIM, f"Expected {WAVELET_FEATURE_DIM} wavelet features, got {len(feat_arr)}"
    return feat_arr
