"""
MonuScript - Histogram & Intensity Statistics Feature Extraction
Adheres strictly to the Image Processing syllabus formulas:
1. Normalized Grayscale Histogram:
   h(k) = n_k / (M * N), for k = 0, ..., K-1
2. Mean intensity:
   mu = (1 / MN) * sum(I(x, y))
3. Variance:
   sigma^2 = (1 / MN) * sum((I(x, y) - mu)^2)
4. Standard deviation:
   sigma = sqrt(sigma^2)
5. Smoothness measure:
   R = 1 - (1 / (1 + sigma^2_normalized))
6. Uniformity (Energy):
   U = sum(h(k)^2)
7. Average Entropy:
   E = -sum(h(k) * log2(h(k) + eps))
8. Skewness (3rd central moment):
   mu_3 = (1 / MN) * sum(((I(x, y) - mu) / sigma)^3)
"""

import numpy as np

HISTOGRAM_BINS = 32

def compute_normalized_histogram(gray_image, num_bins=HISTOGRAM_BINS):
    """
    Computes the normalized 1D grayscale histogram:
    h(k) = n_k / (M * N)

    Parameters:
        gray_image: np.ndarray of shape (H, W), dtype uint8 or float in [0, 255]
        num_bins: int, number of histogram bins (default 32)
    Returns:
        hist: np.ndarray of shape (num_bins,), normalized such that sum(hist) == 1.0
    """
    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got {gray_image.ndim}D")

    # Bin edges from 0 to 256
    hist, _ = np.histogram(gray_image, bins=num_bins, range=(0, 256))
    total_pixels = float(gray_image.shape[0] * gray_image.shape[1])
    if total_pixels == 0:
        return np.zeros(num_bins, dtype=np.float32)

    hist_norm = hist.astype(np.float32) / total_pixels
    return hist_norm

def compute_intensity_statistics(gray_image):
    """
    Computes statistical moments of gray-level intensity:
    - Mean (mu)
    - Variance (sigma^2)
    - Standard deviation (sigma)
    - Smoothness measure (R)
    - Uniformity / Energy (U)
    - Entropy (E)
    - Skewness (mu_3)
    - Median intensity

    Returns:
        stats: np.ndarray of shape (8,), dtype float32
    """
    if gray_image.ndim != 2:
        raise ValueError("Expected 2D grayscale image")

    I = gray_image.astype(np.float32)
    MN = float(I.size)

    # 1. Mean
    mu = np.mean(I)

    # 2. Variance & Standard deviation
    diff = I - mu
    var = np.mean(diff ** 2)
    std = np.sqrt(var)

    # 3. Smoothness measure R = 1 - 1/(1 + (std/255)^2)
    norm_var = (std / 255.0) ** 2
    smoothness = 1.0 - (1.0 / (1.0 + norm_var))

    # Normalized histogram for energy & entropy
    h = compute_normalized_histogram(gray_image, num_bins=HISTOGRAM_BINS)
    
    # 4. Uniformity (Energy)
    uniformity = float(np.sum(h ** 2))

    # 5. Entropy
    eps = 1e-10
    entropy = -float(np.sum(h * np.log2(h + eps)))

    # 6. Skewness
    if std > 1e-5:
        skewness = float(np.mean((diff / std) ** 3))
    else:
        skewness = 0.0

    # 7. Median (normalized to [0, 1])
    median = float(np.median(I)) / 255.0

    # Normalize mean and std to [0, 1] for stable feature scaling
    mu_norm = mu / 255.0
    std_norm = std / 255.0

    return np.array([
        mu_norm,
        std_norm,
        var / (255.0 ** 2),
        smoothness,
        uniformity,
        entropy,
        skewness,
        median
    ], dtype=np.float32)

def extract_histogram_features(gray_image):
    """
    Combines normalized histogram (32 bins) and intensity statistics (8 values).
    Total length: 40 features.
    """
    hist = compute_normalized_histogram(gray_image, num_bins=HISTOGRAM_BINS)
    stats = compute_intensity_statistics(gray_image)
    return np.concatenate([hist, stats]).astype(np.float32)
