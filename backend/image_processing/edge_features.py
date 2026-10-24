"""
MonuScript - Edge Detection & Gradient Features Module
Implements syllabus Sobel spatial filtering and edge descriptors:
1. Sobel kernels:
   Sx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
   Sy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
2. Convolution:
   Gx = I * Sx
   Gy = I * Sy
3. Gradient Magnitude:
   G = sqrt(Gx^2 + Gy^2)
4. Gradient Orientation:
   theta = atan2(Gy, Gx) mapped to [0, 180 degrees]
5. Edge Density:
   ratio of edge pixels exceeding threshold to total pixels
6. Gradient magnitude histogram (16 bins)
7. Gradient orientation histogram weighted by magnitude (16 bins)
8. Directional anisotropy ratio: sum(|Gx|) / (sum(|Gy|) + eps)
"""

import numpy as np
import cv2

NUM_MAGNITUDE_BINS = 16
NUM_ORIENTATION_BINS = 16

# Standard Sobel 3x3 kernels from image processing syllabus
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

def compute_sobel_gradients(gray_image):
    """
    Computes horizontal and vertical spatial derivatives using the Sobel operator.
    Returns:
        Gx, Gy: np.ndarray of shape (H, W), dtype float32
        G: magnitude, np.ndarray of shape (H, W), dtype float32
        theta: orientation in degrees [0, 180), np.ndarray of shape (H, W), dtype float32
    """
    I = gray_image.astype(np.float32)

    # Spatial 2D convolution with Sobel kernels using OpenCV filter2D for efficiency
    Gx = cv2.filter2D(I, cv2.CV_32F, SOBEL_X, borderType=cv2.BORDER_REFLECT)
    Gy = cv2.filter2D(I, cv2.CV_32F, SOBEL_Y, borderType=cv2.BORDER_REFLECT)

    # Gradient magnitude: G = sqrt(Gx^2 + Gy^2)
    G = np.sqrt(Gx ** 2 + Gy ** 2)

    # Gradient orientation: atan2(Gy, Gx) in radians [-pi, pi]
    angle_rad = np.arctan2(Gy, Gx)
    # Convert to degrees [0, 360)
    angle_deg = np.degrees(angle_rad) % 360.0
    # Map to [0, 180) for unsigned edge lines
    angle_180 = angle_deg % 180.0

    return Gx, Gy, G, angle_180

def compute_edge_density(G, threshold=None):
    """
    Computes the fraction of pixels classified as edge points.
    If threshold is None, uses Otsu's threshold on G.
    """
    H, W = G.shape
    total_pixels = float(H * W)
    if total_pixels == 0:
        return 0.0

    if threshold is None:
        # Otsu thresholding on normalized magnitude
        G_norm = np.clip((G / (np.max(G) + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(G_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = otsu_val * (np.max(G) + 1e-6) / 255.0

    edge_count = np.count_nonzero(G > threshold)
    return float(edge_count) / total_pixels

def compute_gradient_histograms(G, theta):
    """
    Computes:
    1. Normalized gradient magnitude histogram (16 bins)
    2. Orientation histogram (16 bins in [0, 180) degrees) weighted by gradient magnitude
    """
    # 1. Magnitude histogram
    max_mag = 4.0 * 255.0  # theoretical maximum for Sobel on [0, 255]
    mag_hist, _ = np.histogram(G, bins=NUM_MAGNITUDE_BINS, range=(0, max_mag))
    mag_sum = np.sum(mag_hist)
    mag_hist_norm = (mag_hist.astype(np.float32) / (mag_sum + 1e-8)) if mag_sum > 0 else np.zeros(NUM_MAGNITUDE_BINS, dtype=np.float32)

    # 2. Weighted orientation histogram (HOG-style direction binning)
    # Bin theta into 16 bins spanning [0, 180) degrees, with weights = G
    ori_hist, _ = np.histogram(theta, bins=NUM_ORIENTATION_BINS, range=(0, 180.0), weights=G)
    ori_sum = np.sum(ori_hist)
    ori_hist_norm = (ori_hist.astype(np.float32) / (ori_sum + 1e-8)) if ori_sum > 0 else np.zeros(NUM_ORIENTATION_BINS, dtype=np.float32)

    return mag_hist_norm, ori_hist_norm

def extract_edge_features(gray_image):
    """
    Extracts all edge features:
    - Magnitude histogram: 16 bins
    - Orientation histogram: 16 bins
    - Edge density: 1 float
    - Mean gradient magnitude: 1 float
    - Std gradient magnitude: 1 float
    - Anisotropy ratio (Gx / Gy): 1 float
    Total length: 36 features.
    """
    Gx, Gy, G, theta = compute_sobel_gradients(gray_image)
    mag_hist, ori_hist = compute_gradient_histograms(G, theta)
    density = compute_edge_density(G)

    mean_mag = float(np.mean(G)) / 255.0
    std_mag = float(np.std(G)) / 255.0
    
    sum_abs_gx = np.sum(np.abs(Gx))
    sum_abs_gy = np.sum(np.abs(Gy))
    anisotropy = float((sum_abs_gx - sum_abs_gy) / (sum_abs_gx + sum_abs_gy + 1e-6))

    scalar_stats = np.array([density, mean_mag, std_mag, anisotropy], dtype=np.float32)

    return np.concatenate([mag_hist, ori_hist, scalar_stats]).astype(np.float32)
