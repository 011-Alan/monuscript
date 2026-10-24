"""
MonuScript - Texture & Spatial Structure Features Module
Implements syllabus-approved statistical texture measures:
1. Local intensity variance & standard deviation filtering:
   Measures local neighborhood roughness and spatial variation.
2. Global moments of the local variation map:
   - Mean of local standard deviation (spatial coarseness)
   - Variance of local standard deviation (texture heterogeneity)
   - Relative smoothness
3. Spatial Block Grid Representation (3x3 grid):
   Computes local mean, local standard deviation, and local edge density
   across 9 spatial sub-regions to capture architectural geometry
   (e.g., domes at top-center, minarets at sides, reflective pool/plinth at bottom).
"""

import numpy as np
import cv2

def compute_local_std_map(gray_image, ksize=5):
    """
    Computes the local standard deviation in a ksize x ksize neighborhood.
    sigma_local = sqrt(E[I^2] - (E[I])^2)
    """
    I = gray_image.astype(np.float32)
    kernel = np.ones((ksize, ksize), dtype=np.float32) / float(ksize * ksize)

    # Local mean E[I]
    mean_local = cv2.filter2D(I, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    # Local mean of squares E[I^2]
    mean_sq_local = cv2.filter2D(I ** 2, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)

    # Local variance: var = E[I^2] - (E[I])^2
    var_local = np.maximum(mean_sq_local - (mean_local ** 2), 0.0)
    std_local = np.sqrt(var_local)
    return std_local

def compute_spatial_grid_statistics(gray_image, grid_rows=3, grid_cols=3):
    """
    Divides the standardized image into grid_rows x grid_cols spatial blocks.
    In each block, computes:
      1. Local Mean intensity (normalized to [0, 1])
      2. Local Standard deviation (normalized to [0, 1])
      3. Local Edge density (using simple gradient threshold)

    Returns:
        grid_features: np.ndarray of shape (grid_rows * grid_cols * 3,), float32
    """
    H, W = gray_image.shape
    block_h = H // grid_rows
    block_w = W // grid_cols

    # Compute a quick Sobel gradient for local edge density
    sobelx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    edge_mask = (grad_mag > 40.0).astype(np.float32)

    features = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            y1 = r * block_h
            y2 = (r + 1) * block_h if r < grid_rows - 1 else H
            x1 = c * block_w
            x2 = (c + 1) * block_w if c < grid_cols - 1 else W

            sub_img = gray_image[y1:y2, x1:x2].astype(np.float32)
            sub_edges = edge_mask[y1:y2, x1:x2]

            sub_mean = float(np.mean(sub_img)) / 255.0
            sub_std = float(np.std(sub_img)) / 255.0
            sub_edge_density = float(np.mean(sub_edges))

            features.extend([sub_mean, sub_std, sub_edge_density])

    return np.array(features, dtype=np.float32)

def extract_texture_features(gray_image):
    """
    Extracts:
    - Global texture moments of local standard deviation map: 4 values
      (mean coarseness, std coarseness, max coarseness, smoothness measure)
    - 3x3 spatial grid statistics: 9 blocks * 3 = 27 values
    Total length: 31 features.
    """
    std_map = compute_local_std_map(gray_image, ksize=5)

    coarseness_mean = float(np.mean(std_map)) / 255.0
    coarseness_std = float(np.std(std_map)) / 255.0
    coarseness_max = float(np.max(std_map)) / 255.0
    smoothness = 1.0 - (1.0 / (1.0 + coarseness_mean ** 2))

    global_texture = np.array([
        coarseness_mean,
        coarseness_std,
        coarseness_max,
        smoothness
    ], dtype=np.float32)

    spatial_grid = compute_spatial_grid_statistics(gray_image, grid_rows=3, grid_cols=3)

    return np.concatenate([global_texture, spatial_grid]).astype(np.float32)
