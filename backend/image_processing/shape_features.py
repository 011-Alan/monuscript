"""
MonuScript - Morphological Shape Descriptors Module
Adheres to Syllabus Unit I & IV: Pixel Connectivity, Boundaries, and Shape Descriptors.

Extracts normalized geometric and silhouette descriptors of the primary monument structure:
1. Relative foreground area: A_fg / (H * W)
2. Bounding box aspect ratio: W_box / (H_box + eps)
3. Bounding box extent: A_contour / (W_box * H_box + eps)
4. Compactness (circularity): 4 * pi * A_contour / (Perimeter^2 + eps)
5. Convexity ratio: A_contour / (A_convex_hull + eps)
6. Normalized perimeter: Perimeter / (2 * (H + W))
7. Vertical centroid position: C_y / H (measures ground vs sky elevation)
8. Horizontal reflection symmetry: IoU(Left_Mask, Flipped_Right_Mask)
9. Top-to-bottom mass ratio: Area_Top_Half / (Area_Bottom_Half + eps)

Total features: 9 features.
Safely handles degenerate contours and empty masks by returning zeros.
"""

import numpy as np
import cv2

SHAPE_FEATURE_DIM = 9
SHAPE_FEATURE_NAMES = [
    "relative_foreground_area",
    "bounding_box_aspect_ratio",
    "bounding_box_extent",
    "compactness_circularity",
    "convexity_ratio",
    "normalized_perimeter",
    "vertical_centroid",
    "horizontal_symmetry",
    "top_to_bottom_ratio"
]


def extract_silhouette_mask(gray_image):
    """
    Extracts binary foreground mask of the primary architectural structure
    using Sobel/Canny gradient boundary detection and morphological closing.

    Parameters:
        gray_image: np.ndarray of shape (H, W), uint8.
    Returns:
        mask: np.ndarray of shape (H, W), uint8 with values {0, 255}.
    """
    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray_image.shape}")

    H, W = gray_image.shape
    total_area = float(H * W)

    # 1. Edge boundary detection
    edges = cv2.Canny(gray_image, 40, 140)

    # 2. Morphological structural closing to connect architectural boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Find candidate external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((H, W), dtype=np.uint8)

    # 4. Find the most prominent centered architectural silhouette
    best_c = None
    max_score = -1.0
    center_pt = np.array([W / 2.0, H / 2.0])

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 0.03 * total_area:
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist_to_center = float(np.linalg.norm(np.array([cx, cy]) - center_pt))
            # Score: contour area prioritized with gentle penalty for extreme edge artifacts
            score = area / (1.0 + 0.5 * dist_to_center)
            if score > max_score:
                max_score = score
                best_c = c

    mask = np.zeros((H, W), dtype=np.uint8)
    if best_c is not None:
        cv2.drawContours(mask, [best_c], -1, 255, -1)
    return mask


def extract_shape_features(gray_image):
    """
    Extracts normalized, resolution-invariant morphological shape features.
    If no valid silhouette contour can be detected, returns safe zeros.

    Returns:
        feats: np.ndarray of shape (9,), dtype float32.
    """
    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray_image.shape}")

    H, W = gray_image.shape
    total_area = float(H * W)
    eps = 1e-6

    mask = extract_silhouette_mask(gray_image)
    fg_pixels = np.count_nonzero(mask)

    # Safe fallback if mask is empty or nearly empty (<2% of image)
    if fg_pixels < int(0.02 * total_area):
        return np.zeros(SHAPE_FEATURE_DIM, dtype=np.float32)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(SHAPE_FEATURE_DIM, dtype=np.float32)

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area <= 0:
        return np.zeros(SHAPE_FEATURE_DIM, dtype=np.float32)

    perimeter = float(cv2.arcLength(c, closed=True))
    x, y, w_box, h_box = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else area

    M = cv2.moments(c)
    cy = (M["m01"] / M["m00"]) if M["m00"] > 0 else (y + h_box / 2.0)

    # 1. Relative foreground area in [0, 1]
    rel_area = area / total_area

    # 2. Bounding box aspect ratio (normalized to ~[0, 1] using tanh or linear ratio)
    aspect_ratio = float(w_box) / (float(h_box) + eps)

    # 3. Bounding box extent (solidity in box) in [0, 1]
    box_area = float(w_box * h_box)
    extent = area / (box_area + eps)

    # 4. Compactness (circularity) 4*pi*A / P^2 in [0, 1]
    compactness = (4.0 * np.pi * area) / ((perimeter ** 2) + eps)
    compactness = float(np.clip(compactness, 0.0, 1.0))

    # 5. Convexity (solidity) in [0, 1]
    convexity = float(np.clip(area / (hull_area + eps), 0.0, 1.0))

    # 6. Normalized perimeter in [0, 1]
    norm_perimeter = perimeter / (2.0 * (H + W))

    # 7. Vertical centroid elevation in [0, 1]
    vertical_center = float(np.clip(cy / float(H), 0.0, 1.0))

    # 8. Horizontal symmetry ratio
    half_w = W // 2
    left_half = mask[:, :half_w]
    right_half_flipped = np.fliplr(mask[:, half_w:])
    # Match shapes in case W is odd
    min_w = min(left_half.shape[1], right_half_flipped.shape[1])
    l_cut = left_half[:, :min_w]
    r_cut = right_half_flipped[:, :min_w]
    intersection = np.count_nonzero(np.bitwise_and(l_cut, r_cut))
    union = np.count_nonzero(np.bitwise_or(l_cut, r_cut))
    symmetry = float(intersection) / float(union + eps)

    # 9. Top-to-bottom area distribution ratio
    half_h = H // 2
    top_area = np.count_nonzero(mask[:half_h, :])
    bot_area = np.count_nonzero(mask[half_h:, :])
    top_bot_ratio = float(top_area) / float(bot_area + eps)
    norm_top_bot = float(np.tanh(top_bot_ratio))  # mapped to [0, 1]

    feats = np.array([
        rel_area,
        aspect_ratio,
        extent,
        compactness,
        convexity,
        norm_perimeter,
        vertical_center,
        symmetry,
        norm_top_bot
    ], dtype=np.float32)

    assert len(feats) == SHAPE_FEATURE_DIM
    return feats
