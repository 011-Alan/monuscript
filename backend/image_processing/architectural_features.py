"""
MonuScript - Monument-Specific Architectural Feature Extraction Module
Extracts domain-specific architectural descriptors derived from structural geometry,
spatial layout, morphological openings, and boundary contours to distinguish
visually similar monuments (especially India Gate vs Qutub Minar vs Hawa Mahal).

Adheres strictly to traditional Digital Image Processing (DIP) syllabus:
- Unit I: Spatial coordinates, pixel connectivity, distance measures
- Unit II: Spatial filtering, directional derivatives, morphological structuring elements
- Unit IV: Boundary detection, Hough transforms, regional moments, compactness

Features (16 dimensions):
1.  india_gate_arch_void_ratio: Center opening luminance vs flanking pylons
2.  india_gate_cornice_dominance: Upper-third horizontal cornice prominence
3.  india_gate_arch_curve_energy: Arch header curvature energy
4.  qutub_minar_tapering_ratio: Upper-slice width vs lower-slice width (conical taper)
5.  qutub_minar_vertical_shaft_dominance: Middle-third vertical gradient prominence
6.  qutub_minar_solid_center_continuity: Solid masonry continuity down the center
7.  hawa_mahal_window_opening_density: Density of small recessed jharokha window openings
8.  hawa_mahal_facade_grid_heterogeneity: Spatial variance of repetitive window lattice
9.  taj_mahal_dome_circularity: Circular Hough gradient profile in upper central quadrant
10. taj_mahal_marble_whiteness_ratio: Proportion of high-luminance, low-saturation pixels
11. bilateral_symmetry_score: Horizontal reflection symmetry score
12. charminar_corner_minaret_balance: Gradient energy balance across 4 outer corners
13. red_fort_horizontal_wall_continuity: Long horizontal edge line continuity (ramparts)
14. red_sandstone_warmth_index: Red/yellow sandstone spectral dominance
15. central_aperture_edge_contrast: Contrast between center void edges and pylon edges
16. vertical_slenderness_aspect: Height-to-width ratio of primary architectural mass
"""

import numpy as np
import cv2

ARCHITECTURAL_FEATURE_DIM = 16
ARCHITECTURAL_FEATURE_NAMES = [
    "india_gate_arch_void_ratio",
    "india_gate_cornice_dominance",
    "india_gate_arch_curve_energy",
    "qutub_minar_tapering_ratio",
    "qutub_minar_vertical_shaft_dominance",
    "qutub_minar_solid_center_continuity",
    "hawa_mahal_window_opening_density",
    "hawa_mahal_facade_grid_heterogeneity",
    "taj_mahal_dome_circularity",
    "taj_mahal_marble_whiteness_ratio",
    "bilateral_symmetry_score",
    "charminar_corner_minaret_balance",
    "red_fort_horizontal_wall_continuity",
    "red_sandstone_warmth_index",
    "central_aperture_edge_contrast",
    "vertical_slenderness_aspect"
]


def extract_architectural_features(gray_image, rgb_image=None):
    """
    Extracts 16 deterministic, scale-invariant architectural descriptors.

    Parameters:
        gray_image: np.ndarray of shape (H, W), dtype uint8.
        rgb_image: optional np.ndarray of shape (H, W, 3), dtype uint8 for color warmth.
    Returns:
        feats: np.ndarray of shape (16,), dtype float32, guaranteed finite and normalized.
    """
    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray_image.shape}")

    H, W = gray_image.shape
    I = gray_image.astype(np.float32) / 255.0

    # Handle degenerate/blank images safely
    if np.max(gray_image) == 0:
        return np.zeros(ARCHITECTURAL_FEATURE_DIM, dtype=np.float32)

    # 1. Spatial 2D Sobel derivatives
    Gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    G = np.sqrt(Gx ** 2 + Gy ** 2)

    # Coordinates for central aperture vs flanking pylons
    r1, r2 = int(H * 0.40), int(H * 0.85)
    c_mid_1, c_mid_2 = int(W * 0.35), int(W * 0.65)
    c_left_1, c_left_2 = int(W * 0.15), int(W * 0.35)
    c_right_1, c_right_2 = int(W * 0.65), int(W * 0.85)

    # 1. India Gate: Central Arch Void Luminance Ratio
    arch_center_lum = float(np.mean(I[r1:r2, c_mid_1:c_mid_2]))
    pylon_lum = float(0.5 * (np.mean(I[r1:r2, c_left_1:c_left_2]) + np.mean(I[r1:r2, c_right_1:c_right_2])) + 1e-4)
    arch_void_contrast = float((arch_center_lum - pylon_lum) / (arch_center_lum + pylon_lum + 1e-4))

    # 2. India Gate: Upper-Third Horizontal Cornice Prominence
    t1 = int(H * 0.33)
    t2 = int(H * 0.66)
    gx_upper = np.sum(np.abs(Gx[:t1, :]))
    gy_upper = np.sum(np.abs(Gy[:t1, :]))
    cornice_dominance = float((gy_upper - gx_upper) / (gx_upper + gy_upper + 1e-6))

    # 3. India Gate: Arch Header Curvature Energy
    arch_header_g = G[int(H * 0.35):int(H * 0.55), c_mid_1:c_mid_2]
    arch_curve_energy = float(np.mean(arch_header_g)) / 255.0

    # 4. Qutub Minar: Conical Tapering Ratio (width at top vs base)
    line_energy = np.mean(G, axis=1)
    active_rows = np.where(line_energy > np.mean(line_energy) * 0.5)[0]
    if len(active_rows) > 10:
        row_top = active_rows[0]
        row_bot = active_rows[-1]
    else:
        row_top, row_bot = int(H * 0.1), int(H * 0.9)

    top_slice = G[row_top:row_top + int(H * 0.15), :]
    bot_slice = G[max(0, row_bot - int(H * 0.15)):row_bot, :]
    top_cols = np.where(np.mean(top_slice, axis=0) > 20.0)[0]
    bot_cols = np.where(np.mean(bot_slice, axis=0) > 20.0)[0]
    w_top = float(top_cols[-1] - top_cols[0]) if len(top_cols) > 2 else float(W * 0.3)
    w_bot = float(bot_cols[-1] - bot_cols[0]) if len(bot_cols) > 2 else float(W * 0.6)
    tapering_ratio = float(w_top / (w_bot + 1e-4))
    tapering_ratio = float(np.clip(tapering_ratio, 0.0, 2.0))

    # 5. Qutub Minar: Middle-Third Vertical Shaft Dominance
    gx_mid = np.sum(np.abs(Gx[t1:t2, :]))
    gy_mid = np.sum(np.abs(Gy[t1:t2, :]))
    vertical_shaft_dominance = float((gx_mid - gy_mid) / (gx_mid + gy_mid + 1e-6))

    # 6. Qutub Minar: Solid Center Column Concentration (Tower concentrated in center columns vs sky flanks)
    mid_e = float(np.mean(G[:, c_mid_1:c_mid_2]))
    side_e = float(0.5 * (np.mean(G[:, :c_left_2]) + np.mean(G[:, c_right_1:])) + 1e-4)
    solid_center_continuity = float(np.clip(mid_e / side_e, 0.0, 5.0)) / 5.0

    # 7. Hawa Mahal: Window Opening / Jharokha Density (Black Top-Hat)
    k_window = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    tophat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, k_window)
    facade_tophat = tophat[t1:t2, int(W * 0.2):int(W * 0.8)]
    window_opening_density = float(np.count_nonzero(facade_tophat > 25)) / float(facade_tophat.size + 1e-6)

    # 8. Hawa Mahal: Facade Lattice Texture Heterogeneity
    facade_std_map = cv2.filter2D(I[t1:t2, int(W * 0.2):int(W * 0.8)] ** 2, -1, np.ones((5, 5))/25.0)
    facade_grid_heterogeneity = float(np.var(facade_std_map))

    # 9. Taj Mahal: Dome Circularity in Upper Central Corridor
    upper_gray = gray_image[int(H * 0.08):int(H * 0.50), int(W * 0.30):int(W * 0.70)]
    if upper_gray.shape[0] > 20 and upper_gray.shape[1] > 20:
        circles = cv2.HoughCircles(
            upper_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=60, param2=38, minRadius=15, maxRadius=int(W * 0.25)
        )
        dome_circularity = 1.0 if circles is not None else 0.0
    else:
        dome_circularity = 0.0

    # 10. Taj Mahal: White Marble High-Luminance Ratio (masked to structural/non-sky pixels)
    struct_mask = (G > 14.0) | (np.arange(H)[:, None] > int(H * 0.45))
    if rgb_image is not None and rgb_image.ndim == 3:
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        white_mask = (hsv[:, :, 2] > 165) & (hsv[:, :, 1] < 60) & struct_mask
        marble_whiteness_ratio = float(np.sum(white_mask)) / float(np.sum(struct_mask) + 1e-5)
    else:
        white_mask = (I > 0.72) & struct_mask
        marble_whiteness_ratio = float(np.sum(white_mask)) / float(np.sum(struct_mask) + 1e-5)

    # 11. Bilateral Reflection Symmetry Score
    left_half = I[:, :W // 2]
    right_half_flipped = np.fliplr(I[:, W // 2:])
    min_w = min(left_half.shape[1], right_half_flipped.shape[1])
    diff = np.abs(left_half[:, :min_w] - right_half_flipped[:, :min_w])
    symmetry_score = float(1.0 - np.mean(diff))

    # 12. Charminar: Four-Corner Structural Minaret Balance
    corner_size = int(min(H, W) * 0.25)
    c_tl = np.mean(G[:corner_size, :corner_size])
    c_tr = np.mean(G[:corner_size, -corner_size:])
    c_bl = np.mean(G[-corner_size:, :corner_size])
    c_br = np.mean(G[-corner_size:, -corner_size:])
    c_mean = 0.25 * (c_tl + c_tr + c_bl + c_br)
    corner_minaret_balance = float(1.0 - (np.std([c_tl, c_tr, c_bl, c_br]) / (c_mean + 1e-4)))

    # 13. Red Fort: Horizontal Wall Continuity (Curtain Walls)
    k_horiz_line = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
    horiz_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, k_horiz_line)
    horizontal_wall_continuity = float(np.mean(horiz_lines)) / 255.0

    # 14. Red-to-Buff Sandstone Warmth Index
    if rgb_image is not None and rgb_image.ndim == 3:
        R = rgb_image[:, :, 0].astype(np.float32)
        G_chan = rgb_image[:, :, 1].astype(np.float32)
        B = rgb_image[:, :, 2].astype(np.float32)
        warmth = np.mean((R - B) / (R + G_chan + B + 1e-4))
        sandstone_warmth_index = float(np.clip(warmth, -1.0, 1.0))
    else:
        sandstone_warmth_index = 0.0

    # 15. Central Aperture Edge Contrast (Open arch has clear boundary)
    arch_edge_center = np.mean(G[r1:r2, c_mid_1:c_mid_2])
    pylon_edge_sides = 0.5 * (np.mean(G[r1:r2, c_left_1:c_left_2]) + np.mean(G[r1:r2, c_right_1:c_right_2])) + 1e-4
    aperture_edge_contrast = float((pylon_edge_sides - arch_edge_center) / (pylon_edge_sides + arch_edge_center + 1e-4))

    # 16. Vertical Slenderness Aspect Ratio (Height of active mass vs width)
    col_e = np.mean(G, axis=0)
    row_e = np.mean(G, axis=1)
    w_active = float(np.count_nonzero(col_e > np.mean(col_e) * 0.6))
    h_active = float(np.count_nonzero(row_e > np.mean(row_e) * 0.6))
    slenderness_aspect = float(np.clip(h_active / (w_active + 1e-4), 0.0, 5.0))

    features = np.array([
        arch_void_contrast,
        cornice_dominance,
        arch_curve_energy,
        tapering_ratio,
        vertical_shaft_dominance,
        solid_center_continuity,
        window_opening_density,
        facade_grid_heterogeneity,
        dome_circularity,
        marble_whiteness_ratio,
        symmetry_score,
        corner_minaret_balance,
        horizontal_wall_continuity,
        sandstone_warmth_index,
        aperture_edge_contrast,
        slenderness_aspect
    ], dtype=np.float32)

    # Sanitize against any NaN or Inf
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    assert len(features) == ARCHITECTURAL_FEATURE_DIM
    return features
