# MonuScript: Mathematical Formulas & Digital Image Processing Reference

This document provides a rigorous, syllabus-aligned mathematical breakdown of every digital image processing formula implemented in **MonuScript**.

---

## 1. Digital Image Representation & Grayscale Conversion

### 1.1 ITU-R BT.601 Weighted Luminance Transformation
* **Location in Code:** `backend/image_processing/preprocessing.py` -> `rgb_to_grayscale_syllabus()`
* **Formula:**
  $$I(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$

* **Symbol Definitions:**
  * $R(x, y), G(x, y), B(x, y)$: Red, Green, and Blue spectral channel intensities at discrete spatial coordinate $(x, y) \in [0, M-1] \times [0, N-1]$, where $R, G, B \in [0, 255]$.
  * $I(x, y)$: Grayscale luminance intensity at coordinate $(x, y)$, where $I \in [0, 255]$.
  * $0.299, 0.587, 0.114$: Human photopic spectral sensitivity coefficients matching the higher sensitivity of the human eye to green light, intermediate sensitivity to red, and lowest sensitivity to blue.

* **Purpose & Syllabus Justification:**
  Monochrome representation removes chromatic variation while preserving structural outlines, silhouettes, arches, and architectural boundaries necessary for gradient filtering.

---

## 2. Gray-Level Intensity Transformations

### 2.1 Linear Brightness and Contrast Transformation
* **Location in Code:** `backend/image_processing/transformations.py` -> `apply_brightness_contrast()`
* **Formula:**
  $$s = \text{clip}(\alpha \cdot r + \beta, 0, L - 1)$$

* **Symbol Definitions:**
  * $r$: Input pixel intensity, $r \in [0, L-1]$.
  * $s$: Output transformed pixel intensity, $s \in [0, L-1]$.
  * $\alpha$: Contrast gain factor ($\alpha > 1$ expands dynamic range, $\alpha < 1$ compresses).
  * $\beta$: Brightness offset (additive intensity shift).
  * $L$: Number of gray levels ($L = 256$ for standard 8-bit quantization).

* **Purpose:**
  Used for dataset augmentation and empirical evaluation of model illumination robustness under varying daylight, weather, and camera exposure conditions.

### 2.2 Power-Law (Gamma) Transformation
* **Location in Code:** `backend/image_processing/transformations.py` -> `apply_gamma_transformation()`
* **Formula:**
  $$s = (L - 1) \cdot \left(\frac{r}{L - 1}\right)^\gamma$$

* **Symbol Definitions:**
  * $r$: Normalized input intensity.
  * $\gamma$: Gamma exponent ($\gamma < 1$ brightens dark shadow regions, $\gamma > 1$ compresses highlights).
  * $L - 1$: Dynamic scale factor ($255$).

* **Purpose:**
  Simulates non-linear sensor response and atmospheric haze variations common in outdoor monument photography.

---

## 3. Histogram Processing & Statistical Moments

### 3.1 Normalized Grayscale Histogram
* **Location in Code:** `backend/image_processing/histogram_features.py` -> `compute_normalized_histogram()`
* **Formula:**
  $$h(k) = \frac{n_k}{M \cdot N}, \quad k \in [0, K-1]$$
  such that:
  $$\sum_{k=0}^{K-1} h(k) = 1.0$$

* **Symbol Definitions:**
  * $n_k$: Number of pixels in the image with gray-level intensity belonging to bin $k$.
  * $M$: Image height in pixels.
  * $N$: Image width in pixels.
  * $MN$: Total number of pixels.
  * $K$: Number of quantization bins ($K = 32$).

* **Contribution to Monument Recognition:**
  Provides a scale-invariant probability distribution of surface reflectances. White marble (Taj Mahal) produces a heavily right-skewed histogram (high intensities), whereas red sandstone (Red Fort, Qutub Minar) produces mid-tone distributions.

### 3.2 Statistical Intensity Moments
* **Location in Code:** `backend/image_processing/histogram_features.py` -> `compute_intensity_statistics()`

1. **Mean Intensity ($\mu$):**
   $$\mu = \frac{1}{MN} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x, y)$$
   Measures overall scene brightness.

2. **Intensity Variance ($\sigma^2$):**
   $$\sigma^2 = \frac{1}{MN} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} (I(x, y) - \mu)^2$$
   Measures global contrast and gray-level spread.

3. **Standard Deviation ($\sigma$):**
   $$\sigma = \sqrt{\sigma^2}$$

4. **Smoothness Measure ($R$):**
   $$R = 1 - \frac{1}{1 + \sigma_{\text{norm}}^2}, \quad \text{where } \sigma_{\text{norm}} = \frac{\sigma}{255}$$
   Normalizes variance to $[0, 1)$, where $R \to 0$ denotes smooth flat surfaces (e.g. polished marble or clear sky) and $R \to 1$ denotes highly textured stone.

5. **Uniformity / Energy ($U$):**
   $$U = \sum_{k=0}^{K-1} h(k)^2$$
   Reaches maximum value $1.0$ when all pixels share the same intensity level.

6. **Average Information Entropy ($E$):**
   $$E = -\sum_{k=0}^{K-1} h(k) \log_2 (h(k) + \epsilon)$$
   Measures randomness and complexity of gray-level distribution.

7. **Skewness (Third Central Moment $\mu_3$):**
   $$\mu_3 = \frac{1}{MN} \sum_{x,y} \left(\frac{I(x,y) - \mu}{\sigma}\right)^3$$
   Indicates asymmetry of the intensity distribution about the mean.

---

## 4. Spatial Filtering & Edge Gradient Features

### 4.1 Sobel Discrete Spatial Convolution Kernels
* **Location in Code:** `backend/image_processing/edge_features.py` -> `compute_sobel_gradients()`
* **Kernels:**
  $$S_x = \begin{bmatrix}-1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1\end{bmatrix}, \quad S_y = \begin{bmatrix}-1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1\end{bmatrix}$$

* **Convolution Operations:**
  $$G_x(x, y) = I(x, y) * S_x = \sum_{i=-1}^{1} \sum_{j=-1}^{1} S_x(i, j) \cdot I(x - i, y - j)$$
  $$G_y(x, y) = I(x, y) * S_y = \sum_{i=-1}^{1} \sum_{j=-1}^{1} S_y(i, j) \cdot I(x - i, y - j)$$

* **Gradient Magnitude:**
  $$G(x, y) = \sqrt{G_x(x, y)^2 + G_y(x, y)^2}$$

* **Gradient Direction:**
  $$\theta(x, y) = \text{atan2}(G_y(x, y), G_x(x, y)) \pmod{180^\circ}$$

* **Edge Density ($D_{\text{edge}}$):**
  $$D_{\text{edge}} = \frac{1}{MN} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} \mathbb{I}(G(x, y) > T_{\text{Otsu}})$$

* **Directional Anisotropy Ratio:**
  $$A_{\text{dir}} = \frac{\sum |G_x| - \sum |G_y|}{\sum |G_x| + \sum |G_y| + \epsilon}$$
  Encodes whether horizontal or vertical linear architectural contours dominate. Tall cylindrical towers (Qutub Minar) exhibit strong vertical edge profiles, while wide ramparts (Red Fort) exhibit horizontal balance.

---

## 5. Texture & Spatial Block Grid Structure

### 5.1 Local Neighborhood Variance Filter
* **Location in Code:** `backend/image_processing/texture_features.py` -> `compute_local_std_map()`
* **Formula:**
  $$\sigma_{\text{local}}(x, y) = \sqrt{\frac{1}{W^2} \sum_{(i,j) \in \mathcal{N}_W} I(x+i, y+j)^2 - \left(\frac{1}{W^2} \sum_{(i,j) \in \mathcal{N}_W} I(x+i, y+j)\right)^2}$$
  with window size $W = 5$.

### 5.2 $3 \times 3$ Spatial Partition Grid
* **Location in Code:** `backend/image_processing/texture_features.py` -> `compute_spatial_grid_statistics()`
* Image is subdivided into $3 \times 3 = 9$ spatial cells. Each cell $c \in \{1, \dots, 9\}$ computes:
  1. Mean intensity $\mu_c$
  2. Standard deviation $\sigma_c$
  3. Local edge density $D_c$
  Total: $9 \times 3 = 27$ descriptors encoding the spatial arrangement of architectural components (e.g. sky on top, main dome/minarets in center, reflection pool/plinth on bottom).

---

## 6. Color Space Representation: HSV

### 6.1 HSV Color Channels
* **Location in Code:** `backend/image_processing/feature_extractor.py` -> `extract_hsv_color_features()`
* Decouples chromaticity (Hue $H \in [0, 180)$, Saturation $S \in [0, 255]$) from illumination (Value $V \in [0, 255]$).
* Normalized histograms: Hue (16 bins), Saturation (8 bins), Value (8 bins) -> 32 features.

---

## 7. Contrast-Limited Adaptive Histogram Equalization (CLAHE)
* **Location in Code:** `backend/image_processing/preprocessing.py` -> `apply_clahe_enhancement()`
* Subdivides the image into $M_B \times N_B$ non-overlapping contextual grid blocks ($8 \times 8$).
* Computes local histograms for each tile and clips histogram bin heights at threshold $\beta_{\text{clip}} = 2.0$:
  $$h_{\text{clipped}}(k) = \min(h(k), \beta_{\text{clip}})$$
* Redistributes excess clipped pixels uniformly across all $L = 256$ gray levels before performing local CDF transformation.
* Merges adjacent tile boundaries using bilinear interpolation to eliminate artificial boundary seams.

---

## 8. Monument-Specific Architectural Descriptors
* **Location in Code:** `backend/image_processing/architectural_features.py` -> `extract_architectural_features()`

1. **Central Arch Void Contrast Ratio ($V_{\text{arch}}$):**
   $$V_{\text{arch}} = \frac{\bar{I}_{\text{center}} - \bar{I}_{\text{pylons}}}{\bar{I}_{\text{center}} + \bar{I}_{\text{pylons}} + \epsilon}$$
   Measures luminance contrast between the open central aperture and the solid flanking stone pylons of India Gate.

2. **Upper Cornice Horizontal Dominance ($C_{\text{cornice}}$):**
   $$C_{\text{cornice}} = \frac{\sum |G_y^{\text{upper}}| - \sum |G_x^{\text{upper}}|}{\sum |G_y^{\text{upper}}| + \sum |G_x^{\text{upper}}| + \epsilon}$$
   Quantifies the prominent horizontal attic/entablature of India Gate.

3. **Arch Header Curvature Energy ($E_{\text{header}}$):**
   $$E_{\text{header}} = \frac{1}{|\Omega_{\text{arch}}|} \sum_{(x,y) \in \Omega_{\text{arch}}} G(x, y)$$
   Mean Sobel gradient magnitude along the semicircular arch curve.

4. **Conical Tapering Ratio ($R_{\text{taper}}$):**
   $$R_{\text{taper}} = \frac{w_{\text{top}}}{w_{\text{base}} + \epsilon}$$
   Measures the characteristic tapering diameter of Qutub Minar from base to apex.

5. **Vertical Shaft Gradient Dominance ($V_{\text{shaft}}$):**
   $$V_{\text{shaft}} = \frac{\sum |G_x^{\text{mid}}| - \sum |G_y^{\text{mid}}|}{\sum |G_x^{\text{mid}}| + \sum |G_y^{\text{mid}}| + \epsilon}$$
   Encodes the vertical flute lines of Qutub Minar's shaft.

6. **Solid Central Masonry Continuity ($M_{\text{center}}$):**
   $$M_{\text{center}} = \frac{\bar{G}_{\text{center column}}}{\bar{G}_{\text{full image}} + \epsilon}$$

7. **Jharokha Window Opening Density ($D_{\text{jharokha}}$):**
   $$D_{\text{jharokha}} = \frac{1}{|\Omega_{\text{facade}}|} \sum_{(x,y) \in \Omega_{\text{facade}}} \mathbb{I}\left(\text{TopHat}_{7\times 7}(I)(x,y) > T_{\text{dark}}\right)$$
   Uses morphological Black Top-Hat filtering to count small recessed window openings characteristic of Hawa Mahal.

8. **Honeycomb Facade Grid Heterogeneity ($H_{\text{grid}}$):**
   Spatial variance of local standard deviation across the central facade grid.

9. **Dome Circularity Profile ($C_{\text{dome}}$):**
   Circular Hough Transform accumulator response ($cv2.HOUGH\_GRADIENT$) in the upper quadrant for Taj Mahal's central onion dome.

10. **White Marble Spectral Ratio ($W_{\text{marble}}$):**
    $$W_{\text{marble}} = \frac{1}{MN} \sum_{x,y} \mathbb{I}(V(x,y) > 160 \;\land\; S(x,y) < 65)$$

11. **Bilateral Reflection Symmetry Score ($S_{\text{bilat}}$):**
    $$S_{\text{bilat}} = 1.0 - \frac{1}{M \cdot (N/2)} \sum_{x=0}^{M-1} \sum_{y=0}^{N/2 - 1} |I(x, y) - I(x, N - 1 - y)|$$

12. **Four-Corner Structural Balance ($B_{\text{corners}}$):**
    Normalized inverse standard deviation of gradient energies across the four outer image corners (Charminar minaret symmetry).

13. **Horizontal Curtain Wall Continuity ($W_{\text{curtain}}$):**
    Morphological linear opening with horizontal structuring element ($17 \times 1$) detecting long fortress ramparts (Red Fort).

14. **Sandstone Warmth Index ($I_{\text{warmth}}$):**
    $$I_{\text{warmth}} = \frac{1}{MN} \sum_{x,y} \frac{R(x,y) - B(x,y)}{R(x,y) + G(x,y) + B(x,y) + \epsilon}$$

15. **Central Aperture Edge Contrast ($K_{\text{aperture}}$):**
    Contrast between inner arch opening contours and exterior masonry boundaries.

16. **Vertical Slenderness Aspect Ratio ($A_{\text{slender}}$):**
    Bounding mass height divided by base width.

---

## 9. Machine Learning (SVM) & Probability Calibration Formulas
* **Location in Code:** `backend/ml/train_model.py`, `backend/ml/inference.py`

### 9.1 Support Vector Machine RBF Dual Formulation
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(\mathbf{z}_i, \mathbf{z}_j)$$
subject to:
$$0 \le \alpha_i \le C, \quad \sum_{i=1}^n \alpha_i y_i = 0$$
where $K(\mathbf{z}_i, \mathbf{z}_j) = \exp(-\gamma \|\mathbf{z}_i - \mathbf{z}_j\|^2)$, $C = 2.0$, and standardized features $\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$.

### 9.2 Platt Scaling Probability Calibration
Transforms uncalibrated decision function scores $f_k(\mathbf{x})$ into well-calibrated class posterior probabilities:
$$P(y = k \mid \mathbf{x}) = \frac{1}{1 + \exp(A \cdot f_k(\mathbf{x}) + B)}$$
Parameters $A$ and $B$ are estimated by maximum likelihood via 5-fold cross-validation on the training set.

### 9.3 Multi-Tier Rejection Filter
A prediction is accepted if and only if:
1. $P_{(1)} \ge 0.42$ (minimum confidence)
2. $P_{(1)} - P_{(2)} \ge 0.12$ (decision margin)
3. $\|\mathbf{z} - \mathbf{c}_{k^*}\|_2 \le \tau_{\text{dist}}$ (in-distribution centroid distance)

---

## 10. Complete Feature Vector Summary

$$\mathbf{x} = [\mathbf{h}_{\text{hist (40)}} \,\|\, \mathbf{h}_{\text{edge (36)}} \,\|\, \mathbf{h}_{\text{texture (31)}} \,\|\, \mathbf{h}_{\text{color (32)}} \,\|\, \mathbf{h}_{\text{arch (16)}}]^T \in \mathbb{R}^{155}$$

Every feature is deterministic, fixed-length, and computed in identical order during training and inference.
