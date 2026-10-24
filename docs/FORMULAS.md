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

## 7. Complete Feature Vector Summary

$$\mathbf{x} = [\mathbf{h}_{\text{hist (40)}} \,\|\, \mathbf{h}_{\text{edge (36)}} \,\|\, \mathbf{h}_{\text{texture (31)}} \,\|\, \mathbf{h}_{\text{color (32)}}]^T \in \mathbb{R}^{139}$$

Every feature is deterministic, fixed-length, and computed in identical order during training and inference.
