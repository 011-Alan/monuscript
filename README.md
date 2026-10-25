# MonuScript

> **Syllabus-Based Indian Monument Recognition and Exploration System**  
> An academic digital image processing project utilizing classical spatial filters, statistical intensity moments, morphological operators, color histograms, and a Support Vector Machine (SVM) classifier.

---

## 1. Overview
**MonuScript** identifies supported Indian heritage monuments from uploaded photographs using **traditional digital image processing (DIP) techniques** and a trained **Support Vector Machine (SVM)** classifier.

In strict compliance with academic syllabus constraints:
* **No Deep Learning:** Zero convolutional neural networks (CNNs), Vision Transformers (ViTs), YOLO, or pretrained vision foundation models.
* **No External Vision APIs:** Zero cloud-based recognition endpoints or generative AI image models.
* **Deterministic Mathematical Features:** 155 handcrafted features derived from ITU-R BT.601 grayscale conversion, Contrast-Limited Adaptive Histogram Equalization (CLAHE), gray-level transformations, normalized intensity histograms, Sobel spatial gradient filters, texture variance maps, 3x3 spatial layout grids, HSV color histograms, and monument-specific architectural descriptors.
* **Confidence & Rejection Mechanism:** Calibrated multi-tier rejection filter that rejects non-monument, blurry, or ambiguous photographs with friendly explanations instead of forcing an incorrect prediction.
* **Exploration Workspace:** Persistent chat history, interactive 3D monument models (via Sketchfab), verified historical details, and a feature inspection modal.

---

## 2. Supported Monuments
The system supports 6 iconic Indian national monuments with distinct architectural styles:
1. **Taj Mahal** (Agra, Uttar Pradesh) — Mughal white Makrana marble, onion dome, four corner minarets.
2. **Qutub Minar** (Mehrauli, New Delhi) — Fluted red/buff sandstone minaret, balconies, calligraphy.
3. **Red Fort (Lal Qila)** (Old Delhi, Delhi) — Massive red sandstone ramparts, Lahori Gate.
4. **India Gate** (New Delhi) — Triumphal arch war memorial, pale red & yellow sandstone.
5. **Charminar** (Hyderabad, Telangana) — Qutb Shahi granite arches, four 48.7m minarets.
6. **Hawa Mahal** (Jaipur, Rajasthan) — Honeycomb facade with 953 casement windows (jharokhas).

---

## 3. Technology Stack
* **Frontend:** React 18, TypeScript, Vite, Tailwind CSS, Lucide React icons.
* **Backend:** Python 3.11, Flask, Flask-CORS, OpenCV, NumPy, Pillow, scikit-learn, SciPy, Joblib.
* **Storage & Persistence:** SQLite (default local mode with zero configuration needed) + Supabase PostgreSQL schema with RLS policies.
* **Testing:** Pytest (36 automated tests passing across 5 test suites).

---

## 4. Quick Start & Installation

### Prerequisites
* Python 3.10+
* Node.js 18+ and npm

### 4.1 Clone Repository
```bash
git clone https://github.com/011-Alan/monuscript.git
cd monuscript
```

### 4.2 Backend Setup
1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```
2. Install Python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Prepare dataset from Wikimedia Commons (downloads verified authentic photos):
   ```bash
   python backend/ml/prepare_dataset.py
   ```
4. Train the Support Vector Machine classifier:
   ```bash
   python backend/ml/train_model.py
   ```
5. Run automated evaluation on the held-out test set:
   ```bash
   python backend/ml/evaluate_model.py
   ```
6. Start the Flask backend server:
   ```bash
   python backend/app.py
   ```
   *The backend will be live at `http://127.0.0.1:5000`.*

### 4.3 Frontend Setup
In a new terminal window:
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   *Open `http://localhost:5173` in your browser.*

---

## 5. Automated Tests
Run the comprehensive automated test suite (image preprocessing, feature vectors, inference, API endpoints):
```bash
pytest backend/tests
```
All **36 automated tests** verify mathematical formulas, bounds, CLAHE preprocessing, architectural features, inference calibration, and error scenarios.

---

## 6. Digital Image Processing Techniques & Formulas

Every feature used by MonuScript is deterministic, syllabus-grounded, and extracted without neural networks. For complete mathematical documentation, symbol definitions, and code references, see [`docs/FORMULAS.md`](docs/FORMULAS.md).

### 6.1 Preprocessing & Image Enhancements
* **Grayscale Conversion (ITU-R BT.601 Weighted Luminance):**  
  $$I(x,y) = 0.299R(x,y) + 0.587G(x,y) + 0.114B(x,y)$$
  Removes chromatic variation while preserving luminance and structural contours.
* **Bilinear Interpolation Spatial Normalization:**  
  All images are rescaled to a standardized spatial resolution of $256 \times 256$ pixels.
* **Contrast-Limited Adaptive Histogram Equalization (CLAHE):**  
  Equalizes local contrast across $8 \times 8$ contextual tiles with a clip limit of $2.0$ to mitigate harsh shadows and overexposure without amplifying noise.
* **Linear Brightness/Contrast Transformation:**  
  $$s = \text{clip}(\alpha r + \beta, 0, L-1)$$
* **Power-Law (Gamma) Illumination Augmentation:**  
  $$s = (L-1)\left(\frac{r}{L-1}\right)^\gamma, \quad \gamma \in [0.75, 1.35]$$

### 6.2 Intensity Histograms & Statistical Moments (40 features)
* **Normalized Grayscale Histogram (32 bins):**  
  $$h(k) = \frac{n_k}{M \cdot N}, \quad \sum_{k=0}^{K-1} h(k) = 1.0$$
* **Statistical Moments (8 values):**  
  - Mean ($\mu$), Variance ($\sigma^2$), Standard Deviation ($\sigma$).
  - Smoothness metric: $R = 1 - \frac{1}{1 + \sigma_{\text{norm}}^2}$.
  - Uniformity / Energy: $U = \sum_{k} h(k)^2$.
  - Average Information Entropy: $E = -\sum_{k} h(k) \log_2 (h(k) + \epsilon)$.
  - Skewness (third central moment): $\mu_3 = \frac{1}{MN} \sum \left(\frac{I(x,y) - \mu}{\sigma}\right)^3$.

### 6.3 Spatial Filtering & Sobel Edge Derivatives (36 features)
* **Sobel Convolution Kernels:**  
  $$S_x = \begin{bmatrix}-1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1\end{bmatrix}, \quad S_y = \begin{bmatrix}-1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1\end{bmatrix}$$
* **Gradient Magnitude & Orientation:**  
  $$G(x,y) = \sqrt{(I * S_x)^2 + (I * S_y)^2}, \quad \theta(x,y) = \text{atan2}(G_y, G_x) \pmod{180^\circ}$$
* **Edge Descriptors:**  
  - 16-bin gradient magnitude histogram.
  - 16-bin gradient orientation histogram (magnitude-weighted).
  - Otsu thresholded global edge density $D_{\text{edge}}$.
  - Directional Anisotropy Ratio: $A_{\text{dir}} = \frac{\sum |G_x| - \sum |G_y|}{\sum |G_x| + \sum |G_y| + \epsilon}$ (distinguishes tall vertical structures like Qutub Minar from horizontal fortifications like Red Fort).

### 6.4 Texture & Spatial Layout Grid (31 features)
* **Local Neighborhood Variance Map:**  
  $$\sigma_{\text{local}}(x, y) = \sqrt{\frac{1}{W^2} \sum_{\mathcal{N}_W} I^2 - \left(\frac{1}{W^2} \sum_{\mathcal{N}_W} I\right)^2}, \quad W = 5$$
* **$3 \times 3$ Spatial Partition Grid:**  
  Divides the image into 9 spatial quadrants to capture spatial composition (sky at top, monument mass in center, grounds/water at base). Computes cell mean $\mu_c$, standard deviation $\sigma_c$, and local edge density $D_c$ ($9 \times 3 = 27$ descriptors).

### 6.5 Color Space Representation: HSV (32 features)
* Decouples chromaticity from luminance into Hue ($H \in [0, 180)$), Saturation ($S \in [0, 255]$), and Value ($V \in [0, 255]$).
* Normalized Histograms: Hue (16 bins), Saturation (8 bins), Value (8 bins) = 32 color features.

### 6.6 Monument-Specific Architectural Descriptors (16 features)
Domain-specific descriptors that resolve ambiguities between visually similar monuments:
1. **India Gate Arch Void Ratio:** Luminance contrast between the central arch void and flanking stone pylons.
2. **India Gate Cornice Dominance:** Ratio of upper-third horizontal edges to vertical edges (heavy horizontal entablature).
3. **India Gate Arch Curve Energy:** Mean gradient energy along the semicircular arch header.
4. **Qutub Minar Tapering Ratio:** Ratio of upper shaft width to base width ($w_{\text{top}} / w_{\text{base}}$), capturing the conical taper.
5. **Qutub Minar Vertical Shaft Dominance:** Ratio of vertical to horizontal gradient energy in the mid-section.
6. **Qutub Minar Solid Center Continuity:** Continuity of high-density masonry through the central column.
7. **Hawa Mahal Jharokha Window Opening Density:** Black Top-Hat morphological filtering ($7 \times 7$ structuring element) capturing the dense grid of 953 recessed casements.
8. **Hawa Mahal Facade Grid Heterogeneity:** Spatial variance of the lattice texture across the facade.
9. **Taj Mahal Dome Circularity Profile:** Circular Hough Transform ($cv2.HOUGH\_GRADIENT$) in the upper central quadrant.
10. **Taj Mahal White Marble Ratio:** Proportion of pixels with high luminance ($V > 160$) and low saturation ($S < 65$).
11. **Bilateral Reflection Symmetry Score:** Horizontal reflection symmetry score across the vertical center axis: $1 - \frac{1}{HW} \sum |I(x, y) - I(x, W - y)|$.
12. **Charminar Four-Corner Minaret Balance:** Variance of edge gradient energy across the four outer image corners.
13. **Red Fort Horizontal Wall Continuity:** Linear morphological opening ($17 \times 1$ structuring element) measuring long horizontal rampart walls.
14. **Red Sandstone Warmth Index:** Spectral dominance ratio $\frac{R - B}{R + G + B + \epsilon}$ distinguishing red/buff sandstone from white marble or gray granite.
15. **Central Aperture Edge Contrast:** Edge gradient difference between central void boundary and side pylons.
16. **Vertical Slenderness Aspect Ratio:** Height-to-width ratio of the segmented architectural silhouette.

**Total Feature Vector Dimension:** **155 deterministic features** ($40 + 36 + 31 + 32 + 16$).

---

## 7. Machine Learning (SVM) & Rejection Architecture

### 7.1 Classifier Specification
* **Algorithm:** Support Vector Machine (SVM) using `scikit-learn`'s `SVC`.
* **Kernel:** Radial Basis Function (RBF Kernel):  
  $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$
* **Hyperparameters:** Penalty parameter $C = 2.0$, kernel coefficient $\gamma = \text{'scale'} = \frac{1}{d \cdot \text{Var}(X)}$, with $d = 155$.
* **Feature Scaling:** `StandardScaler` applies zero-mean, unit-variance standardization:  
  $$z = \frac{x - \mu}{\sigma}$$
  Scaler parameters ($\mu, \sigma$) are fitted strictly on the training set and serialized with the model artifact.

### 7.2 Probability Calibration (Platt Scaling)
Standard SVM decision values $f_k(\mathbf{x})$ represent uncalibrated signed distances to separating hyperplanes. MonuScript fits a logistic sigmoid transformation via 5-fold cross-validation:
$$P(y = k \mid \mathbf{x}) = \frac{1}{1 + \exp(A \cdot f_k(\mathbf{x}) + B)}$$
Calibrated probabilities sum to $1.0$ across all 6 monument classes, providing interpretable confidence scores.

### 7.3 Multi-Tier Rejection & Out-of-Distribution Guardrail
To prevent classifying non-monuments, arbitrary objects, or highly ambiguous images, MonuScript evaluates every prediction against a calibrated three-tier rejection filter:
1. **Confidence Threshold:** The top-ranked class probability must meet $P_{(1)} \ge 0.42$.
2. **Confidence Margin Threshold:** The margin between top-1 and top-2 class probabilities must satisfy $\Delta P = P_{(1)} - P_{(2)} \ge 0.12$.
3. **Centroid Distance Guard:** Evaluates the Euclidean distance to the predicted class centroid in normalized feature space to reject out-of-distribution inputs.

If an image fails any rejection criterion, the system gracefully triggers the **Rejection Filter**, explaining the ambiguity and presenting top candidate resemblance percentages rather than returning a misleading classification.

---

## 8. 3D Monument Models & Attributions
Interactive 3D monument visualizations in MonuScript are embedded via [Sketchfab](https://sketchfab.com) and created by 3D artists. All models are credited with their respective creators and license terms below:

| Monument | 3D Model Title | Artist / Creator | Platform & Model Link | License |
| :--- | :--- | :--- | :--- | :--- |
| **Taj Mahal** | [Taj Mahal](https://sketchfab.com/3d-models/taj-mahal-33149233cefd492b9abdd50fe5a8c921) | [uday14viru](https://sketchfab.com/uday14viru) | Sketchfab | Editorial / Standard |
| **Qutub Minar** | [Qutub Minar](https://sketchfab.com/3d-models/qutub-minar-e165e25ec8f1497aa83ab72419f40ddc) | [3D Dreamscapes](https://sketchfab.com/pradeep1) | Sketchfab | Standard License |
| **Red Fort (Lal Qila)** | [Red Fort- Delhi](https://sketchfab.com/3d-models/red-fort-delhi-74ff6d703a174f9fb1ba266003f5c4fc) | [STHAPATYA](https://sketchfab.com/Sthapatya) | Sketchfab | Standard License |
| **India Gate** | [India Gate](https://sketchfab.com/3d-models/india-gate-59fe55328271479d82acb65310178d99) | [Macwelshman](https://sketchfab.com/Macwelshman) | Sketchfab | Standard License |
| **Charminar** | [Charminar, Hyderabad](https://sketchfab.com/3d-models/charminar-hyderabad-82603f1fe3ed4c31b37786e2bc2b63b0) | [Dolores](https://sketchfab.com/upto21) | Sketchfab | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) |
| **Hawa Mahal** | [hawa mahal 3d model](https://sketchfab.com/3d-models/hawa-mahal-3d-model-c6bbf5822010490d995869ae625b1be0) | [meigandan](https://sketchfab.com/meigandan) | Sketchfab | Standard License |

> We express our gratitude to the 3D artists and heritage preservationists on Sketchfab for their architectural digital reconstructions of Indian monuments.

---

## 9. Documentation Index
* [`docs/FORMULAS.md`](docs/FORMULAS.md): Detailed syllabus formula derivations, architectural descriptors, and ML equations.
* [`docs/DATASET.md`](docs/DATASET.md): Dataset catalog, licenses, partitioning, and leakage prevention.
* [`docs/TESTING.md`](docs/TESTING.md): Measured test set accuracy, confusion matrices, latency, and robustness.
* [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md): Handcrafted features vs deep learning, class overlaps, sliding window discussion.
* [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md): Architecture report and schema specifications.

---

## 10. License
Academic project released under the MIT License. Monument images sourced from Wikimedia Commons under Creative Commons licenses (see `backend/data/dataset_catalog.json`).
