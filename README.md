# MonuScript

> **Syllabus-Based Indian Monument Recognition and Exploration System**  
> An academic digital image processing project utilizing classical spatial filters, statistical intensity moments, color histograms, and a Support Vector Machine (SVM) classifier.

---

## 1. Overview
**MonuScript** identifies supported Indian heritage monuments from uploaded photographs using **traditional digital image processing (DIP) techniques** and a trained **Support Vector Machine (SVM)** classifier.

In strict compliance with the academic syllabus constraints:
* **No Deep Learning:** Zero convolutional neural networks (CNNs), Vision Transformers (ViTs), YOLO, or pretrained vision foundation models.
* **No External Vision APIs:** Zero cloud-based recognition endpoints or generative AI image models.
* **Deterministic Mathematical Features:** 139 handcrafted features derived from ITU-R BT.601 grayscale conversion, gray-level transformations, normalized intensity histograms, Sobel spatial gradient filters, texture variance maps, 3x3 spatial layout grids, and HSV color histograms.
* **Confidence & Rejection Mechanism:** Rejects non-monument, unrelated, or ambiguous photographs with friendly explanations instead of forcing an incorrect prediction.
* **Exploration Workspace:** Persistent chat history, interactive 3D monument models (via Sketchfab), verified historical details, and a feature inspection modal.

---

## 2. Supported Monuments
The system supports 6 iconic Indian national monuments with distinct architectural styles:
1. **Taj Mahal** (Agra, Uttar Pradesh) — Mughal white Makrana marble, onion dome, four minarets.
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
* **Testing:** Pytest (18 automated tests passing).

---

## 4. Quick Start & Installation

### Prerequisites
* Python 3.10+
* Node.js 18+ and npm

### 4.1 Clone Repository
```bash
git clone https://github.com/example/monuscript.git
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
All 18 automated tests verify mathematical formulas, bounds, error scenarios, and persistence lifecycle.

---

## 6. Digital Image Processing Formulas
For full mathematical documentation including symbol definitions, derivations, and code locations, see [`docs/FORMULAS.md`](docs/FORMULAS.md).

* **Grayscale Conversion:**  
  $$I(x,y) = 0.299R(x,y) + 0.587G(x,y) + 0.114B(x,y)$$
* **Linear Intensity Transformation:**  
  $$s = \text{clip}(\alpha r + \beta, 0, L-1)$$
* **Gamma Power-Law:**  
  $$s = (L-1)\left(\frac{r}{L-1}\right)^\gamma$$
* **Normalized Grayscale Histogram (32 bins):**  
  $$h(k) = \frac{n_k}{M \cdot N}$$
* **Sobel Spatial Convolution & Gradient Magnitude:**  
  $$G = \sqrt{(I * S_x)^2 + (I * S_y)^2}, \quad \theta = \text{atan2}(G_y, G_x)$$
* **Local Neighborhood Variance & Texture Coarseness:**  
  $$\sigma_{\text{local}}(x, y) = \sqrt{E[I^2] - (E[I])^2}$$
* **$3 \times 3$ Spatial Layout Grid:** 9 spatial cells $\times$ 3 descriptors (mean, std dev, edge density).
* **HSV Color Histograms:** Hue (16 bins), Saturation (8 bins), Value (8 bins).
* **Total Dimension:** 139 deterministic features.

---

## 7. Documentation Index
* [`docs/FORMULAS.md`](docs/FORMULAS.md): Detailed syllabus formula derivations and symbol definitions.
* [`docs/DATASET.md`](docs/DATASET.md): Dataset catalog, licenses, partitioning, and leakage prevention.
* [`docs/TESTING.md`](docs/TESTING.md): Measured test set accuracy, confusion matrices, latency, and robustness.
* [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md): Handcrafted features vs deep learning, class overlaps, sliding window discussion.
* [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md): Architecture report and schema specifications.

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

## 9. License
Academic project released under the MIT License. Monument images sourced from Wikimedia Commons under Creative Commons licenses (see `backend/data/dataset_catalog.json`).
