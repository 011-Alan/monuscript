# MonuScript: Dataset Specification & Preparation

## 1. Overview
The **MonuScript** dataset comprises authentic, high-resolution photographs of supported Indian national heritage monuments along with a negative out-of-distribution set, gathered from **Wikimedia Commons** under Creative Commons (CC BY-SA / CC BY / CC0) and Public Domain licenses.

Zero artificial/synthetic generative images were used for training.

---

## 2. Supported Monument Classes
Six distinct architectural monuments across India were selected for the academic prototype:

| Class ID | Monument Name | Location | Primary Architectural Style | Key Materials |
|---|---|---|---|---|
| `taj_mahal` | Taj Mahal | Agra, Uttar Pradesh | Mughal (Symmetrical) | White Makrana Marble |
| `qutub_minar` | Qutub Minar | Mehrauli, New Delhi | Indo-Islamic (Mamluk/Ghurid) | Red/Buff Sandstone & Marble |
| `red_fort` | Red Fort (Lal Qila) | Old Delhi, Delhi | Mughal Fortified | Red Sandstone Ramparts |
| `india_gate` | India Gate | New Delhi | Neo-Classical Triumphal Arch | Pale Red & Yellow Sandstone |
| `charminar` | Charminar | Hyderabad, Telangana | Indo-Islamic (Qutb Shahi) | Granite, Lime Mortar |
| `hawa_mahal` | Hawa Mahal | Jaipur, Rajasthan | Rajput & Mughal Synthesis | Pink & Red Sandstone |

---

## 3. Dataset Partitions & Counts

| Partition | Samples per Class | Total Supported Samples | Purpose |
|---|---|---|---|
| `train` | 12 | 72 | Model parameter training & StandardScaler fitting |
| `val` | 2 | 12 | Kernel tuning ($C$, $\gamma$) & threshold selection |
| `test` | 4 | 24 | Held-out benchmark evaluation (never seen during training) |
| `negatives` | - | 15 | Evaluation of rejection mechanism on unrelated images |
| **Total** | - | **123** | Complete authentic dataset |

### Strict Prevention of Data Leakage
1. **Physical Independence:** Every file in `train`, `val`, `test`, and `negatives` is a physically distinct photograph. Near-duplicates, burst shots, and re-encodings of identical scenes were eliminated during catalog curation.
2. **Scaler Isolation:** The `StandardScaler` is fitted *strictly* on the `train` partition ($N=72$). Validation and test feature vectors are normalized using the previously fitted training mean and variance.
3. **No Augmented Test Samples:** Data transformations (contrast, brightness, gamma) are used solely for robustness benchmarking, not as augmented training samples.

---

## 4. Negative / Rejection Test Set
The rejection benchmark set contains 15 authentic images of non-supported subjects:
* Modern glass skyscrapers and commercial towers (e.g. Bonn Post-Tower)
* Unrelated European monuments (Eiffel Tower in Paris, Colosseum in Rome)
* Natural mountain and landscape scenery
* Indoor offices and room settings

---

## 5. Dataset Catalog
Full licensing, author attribution, and source URLs are tracked in `backend/data/dataset_catalog.json`.

To reproduce the dataset from scratch:
```bash
python backend/ml/prepare_dataset.py
```
