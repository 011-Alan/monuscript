# MonuScript: Academic Limitations & Engineering Discussion

## 1. Traditional Digital Image Processing Constraints
MonuScript was built under strict syllabus constraints to reinforce fundamental image processing mathematics (Sobel derivatives, gray-level transformations, histogram moments, and texture filters). In accordance with the academic rubric, deep neural networks (CNNs, ResNets, YOLO, Vision Transformers) and pretrained visual foundation embeddings were deliberately excluded.

Traditional handcrafted features operate on low-level intensity and edge gradients:
* While Sobel spatial filters capture local orientation (e.g. vertical minarets vs horizontal cornices), they do not learn hierarchical semantic abstractions (such as multi-scale architectural friezes, dome pedestals, or intricate jali stone tracery).
* Handcrafted features are sensitive to scale changes when the monument occupies a very small fraction of the image frame.

---

## 2. Dataset Size & Class Visual Overlaps
* **Sample Count:** The training set contains 72 authentic images (12 per supported class), split across 6 classes. While sufficient for a functioning linear/RBF SVM prototype, larger datasets (hundreds of images per class) are typically required to cover wide seasonal, weather, and nocturnal variations.
* **Material Similarity:** Several supported monuments share similar stone compositions:
  * Red Fort, Qutub Minar, and India Gate all feature red or buff sandstone, causing their HSV color histograms to overlap.
  * In these cases, the classifier relies predominantly on structural edge anisotropy (vertical minaret vs triumphal arch) rather than color.

---

## 3. Sliding-Window Multi-Candidate Detection
* Conventional sliding-window classification divides the image into overlapping sub-regions at fixed ratios (full, left, right, center, top).
* Unlike modern anchor-based object detectors, sliding window evaluates arbitrary sub-regions that may bisect a monument or capture background clutter.
* To prevent duplicate detections, MonuScript implements greedy spatial deduplication and confidence thresholding. This is an experimental prototype extension, not a full generalized real-time detector.

---

## 4. Confidence Scores vs Calibrated Probabilities
* In MonuScript, recognition scores are derived from SVM margin distance and Platt scaling via `predict_proba()`.
* On small training sets, Platt scaling probabilities can be overconfident or poorly calibrated near decision boundaries.
* The system therefore designates the metric as a **Recognition Score** rather than a guaranteed Bayesian posterior probability.
* A multi-criteria rejection judge (score threshold + runner-up margin threshold + centroid distance) is applied to mitigate false acceptances of out-of-distribution inputs.

---

## 5. Viewpoint and Partial Occlusion
* Severe viewpoint distortions (e.g. extreme low-angle shots from directly beneath a minaret) or heavy occlusions (monument covered by scaffolding, trees, or crowd heads) alter the 3x3 spatial grid statistics, which can reduce classification confidence below the acceptance threshold.
