import sys
from pathlib import Path
import numpy as np
import joblib

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from image_processing.feature_extractor import extract_features, FEATURE_GROUPS

DATASET_ROOT = BACKEND_DIR / "data" / "dataset"
pipeline = joblib.load(BACKEND_DIR / "ml" / "model_artifacts" / "svm_pipeline.joblib")
scaler = pipeline["scaler"]
clf = pipeline["classifier"]
classes = pipeline["classes"]
use_clahe = pipeline.get("use_clahe", False)

print("Classes in model:", classes)
ig_idx = classes.index("india_gate")
qm_idx = classes.index("qutub_minar")

# Inspect all India Gate images in train and test
for split in ["train", "test"]:
    ig_dir = DATASET_ROOT / split / "india_gate"
    print(f"\n=================== INDIA GATE ({split.upper()}) ===================")
    for f in sorted(ig_dir.glob("*.*")):
        feats, prep = extract_features(str(f), use_clahe=use_clahe)
        sc = scaler.transform(feats.reshape(1, -1))[0]
        probs = clf.predict_proba(sc.reshape(1, -1))[0]
        
        top_c = classes[np.argmax(probs)]
        top_p = np.max(probs)
        ig_p = probs[ig_idx]
        qm_p = probs[qm_idx]
        
        print(f"[{f.name}] Pred: {top_c:12s} (p={top_p:.2f}) | P(IndiaGate)={ig_p:.2f}, P(QutubMinar)={qm_p:.2f}")

# Also check SVM coefficients if linear
if hasattr(clf, "coef_"):
    print("\n=================== SVM COEFFICIENTS ANALYSIS ===================")
    # Binary classifiers in one-vs-rest or one-vs-one
    print("Classifier coef shape:", clf.coef_.shape)
