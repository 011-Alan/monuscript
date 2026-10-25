import sys
from pathlib import Path
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from image_processing.feature_extractor import extract_features, FEATURE_GROUPS
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATASET_ROOT = BACKEND_DIR / "data" / "dataset"

# Load train set
classes = sorted([d.name for d in (DATASET_ROOT / "train").iterdir() if d.is_dir()])
X_tr, y_tr = [], []
for cls in classes:
    for f in sorted((DATASET_ROOT / "train" / cls).glob("*.*")):
        feats, _ = extract_features(str(f), use_clahe=True)
        X_tr.append(feats)
        y_tr.append(cls)
X_tr = np.array(X_tr, dtype=np.float32)
y_tr = np.array(y_tr)

scaler = StandardScaler()
X_tr_sc = np.clip(scaler.fit_transform(X_tr), -5.0, 5.0)

def apply_weights(X):
    Xw = X.copy()
    for g, info in FEATURE_GROUPS.items():
        s, e = info["start"], info["end"]
        Xw[:, s:e] *= np.sqrt(info["default_weight"])
    return Xw

# Compare Unweighted vs Weighted
for use_w in [False, True]:
    print(f"\n==================== WEIGHTED = {use_w} ====================")
    X_train_proc = apply_weights(X_tr_sc) if use_w else X_tr_sc
    clf = SVC(kernel="linear", C=0.5, probability=True, random_state=42)
    clf.fit(X_train_proc, y_tr)
    
    # Test on test India Gate images
    for f in sorted((DATASET_ROOT / "test" / "india_gate").glob("*.*")):
        feats, _ = extract_features(str(f), use_clahe=True)
        sc = np.clip(scaler.transform(feats.reshape(1, -1)), -5.0, 5.0)
        proc = apply_weights(sc) if use_w else sc
        probs = clf.predict_proba(proc)[0]
        top_idx = np.argmax(probs)
        top_c = classes[top_idx]
        top_p = probs[top_idx]
        ig_p = probs[classes.index("india_gate")]
        qm_p = probs[classes.index("qutub_minar")]
        print(f"[{f.name}] Pred: {top_c:12s} ({top_p:.2f}) | P(IndiaGate)={ig_p:.2f}, P(QutubMinar)={qm_p:.2f}")

    # Test on user upload a73028
    upload_ig = BACKEND_DIR / "uploads" / "a73028a07b1f4f74ba55b7066cfb8366.webp"
    if upload_ig.exists():
        feats, _ = extract_features(str(upload_ig), use_clahe=True)
        sc = np.clip(scaler.transform(feats.reshape(1, -1)), -5.0, 5.0)
        proc = apply_weights(sc) if use_w else sc
        probs = clf.predict_proba(proc)[0]
        top_idx = np.argmax(probs)
        print(f"[Upload a73028] Pred: {classes[top_idx]:12s} ({probs[top_idx]:.2f}) | P(IndiaGate)={probs[classes.index('india_gate')]:.2f}, P(QutubMinar)={probs[classes.index('qutub_minar')]:.2f}")
