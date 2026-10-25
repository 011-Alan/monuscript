import sys
from pathlib import Path
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.compare_baseline_vs_enhanced import get_baseline_model, get_current_production
from image_processing.feature_extractor import extract_features

b_scaler, b_clf, b_judge, b_classes = get_baseline_model()
e_scaler, e_clf, e_judge, e_classes, e_use_clahe = get_current_production()

uploads = BACKEND_DIR / "uploads"
target_files = [
    "10f68ffa0ec04bbd99222c88c9d65e68.jpg",
    "bc4e1c066f5c4340b1e02bdba1503828.webp",
    "2c0b89cba25e4478ae53a704e53c86ad.webp",
    "9fe7dae1ef2c4148be5a37144d81b1cb.webp"
]

for fname in target_files:
    fpath = uploads / fname
    if not fpath.exists():
        continue
    print(f"=== {fname} ===")

    # Baseline
    bf, _ = extract_features(str(fpath), use_clahe=False)
    bsc = b_scaler.transform(bf.reshape(1, -1))
    bp = b_clf.predict_proba(bsc)[0]
    bdec = b_judge.evaluate(bp, b_classes)

    # Enhanced
    ef, _ = extract_features(str(fpath), use_clahe=e_use_clahe)
    esc = e_scaler.transform(ef.reshape(1, -1))
    ep = e_clf.predict_proba(esc)[0]
    edec = e_judge.evaluate(ep, e_classes)

    print(" Baseline probs:")
    for c, p in sorted(zip(b_classes, bp), key=lambda x: -x[1]):
        print(f"   {c:12s}: {p:.3f}")
    print(f" Baseline Decision: accepted={bdec['accepted']}, top={bdec['top_class']}, score={bdec['top_score']:.3f}, margin={bdec['score_margin']:.3f}")

    print(" Enhanced probs:")
    for c, p in sorted(zip(e_classes, ep), key=lambda x: -x[1]):
        print(f"   {c:12s}: {p:.3f}")
    print(f" Enhanced Decision: accepted={edec['accepted']}, top={edec['top_class']}, score={edec['top_score']:.3f}, margin={edec['score_margin']:.3f}, reason={edec['rejection_reason']}")
    print()
