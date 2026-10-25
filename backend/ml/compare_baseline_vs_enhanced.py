"""
MonuScript - Deep Image-by-Image Diagnostic: Baseline vs Enhanced Model
Compares predictions, probabilities, margins, and accept/reject decisions across:
1. Test Set (24 images)
2. Negative Rejection Set (15 images)
3. Uploads directory (all user-uploaded images)
"""

import sys
import json
from pathlib import Path
import numpy as np
import joblib

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from image_processing.feature_extractor import extract_features
from ml.rejection import RejectionJudge

DATASET_ROOT = BACKEND_DIR / "data" / "dataset"
ARTIFACTS_DIR = BACKEND_DIR / "ml" / "model_artifacts"
UPLOADS_DIR = BACKEND_DIR / "uploads"

# Train a pure baseline model (E0) on raw train data without CLAHE, C=0.5, linear
def get_baseline_model():
    train_dir = DATASET_ROOT / "train"
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    X = []
    y = []
    for cls in classes:
        for f in sorted((train_dir / cls).iterdir()):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                feats, _ = extract_features(str(f), use_clahe=False)
                X.append(feats)
                y.append(cls)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = SVC(kernel="linear", C=0.5, probability=True, random_state=42)
    clf.fit(X_scaled, y)
    judge = RejectionJudge(accept_threshold=0.40, margin_threshold=0.12, uncertain_threshold=0.28)
    return scaler, clf, judge, classes

# Load current production pipeline
def get_current_production():
    pipeline = joblib.load(ARTIFACTS_DIR / "svm_pipeline.joblib")
    scaler = pipeline["scaler"]
    clf = pipeline["classifier"]
    classes = pipeline["classes"]
    use_clahe = pipeline.get("use_clahe", False)
    rejection_cfg = pipeline.get("rejection_config", {})
    judge = RejectionJudge(
        accept_threshold=rejection_cfg.get("accept_threshold", 0.40),
        margin_threshold=rejection_cfg.get("margin_threshold", 0.12),
        uncertain_threshold=rejection_cfg.get("uncertain_threshold", 0.28)
    )
    return scaler, clf, judge, classes, use_clahe

def run_comparison():
    b_scaler, b_clf, b_judge, b_classes = get_baseline_model()
    e_scaler, e_clf, e_judge, e_classes, e_use_clahe = get_current_production()

    print("================================================================================")
    print(" 1. COMPARISON ON TEST SET (24 images)")
    print("================================================================================")
    test_dir = DATASET_ROOT / "test"
    
    flips_accepted_to_rejected = []
    flips_rejected_to_accepted = []
    flips_wrong_to_correct = []
    flips_correct_to_wrong = []

    for cls in b_classes:
        cls_dir = test_dir / cls
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            
            # Baseline inference
            b_feats, _ = extract_features(str(img_path), use_clahe=False)
            b_scaled = b_scaler.transform(b_feats.reshape(1, -1))
            b_probs = b_clf.predict_proba(b_scaled)[0]
            b_dec = b_judge.evaluate(b_probs, b_classes)

            # Enhanced inference
            e_feats, _ = extract_features(str(img_path), use_clahe=e_use_clahe)
            e_scaled = e_scaler.transform(e_feats.reshape(1, -1))
            e_probs = e_clf.predict_proba(e_scaled)[0]
            e_dec = e_judge.evaluate(e_probs, e_classes)

            b_corr = (b_dec["top_class"] == cls)
            e_corr = (e_dec["top_class"] == cls)

            status_str = f"[{img_path.name}] True: {cls}\n"
            status_str += f"   Baseline: pred={b_dec['top_class']} ({b_dec['top_score']:.2f}, marg={b_dec['score_margin']:.2f}), accepted={b_dec['accepted']} (corr={b_corr})\n"
            status_str += f"   Enhanced: pred={e_dec['top_class']} ({e_dec['top_score']:.2f}, marg={e_dec['score_margin']:.2f}), accepted={e_dec['accepted']} (corr={e_corr})"
            print(status_str)

            if b_dec["accepted"] and not e_dec["accepted"]:
                flips_accepted_to_rejected.append((img_path.name, cls, b_dec, e_dec, b_corr, e_corr))
            if not b_dec["accepted"] and e_dec["accepted"]:
                flips_rejected_to_accepted.append((img_path.name, cls, b_dec, e_dec, b_corr, e_corr))
            if not b_corr and e_corr:
                flips_wrong_to_correct.append((img_path.name, cls, b_dec, e_dec))
            if b_corr and not e_corr:
                flips_correct_to_wrong.append((img_path.name, cls, b_dec, e_dec))

    print("\n--------------------------------------------------------------------------------")
    print(" SUMMARY OF TEST SET FLIPS:")
    print(f" - Accepted in Baseline, but now REJECTED in Enhanced: {len(flips_accepted_to_rejected)}")
    for name, cls, b_d, e_d, bc, ec in flips_accepted_to_rejected:
        print(f"    * {name} ({cls}): Baseline accepted {b_d['top_class']} ({b_d['top_score']:.2f}) -> Enhanced rejected/uncertain {e_d['top_class']} ({e_d['top_score']:.2f}, marg={e_d['score_margin']:.2f}, reason={e_d['rejection_reason']})")

    print(f" - Rejected in Baseline, but now ACCEPTED in Enhanced: {len(flips_rejected_to_accepted)}")
    for name, cls, b_d, e_d, bc, ec in flips_rejected_to_accepted:
        print(f"    * {name} ({cls}): Baseline rejected {b_d['top_class']} ({b_d['top_score']:.2f}) -> Enhanced accepted {e_d['top_class']} ({e_d['top_score']:.2f})")

    print(f" - Wrong in Baseline, but now CORRECT in Enhanced: {len(flips_wrong_to_correct)}")
    for name, cls, b_d, e_d in flips_wrong_to_correct:
        print(f"    * {name} ({cls}): was {b_d['top_class']} -> now {e_d['top_class']}")

    print(f" - Correct in Baseline, but now WRONG in Enhanced: {len(flips_correct_to_wrong)}")
    for name, cls, b_d, e_d in flips_correct_to_wrong:
        print(f"    * {name} ({cls}): was {b_d['top_class']} -> now {e_d['top_class']}")

    print("\n================================================================================")
    print(" 2. COMPARISON ON NEGATIVES SET (15 images)")
    print("================================================================================")
    neg_dir = DATASET_ROOT / "negatives"
    neg_accepted_b = 0
    neg_accepted_e = 0
    if neg_dir.exists():
        for f in sorted(neg_dir.iterdir()):
            if f.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            b_feats, _ = extract_features(str(f), use_clahe=False)
            b_scaled = b_scaler.transform(b_feats.reshape(1, -1))
            b_probs = b_clf.predict_proba(b_scaled)[0]
            b_dec = b_judge.evaluate(b_probs, b_classes)

            e_feats, _ = extract_features(str(f), use_clahe=e_use_clahe)
            e_scaled = e_scaler.transform(e_feats.reshape(1, -1))
            e_probs = e_clf.predict_proba(e_scaled)[0]
            e_dec = e_judge.evaluate(e_probs, e_classes)

            if b_dec["accepted"]: neg_accepted_b += 1
            if e_dec["accepted"]: neg_accepted_e += 1

            diff = ""
            if b_dec["accepted"] != e_dec["accepted"]:
                diff = f" <--- CHANGED! (Baseline: {'ACCEPTED' if b_dec['accepted'] else 'REJECTED'} -> Enhanced: {'ACCEPTED' if e_dec['accepted'] else 'REJECTED'})"
            print(f"[{f.name}] Baseline: {b_dec['top_class']} ({b_dec['top_score']:.2f}, marg={b_dec['score_margin']:.2f}, accepted={b_dec['accepted']}) | Enhanced: {e_dec['top_class']} ({e_dec['top_score']:.2f}, marg={e_dec['score_margin']:.2f}, accepted={e_dec['accepted']}){diff}")

    print(f"\nNegatives Accepted: Baseline = {neg_accepted_b}/15 (FAR={neg_accepted_b/15:.1%}) | Enhanced = {neg_accepted_e}/15 (FAR={neg_accepted_e/15:.1%})")

    print("\n================================================================================")
    print(" 3. COMPARISON ON UPLOADED IMAGES (backend/uploads)")
    print("================================================================================")
    upload_files = [f for f in UPLOADS_DIR.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    for f in sorted(upload_files):
        b_feats, _ = extract_features(str(f), use_clahe=False)
        b_scaled = b_scaler.transform(b_feats.reshape(1, -1))
        b_probs = b_clf.predict_proba(b_scaled)[0]
        b_dec = b_judge.evaluate(b_probs, b_classes)

        e_feats, _ = extract_features(str(f), use_clahe=e_use_clahe)
        e_scaled = e_scaler.transform(e_feats.reshape(1, -1))
        e_probs = e_clf.predict_proba(e_scaled)[0]
        e_dec = e_judge.evaluate(e_probs, e_classes)

        diff = ""
        if b_dec["accepted"] != e_dec["accepted"] or b_dec["top_class"] != e_dec["top_class"]:
            diff = f" <--- FLIP/CHANGE! (Acc: {b_dec['accepted']}->{e_dec['accepted']}, Class: {b_dec['top_class']}->{e_dec['top_class']})"
        print(f"[{f.name[:16]}...] Baseline: {b_dec['top_class']} ({b_dec['top_score']:.2f}, acc={b_dec['accepted']}) | Enhanced: {e_dec['top_class']} ({e_dec['top_score']:.2f}, acc={e_dec['accepted']}){diff}")

if __name__ == "__main__":
    run_comparison()
