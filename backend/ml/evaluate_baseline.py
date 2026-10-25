"""
MonuScript - Baseline Evaluation Script (E0)
Evaluates the existing DIP + SVM model and saves baseline_results.json.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from image_processing.feature_extractor import extract_features, FEATURE_GROUPS
from ml.rejection import RejectionJudge

ARTIFACTS_DIR = BACKEND_DIR / "ml" / "model_artifacts"
DATASET_ROOT = BACKEND_DIR / "data" / "dataset"
PIPELINE_PATH = ARTIFACTS_DIR / "svm_pipeline.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
BASELINE_OUT_PATH = ARTIFACTS_DIR / "baseline_results.json"


def apply_feature_group_weights(X, use_weights=True):
    if not use_weights:
        return X
    X_weighted = X.copy()
    for group_name, info in FEATURE_GROUPS.items():
        s = info["start"]
        e = info["end"]
        w = info["default_weight"]
        scale_factor = np.sqrt(w)
        X_weighted[:, s:e] = X_weighted[:, s:e] * scale_factor
    return X_weighted


def run_baseline_evaluation():
    print("=" * 60)
    print(" MonuScript - Recording Baseline (E0) Evaluation Results")
    print("=" * 60)

    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Model pipeline not found at {PIPELINE_PATH}")

    pipeline = joblib.load(PIPELINE_PATH)
    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    scaler = pipeline["scaler"]
    clf = pipeline["classifier"]
    classes = pipeline["classes"]
    use_weights = pipeline.get("use_feature_weights", False)
    centroids = pipeline.get("class_centroids")
    judge = RejectionJudge(class_centroids=centroids)

    test_dir = DATASET_ROOT / "test"
    y_true = []
    y_pred = []
    y_prob = []
    accepted_flags = []
    latencies = []

    for cls in classes:
        cls_dir = test_dir / cls
        for img_path in sorted(cls_dir.glob("*.*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            t0 = time.time()
            feats, _ = extract_features(str(img_path))
            scaled = scaler.transform(feats.reshape(1, -1))
            proc = apply_feature_group_weights(scaled, use_weights=use_weights)
            probs = clf.predict_proba(proc)[0]
            latencies.append((time.time() - t0) * 1000.0)

            decision = judge.evaluate(probs, classes, scaled_features=proc[0])
            y_true.append(cls)
            y_pred.append(decision["top_class"])
            y_prob.append(probs.tolist())
            accepted_flags.append(decision["accepted"])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accepted_flags = np.array(accepted_flags)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    correct_mask = (y_true == y_pred)
    car = float(np.mean(accepted_flags[correct_mask])) if np.any(correct_mask) else 0.0
    frr = 1.0 - car

    # Negatives evaluation
    neg_dir = DATASET_ROOT / "negatives"
    neg_total = 0
    neg_accepted = 0
    neg_details = []
    if neg_dir.exists():
        for f in sorted(neg_dir.glob("*.*")):
            if f.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            neg_total += 1
            feats, _ = extract_features(str(f))
            scaled = scaler.transform(feats.reshape(1, -1))
            proc = apply_feature_group_weights(scaled, use_weights=use_weights)
            probs = clf.predict_proba(proc)[0]
            decision = judge.evaluate(probs, classes, scaled_features=proc[0])
            if decision["accepted"]:
                neg_accepted += 1
            neg_details.append({
                "filename": f.name,
                "top_class": decision["top_class"],
                "score": round(float(decision["top_score"]), 4),
                "accepted": decision["accepted"],
                "reason": decision["rejection_reason"]
            })

    far = (float(neg_accepted) / neg_total) if neg_total > 0 else 0.0
    trr = 1.0 - far

    baseline_data = {
        "experiment_id": "E0",
        "name": "Existing Baseline (DIP 139D + Linear SVM)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_dimension": 139,
        "training_time_seconds": metadata.get("training_duration_seconds", 9.67),
        "mean_inference_latency_ms": round(float(np.mean(latencies)), 2),
        "overall_test_accuracy": round(float(acc), 4),
        "macro_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "per_class_metrics": {
            cls: {
                "precision": round(float(report[cls]["precision"]), 4),
                "recall": round(float(report[cls]["recall"]), 4),
                "f1-score": round(float(report[cls]["f1-score"]), 4),
                "support": int(report[cls]["support"])
            }
            for cls in classes
        },
        "confusion_matrix": {
            "classes": classes,
            "matrix": conf_mat
        },
        "rejection_metrics": {
            "test_samples_count": len(y_true),
            "correct_acceptance_rate": round(car, 4),
            "false_rejection_rate": round(frr, 4),
            "negative_samples_count": neg_total,
            "false_acceptance_rate": round(far, 4),
            "true_rejection_rate": round(trr, 4),
            "negative_accepted_count": neg_accepted
        },
        "hyperparameters": metadata.get("hyperparameters", {})
    }

    BASELINE_OUT_PATH.write_text(json.dumps(baseline_data, indent=2), encoding="utf-8")
    print(f"[OK] Baseline results saved to: {BASELINE_OUT_PATH}")
    print(f"  Test Accuracy: {acc:.2%}")
    print(f"  Macro-F1: {report['macro avg']['f1-score']:.4f}")
    print(f"  FAR (Negatives accepted): {far:.2%} ({neg_accepted}/{neg_total})")
    print(f"  TRR (Negatives rejected): {trr:.2%}")
    print(f"  Mean Latency: {np.mean(latencies):.2f} ms")

    return baseline_data


if __name__ == "__main__":
    run_baseline_evaluation()
