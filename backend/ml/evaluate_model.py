"""
MonuScript - Comprehensive Model Evaluation Script
Evaluates the trained SVM classifier on held-out test data and out-of-distribution negatives.
Measures:
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- Processing latency per image
- Rejection rates: FAR (False Acceptance), TRR (True Rejection), CAR (Correct Acceptance), FRR (False Rejection)
- Robustness under brightness, contrast, and gamma transformations
"""

import os
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

from image_processing.feature_extractor import (
    extract_features,
    extract_features_from_preprocessed,
    FEATURE_GROUPS
)
from image_processing.preprocessing import preprocess_image_for_features
from image_processing.transformations import apply_brightness_contrast, apply_gamma_transformation
from ml.rejection import RejectionJudge

DATASET_ROOT = BACKEND_DIR / "data" / "dataset"
ARTIFACTS_DIR = BACKEND_DIR / "ml" / "model_artifacts"
MODEL_PIPELINE_PATH = ARTIFACTS_DIR / "svm_pipeline.joblib"
EVAL_OUTPUT_PATH = ARTIFACTS_DIR / "evaluation_results.json"


def apply_feature_group_weights(X, feature_groups=FEATURE_GROUPS, use_weights=True):
    if not use_weights:
        return X
    X_weighted = X.copy()
    for group_name, info in feature_groups.items():
        s = info["start"]
        e = info["end"]
        w = info["default_weight"]
        scale_factor = np.sqrt(w)
        X_weighted[:, s:e] = X_weighted[:, s:e] * scale_factor
    return X_weighted


def evaluate_test_set():
    print("==================================================")
    print(" MonuScript - Model Evaluation on Held-Out Test Set ")
    print("==================================================")

    if not MODEL_PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Model pipeline not found at {MODEL_PIPELINE_PATH}. Please train the model first.")

    pipeline = joblib.load(MODEL_PIPELINE_PATH)
    scaler = pipeline["scaler"]
    clf = pipeline["classifier"]
    classes = pipeline["classes"]
    use_weights = pipeline.get("use_feature_weights", True)
    class_centroids = pipeline.get("class_centroids")
    feature_groups = pipeline.get("feature_groups", FEATURE_GROUPS)

    use_clahe = pipeline.get("use_clahe", False)
    clahe_clip_limit = pipeline.get("clahe_clip_limit", 2.0)
    clahe_grid_size = tuple(pipeline.get("clahe_grid_size", (8, 8)))
    include_wavelet = pipeline.get("include_wavelet", False)
    include_shape = pipeline.get("include_shape", False)
    include_architectural = pipeline.get("include_architectural", False)
    rejection_cfg = pipeline.get("rejection_config", {})

    judge = RejectionJudge(
        accept_threshold=rejection_cfg.get("accept_threshold", 0.40),
        margin_threshold=rejection_cfg.get("margin_threshold", 0.12),
        uncertain_threshold=rejection_cfg.get("uncertain_threshold", 0.28),
        class_centroids=class_centroids
    )

    test_dir = DATASET_ROOT / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found at {test_dir}")

    # 1. Collect test samples
    y_true = []
    y_pred = []
    y_prob = []
    accepted_flags = []
    inference_times = []

    for cls in classes:
        cls_dir = test_dir / cls
        if not cls_dir.exists():
            continue
        test_images = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
        for img_path in test_images:
            t0 = time.time()
            feats, prep = extract_features(
                str(img_path),
                use_clahe=use_clahe,
                clahe_clip_limit=clahe_clip_limit,
                clahe_grid_size=clahe_grid_size,
                include_wavelet=include_wavelet,
                include_shape=include_shape,
                include_architectural=include_architectural
            )
            scaled_feats = np.clip(scaler.transform(feats.reshape(1, -1)), -5.0, 5.0)
            proc_feats = apply_feature_group_weights(scaled_feats, feature_groups=feature_groups, use_weights=use_weights)

            probs = clf.predict_proba(proc_feats)[0]
            elapsed_ms = (time.time() - t0) * 1000.0
            inference_times.append(elapsed_ms)

            decision = judge.evaluate(probs, classes, scaled_features=proc_feats[0])
            pred_class = decision["top_class"]

            y_true.append(cls)
            y_pred.append(pred_class)
            y_prob.append(probs.tolist())
            accepted_flags.append(decision["accepted"])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accepted_flags = np.array(accepted_flags)

    # Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    # Correct Acceptance Rate (CAR) and False Rejection Rate (FRR) on supported classes
    correct_mask = (y_true == y_pred)
    car = float(np.mean(accepted_flags[correct_mask])) if np.any(correct_mask) else 0.0
    frr = 1.0 - car

    print(f"\n[Test Set Results]")
    print(f"Total Test Samples: {len(y_true)}")
    print(f"Raw Classifier Accuracy: {acc:.2%}")
    print(f"Correct Acceptance Rate (CAR): {car:.2%}")
    print(f"False Rejection Rate (FRR): {frr:.2%}")
    print(f"Average Inference Latency: {np.mean(inference_times):.1f} ms per image")

    # 2. Evaluate on Negative/Rejection Samples
    neg_dir = DATASET_ROOT / "negatives"
    neg_accepted = 0
    neg_total = 0
    neg_results = []

    if neg_dir.exists():
        neg_images = sorted([f for f in neg_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
        neg_total = len(neg_images)
        for img_path in neg_images:
            try:
                feats, _ = extract_features(
                    str(img_path),
                    use_clahe=use_clahe,
                    clahe_clip_limit=clahe_clip_limit,
                    clahe_grid_size=clahe_grid_size,
                    include_wavelet=include_wavelet,
                    include_shape=include_shape,
                    include_architectural=include_architectural
                )
                scaled_feats = np.clip(scaler.transform(feats.reshape(1, -1)), -5.0, 5.0)
                proc_feats = apply_feature_group_weights(scaled_feats, feature_groups=feature_groups, use_weights=use_weights)
                probs = clf.predict_proba(proc_feats)[0]
                decision = judge.evaluate(probs, classes, scaled_features=proc_feats[0])
                if decision["accepted"]:
                    neg_accepted += 1
                neg_results.append({
                    "filename": img_path.name,
                    "top_class": decision["top_class"],
                    "score": decision["top_score"],
                    "accepted": decision["accepted"],
                    "reason": decision["rejection_reason"]
                })
            except Exception as e:
                print(f"Error on negative sample {img_path.name}: {e}")

    far = (float(neg_accepted) / neg_total) if neg_total > 0 else 0.0
    trr = 1.0 - far
    print(f"\n[Rejection Performance on Unrelated/Negative Images]")
    print(f"Total Negative Images: {neg_total}")
    print(f"False Acceptance Rate (FAR): {far:.2%} ({neg_accepted}/{neg_total})")
    print(f"True Rejection Rate (TRR): {trr:.2%}")

    # 3. Robustness Transformations on Test Set
    print("\n[Robustness Analysis on Transformed Test Images]")
    robustness_results = {}

    transform_scenarios = [
        ("brightness_darker", lambda arr: apply_brightness_contrast(arr, alpha=1.0, beta=-25)),
        ("brightness_brighter", lambda arr: apply_brightness_contrast(arr, alpha=1.0, beta=25)),
        ("contrast_lower", lambda arr: apply_brightness_contrast(arr, alpha=0.8, beta=0)),
        ("contrast_higher", lambda arr: apply_brightness_contrast(arr, alpha=1.2, beta=0)),
        ("gamma_compressed", lambda arr: apply_gamma_transformation(arr, gamma=1.2)),
        ("gamma_expanded", lambda arr: apply_gamma_transformation(arr, gamma=0.8)),
    ]

    for scenario_name, func in transform_scenarios:
        t_correct = 0
        t_count = 0
        for cls in classes:
            cls_dir = test_dir / cls
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                    continue
                try:
                    # Preprocess and apply transformation to grayscale/rgb
                    prep = preprocess_image_for_features(
                        str(img_path),
                        use_clahe=use_clahe,
                        clahe_clip_limit=clahe_clip_limit,
                        clahe_grid_size=clahe_grid_size
                    )
                    transformed_gray = func(prep["gray"])
                    prep["gray"] = transformed_gray
                    feats = extract_features_from_preprocessed(
                        prep,
                        include_wavelet=include_wavelet,
                        include_shape=include_shape,
                        include_architectural=include_architectural
                    )

                    scaled = np.clip(scaler.transform(feats.reshape(1, -1)), -5.0, 5.0)
                    proc = apply_feature_group_weights(scaled, feature_groups=feature_groups, use_weights=use_weights)
                    pred = clf.predict(proc)[0]

                    if pred == cls:
                        t_correct += 1
                    t_count += 1
                except Exception:
                    pass
        scenario_acc = (float(t_correct) / t_count) if t_count > 0 else 0.0
        robustness_results[scenario_name] = round(scenario_acc, 4)
        print(f"  {scenario_name:22s}: Accuracy = {scenario_acc:.2%}")

    # Compile Evaluation Report
    full_report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_samples_count": len(y_true),
        "test_accuracy": round(float(acc), 4),
        "correct_acceptance_rate": round(car, 4),
        "false_rejection_rate": round(frr, 4),
        "negative_samples_count": neg_total,
        "false_acceptance_rate": round(far, 4),
        "true_rejection_rate": round(trr, 4),
        "mean_latency_ms": round(float(np.mean(inference_times)), 2),
        "confusion_matrix": {
            "classes": classes,
            "matrix": conf_mat
        },
        "per_class_metrics": {cls: report[cls] for cls in classes if cls in report},
        "robustness_accuracy": robustness_results
    }

    with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[OK] Evaluation report saved to {EVAL_OUTPUT_PATH}")

    return full_report


if __name__ == "__main__":
    evaluate_test_set()
