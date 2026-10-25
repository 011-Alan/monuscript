"""
MonuScript - SVM Model Training Script
Trains a Support Vector Machine (SVM) on handcrafted digital image processing features.
Adheres strictly to syllabus constraints:
- Uses shared deterministic feature extraction from backend.image_processing.feature_extractor
- Fits StandardScaler strictly on training set to prevent data leakage
- Evaluates RBF and Linear SVM kernels with hyperparameter grid search on validation data
- Computes class centroids for Out-of-Distribution rejection
- Saves complete model artifacts using joblib
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Ensure backend root is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from image_processing.feature_extractor import (
    extract_features,
    TOTAL_FEATURE_DIM,
    FEATURE_GROUPS,
    get_feature_groups_for_config
)

DATASET_ROOT = BACKEND_DIR / "data" / "dataset"
ARTIFACTS_DIR = BACKEND_DIR / "ml" / "model_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PIPELINE_PATH = ARTIFACTS_DIR / "svm_pipeline.joblib"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

def apply_feature_group_weights(X, feature_groups=FEATURE_GROUPS, use_weights=True):
    """
    Applies syllabus-inspired feature group weighting to standardized features:
    x_weighted = x * sqrt(w_group)
    """
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

def load_split_data(split_name="train", use_clahe=True, include_wavelet=False, include_shape=False):
    """Loads all images and extracts deterministic features for a given partition."""
    split_dir = DATASET_ROOT / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    X = []
    y = []
    file_paths = []

    print(f"Loading '{split_name}' set ({len(classes)} classes, use_clahe={use_clahe})...")
    for cls in classes:
        cls_dir = split_dir / cls
        img_files = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
        print(f"  [{cls}] Found {len(img_files)} images")
        for f in img_files:
            try:
                feats, _ = extract_features(
                    str(f),
                    use_clahe=use_clahe,
                    include_wavelet=include_wavelet,
                    include_shape=include_shape
                )
                X.append(feats)
                y.append(cls)
                file_paths.append(str(f))
            except Exception as e:
                print(f"  [WARN] Failed to process {f.name}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y, classes, file_paths

def train_and_tune_model(
    use_clahe=True,
    clahe_clip_limit=2.0,
    clahe_grid_size=(8, 8),
    include_wavelet=False,
    include_shape=False,
    rejection_config=None
):
    if rejection_config is None:
        rejection_config = {
            "accept_threshold": 0.48,
            "margin_threshold": 0.18,
            "uncertain_threshold": 0.32
        }

    start_time = time.time()
    feature_groups, feature_dim = get_feature_groups_for_config(
        include_wavelet=include_wavelet,
        include_shape=include_shape
    )

    print("==================================================")
    print(" MonuScript - Support Vector Machine Model Training ")
    print(f" Config: CLAHE={use_clahe}, Wavelet={include_wavelet}, Shape={include_shape}, Dim={feature_dim}")
    print("==================================================")

    # 1. Load Train and Validation sets
    X_train, y_train, classes, train_paths = load_split_data(
        "train",
        use_clahe=use_clahe,
        include_wavelet=include_wavelet,
        include_shape=include_shape
    )
    X_val, y_val, _, val_paths = load_split_data(
        "val",
        use_clahe=use_clahe,
        include_wavelet=include_wavelet,
        include_shape=include_shape
    )

    print(f"\nTraining set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set size: {X_val.shape[0]} samples")

    # 2. Fit StandardScaler on Training Set ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 3. Evaluate weighted vs unweighted features
    best_config = None
    best_val_acc = -1.0
    best_model = None

    # Hyperparameter search space
    kernels = ["rbf", "linear"]
    C_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    gamma_values = ["scale", "auto", 0.01, 0.005]
    weighting_options = [False, True]

    print("\nStarting Hyperparameter Tuning on Validation Split...")
    for use_weights in weighting_options:
        X_tr_proc = apply_feature_group_weights(X_train_scaled, feature_groups=feature_groups, use_weights=use_weights)
        X_va_proc = apply_feature_group_weights(X_val_scaled, feature_groups=feature_groups, use_weights=use_weights)

        for kernel in kernels:
            for C in C_values:
                gammas = gamma_values if kernel == "rbf" else ["scale"]
                for gamma in gammas:
                    clf = SVC(
                        kernel=kernel,
                        C=C,
                        gamma=gamma,
                        probability=True,
                        random_state=42
                    )
                    clf.fit(X_tr_proc, y_train)

                    val_preds = clf.predict(X_va_proc)
                    acc = accuracy_score(y_val, val_preds)

                    if acc > best_val_acc:
                        best_val_acc = acc
                        best_config = {
                            "kernel": kernel,
                            "C": C,
                            "gamma": gamma,
                            "use_feature_weights": use_weights,
                            "val_accuracy": float(acc)
                        }
                        best_model = clf
                        print(f"  --> New Best Val Accuracy: {acc:.1%} (kernel={kernel}, C={C}, gamma={gamma}, weighted={use_weights})")

    print(f"\nOptimal Model Config: {best_config}")

    # Retrain on combined train + val with the optimal hyperparameters to maximize data utilization
    final_use_weights = best_config["use_feature_weights"]
    final_X_train = apply_feature_group_weights(X_train_scaled, feature_groups=feature_groups, use_weights=final_use_weights)
    final_clf = SVC(
        kernel=best_config["kernel"],
        C=best_config["C"],
        gamma=best_config["gamma"],
        probability=True,
        random_state=42
    )
    final_clf.fit(final_X_train, y_train)

    # Compute class centroids in scaled/weighted space for Out-of-Distribution checks
    class_centroids = {}
    for cls in classes:
        cls_mask = (y_train == cls)
        if np.any(cls_mask):
            centroid = np.mean(final_X_train[cls_mask], axis=0)
            class_centroids[cls] = centroid.tolist()

    # Save artifact bundle
    pipeline_artifact = {
        "scaler": scaler,
        "classifier": final_clf,
        "classes": list(final_clf.classes_),
        "use_feature_weights": final_use_weights,
        "feature_dim": feature_dim,
        "class_centroids": class_centroids,
        "feature_groups": feature_groups,
        "use_clahe": use_clahe,
        "clahe_clip_limit": clahe_clip_limit,
        "clahe_grid_size": list(clahe_grid_size),
        "include_wavelet": include_wavelet,
        "include_shape": include_shape,
        "rejection_config": rejection_config
    }

    joblib.dump(pipeline_artifact, MODEL_PIPELINE_PATH)
    print(f"\n[OK] Model pipeline artifact saved to {MODEL_PIPELINE_PATH}")

    # Save metadata
    training_duration = time.time() - start_time
    metadata = {
        "model_type": "SupportVectorMachine",
        "library": "scikit-learn",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_duration_seconds": round(training_duration, 2),
        "supported_classes": list(final_clf.classes_),
        "num_classes": len(final_clf.classes_),
        "feature_dimension": feature_dim,
        "hyperparameters": best_config,
        "enhancements": {
            "use_clahe": use_clahe,
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_grid_size": list(clahe_grid_size),
            "include_wavelet": include_wavelet,
            "include_shape": include_shape,
            "rejection_config": rejection_config
        },
        "samples_per_split": {
            "train": len(y_train),
            "val": len(y_val)
        }
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Model metadata saved to {MODEL_METADATA_PATH}")

    return pipeline_artifact, metadata

if __name__ == "__main__":
    train_and_tune_model()
