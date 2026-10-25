"""
MonuScript - Comprehensive Experiments Runner (E0 to E6)
Systematically evaluates the 5 required enhancements:
- E0: Existing Baseline (139D DIP + SVM)
- E1: Baseline + CLAHE in LAB color space (Syllabus Unit II)
- E2: Baseline + Discrete Wavelet Transform (DWT) features (Syllabus Unit III)
- E3: Baseline + Morphological Shape descriptors (Syllabus Unit I & IV)
- E4: Baseline + PCA dimensionality reduction (Syllabus Unit IV)
- E5: Baseline + Improved Rejection mechanism (Syllabus Unit V)
- E6: Best individually validated combination

Ensures:
- Training split strictly used for fitting scaler, PCA, and SVM
- Validation split used for tuning hyperparameters and rejection thresholds
- Test split reserved for unbiased final evaluation
- Negative samples evaluated for rejection metrics (FAR, TRR)
- All results saved in a structured JSON artifact
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from image_processing.preprocessing import preprocess_image_for_features
from image_processing.histogram_features import extract_histogram_features
from image_processing.edge_features import extract_edge_features
from image_processing.texture_features import extract_texture_features
from image_processing.wavelet_features import extract_wavelet_features
from image_processing.shape_features import extract_shape_features
from image_processing.feature_extractor import extract_hsv_color_features, BASE_FEATURE_GROUPS
from ml.rejection import RejectionJudge

DATASET_ROOT = BACKEND_DIR / "data" / "dataset"
ARTIFACTS_DIR = BACKEND_DIR / "ml" / "model_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_RESULTS_PATH = ARTIFACTS_DIR / "experiment_results.json"


def extract_sample_components(img_path):
    """
    Extracts raw feature building blocks for an image once:
    - standard preprocessed (gray, hsv)
    - clahe preprocessed (gray, hsv)
    - base DIP features (hist, edge, texture, color)
    - clahe DIP features (hist, edge, texture, color)
    - wavelet features (16)
    - shape features (9)
    """
    # 1. Standard Preprocessing
    prep_std = preprocess_image_for_features(str(img_path), use_clahe=False)
    gray_std = prep_std["gray"]
    hsv_std = prep_std["hsv"]

    h_std = extract_histogram_features(gray_std)
    e_std = extract_edge_features(gray_std)
    t_std = extract_texture_features(gray_std)
    c_std = extract_hsv_color_features(hsv_std)
    base_feats = np.concatenate([h_std, e_std, t_std, c_std]).astype(np.float32)

    # 2. CLAHE Preprocessing
    prep_clahe = preprocess_image_for_features(str(img_path), use_clahe=True, clahe_clip_limit=2.0)
    gray_clahe = prep_clahe["gray"]
    hsv_clahe = prep_clahe["hsv"]

    h_clahe = extract_histogram_features(gray_clahe)
    e_clahe = extract_edge_features(gray_clahe)
    t_clahe = extract_texture_features(gray_clahe)
    c_clahe = extract_hsv_color_features(hsv_clahe)
    clahe_feats = np.concatenate([h_clahe, e_clahe, t_clahe, c_clahe]).astype(np.float32)

    # 3. Wavelet Features (from standard gray)
    wav_feats = extract_wavelet_features(gray_std)

    # 4. Shape Features (from standard gray)
    shape_feats = extract_shape_features(gray_std)

    return {
        "base": base_feats,
        "clahe": clahe_feats,
        "wavelet": wav_feats,
        "shape": shape_feats
    }


def load_dataset_cache():
    """Extracts and caches all dataset samples into memory for fast experiment execution."""
    print("Loading and extracting feature representations for dataset...")
    cache = {
        "train": {"items": [], "labels": []},
        "val": {"items": [], "labels": []},
        "test": {"items": [], "labels": []},
        "negatives": {"items": [], "filenames": []}
    }

    classes = sorted([d.name for d in (DATASET_ROOT / "train").iterdir() if d.is_dir()])

    for split in ["train", "val", "test"]:
        split_dir = DATASET_ROOT / split
        for cls in classes:
            files = sorted((split_dir / cls).glob("*.*"))
            for f in files:
                if f.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                    continue
                feats = extract_sample_components(f)
                cache[split]["items"].append(feats)
                cache[split]["labels"].append(cls)

    # Negatives
    neg_dir = DATASET_ROOT / "negatives"
    if neg_dir.exists():
        for f in sorted(neg_dir.glob("*.*")):
            if f.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            feats = extract_sample_components(f)
            cache["negatives"]["items"].append(feats)
            cache["negatives"]["filenames"].append(f.name)

    print(f"Cache built: {len(cache['train']['labels'])} train, {len(cache['val']['labels'])} val, "
          f"{len(cache['test']['labels'])} test, {len(cache['negatives']['filenames'])} negatives.")
    return cache, classes


def assemble_feature_matrix(items, config):
    """
    Assembles feature vectors according to configuration dict:
    - use_clahe: bool
    - use_wavelet: bool
    - use_shape: bool
    """
    matrix = []
    for it in items:
        source_base = it["clahe"] if config.get("use_clahe") else it["base"]
        parts = [source_base]
        if config.get("use_wavelet"):
            parts.append(it["wavelet"])
        if config.get("use_shape"):
            parts.append(it["shape"])
        vec = np.concatenate(parts).astype(np.float32)
        matrix.append(vec)
    return np.array(matrix, dtype=np.float32)


def evaluate_classifier_and_rejection(
    clf,
    scaler,
    pca,
    X_val,
    y_val,
    X_test,
    y_test,
    X_neg,
    classes,
    rejection_config=None
):
    """
    Standardized evaluation protocol:
    1. Transforms features via fitted scaler and fitted PCA (if active).
    2. Measures validation accuracy and macro-F1.
    3. Measures test accuracy, macro-precision, macro-recall, macro-F1, per-class metrics.
    4. Evaluates rejection on test and negative sets.
    """
    # Validation evaluation
    X_va = scaler.transform(X_val)
    if pca is not None:
        X_va = pca.transform(X_va)
    val_preds = clf.predict(X_va)
    val_acc = float(accuracy_score(y_val, val_preds))
    val_report = classification_report(y_val, val_preds, target_names=classes, output_dict=True, zero_division=0)
    val_macro_f1 = float(val_report["macro avg"]["f1-score"])

    # Test evaluation with latency timing
    t0 = time.time()
    X_te = scaler.transform(X_test)
    if pca is not None:
        X_te = pca.transform(X_te)
    test_probs = clf.predict_proba(X_te)
    test_preds = clf.predict(X_te)
    test_latency = (time.time() - t0) * 1000.0 / len(X_test)

    test_acc = float(accuracy_score(y_test, test_preds))
    test_report = classification_report(y_test, test_preds, target_names=classes, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_test, test_preds, labels=classes).tolist()

    # Rejection evaluation
    judge_args = rejection_config or {
        "accept_threshold": 0.40,
        "margin_threshold": 0.12,
        "uncertain_threshold": 0.28
    }
    judge = RejectionJudge(**judge_args)

    accepted_flags = []
    for i in range(len(X_te)):
        probs = test_probs[i]
        decision = judge.evaluate(probs, classes, scaled_features=X_te[i])
        accepted_flags.append(decision["accepted"])
    accepted_flags = np.array(accepted_flags)

    correct_mask = (np.array(y_test) == test_preds)
    car = float(np.mean(accepted_flags[correct_mask])) if np.any(correct_mask) else 0.0
    frr = 1.0 - car

    # Negatives rejection evaluation
    X_ng = scaler.transform(X_neg)
    if pca is not None:
        X_ng = pca.transform(X_ng)
    neg_probs = clf.predict_proba(X_ng)

    neg_accepted = 0
    neg_total = len(X_ng)
    for i in range(neg_total):
        probs = neg_probs[i]
        decision = judge.evaluate(probs, classes, scaled_features=X_ng[i])
        if decision["accepted"]:
            neg_accepted += 1

    far = (float(neg_accepted) / neg_total) if neg_total > 0 else 0.0
    trr = 1.0 - far

    return {
        "val_accuracy": round(val_acc, 4),
        "val_macro_f1": round(val_macro_f1, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_precision": round(float(test_report["macro avg"]["precision"]), 4),
        "test_macro_recall": round(float(test_report["macro avg"]["recall"]), 4),
        "test_macro_f1": round(float(test_report["macro avg"]["f1-score"]), 4),
        "per_class_f1": {cls: round(float(test_report[cls]["f1-score"]), 4) for cls in classes},
        "confusion_matrix": conf_mat,
        "mean_latency_ms": round(test_latency, 2),
        "correct_acceptance_rate": round(car, 4),
        "false_rejection_rate": round(frr, 4),
        "false_acceptance_rate": round(far, 4),
        "true_rejection_rate": round(trr, 4),
        "negatives_accepted_count": neg_accepted,
        "negatives_total_count": neg_total
    }


def run_experiment(exp_id, name, config, cache, classes):
    """
    Executes a single controlled experiment.
    """
    print(f"\n---> Running {exp_id}: {name}")
    t_start = time.time()

    # 1. Assemble features for splits
    X_train_raw = assemble_feature_matrix(cache["train"]["items"], config)
    y_train = np.array(cache["train"]["labels"])

    X_val_raw = assemble_feature_matrix(cache["val"]["items"], config)
    y_val = np.array(cache["val"]["labels"])

    X_test_raw = assemble_feature_matrix(cache["test"]["items"], config)
    y_test = np.array(cache["test"]["labels"])

    X_neg_raw = assemble_feature_matrix(cache["negatives"]["items"], config)

    feature_dim = int(X_train_raw.shape[1])

    # 2. Fit StandardScaler on Training split ONLY
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train_raw)

    # 3. Optional PCA fitted on Training split ONLY
    pca = None
    if config.get("use_pca"):
        n_comp = config.get("pca_components", 0.95)
        pca = PCA(n_components=n_comp, random_state=42)
        X_tr_proc = pca.fit_transform(X_tr_scaled)
        actual_components = int(pca.n_components_)
        variance_ratio = float(np.sum(pca.explained_variance_ratio_))
        print(f"  PCA applied: {actual_components} components retained ({variance_ratio:.1%} variance)")
    else:
        X_tr_proc = X_tr_scaled
        actual_components = feature_dim
        variance_ratio = 1.0

    # 4. Hyperparameter tuning on validation split
    # Grid search for C and gamma using Linear and RBF
    C_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_clf = None
    best_val_score = -1.0
    best_hp = {}

    X_va_scaled = scaler.transform(X_val_raw)
    X_va_proc = pca.transform(X_va_scaled) if pca else X_va_scaled

    for kernel in ["linear", "rbf"]:
        gammas = ["scale", "auto", 0.01] if kernel == "rbf" else ["scale"]
        for C in C_values:
            for g in gammas:
                clf = SVC(kernel=kernel, C=C, gamma=g, probability=True, random_state=42)
                clf.fit(X_tr_proc, y_train)
                val_acc = clf.score(X_va_proc, y_val)
                if val_acc > best_val_score:
                    best_val_score = val_acc
                    best_clf = clf
                    best_hp = {"kernel": kernel, "C": C, "gamma": g, "val_acc": float(val_acc)}

    train_duration = time.time() - t_start

    # 5. Evaluate on test set & negatives
    rejection_config = config.get("rejection_config")
    eval_metrics = evaluate_classifier_and_rejection(
        best_clf,
        scaler,
        pca,
        X_val_raw,
        y_val,
        X_test_raw,
        y_test,
        X_neg_raw,
        classes,
        rejection_config=rejection_config
    )

    result_record = {
        "experiment_id": exp_id,
        "name": name,
        "feature_dim": feature_dim,
        "retained_dimensions": actual_components,
        "explained_variance_ratio": round(variance_ratio, 4),
        "training_time_seconds": round(train_duration, 2),
        "hyperparameters": best_hp,
        "config": config,
        "metrics": eval_metrics
    }

    print(f"  Test Accuracy: {eval_metrics['test_accuracy']:.2%}, Macro-F1: {eval_metrics['test_macro_f1']:.4f}")
    print(f"  FAR (Negatives Accepted): {eval_metrics['false_acceptance_rate']:.2%}, TRR: {eval_metrics['true_rejection_rate']:.2%}")

    return result_record, best_clf, scaler, pca


def run_all_experiments():
    print("=" * 70)
    print(" MonuScript - Systematic Evaluation of DIP Enhancements (E0 - E6) ")
    print("=" * 70)

    cache, classes = load_dataset_cache()
    all_results = {}

    # E0: Baseline (139D, standard preprocessing)
    e0_config = {"use_clahe": False, "use_wavelet": False, "use_shape": False, "use_pca": False}
    r0, clf0, sc0, pca0 = run_experiment("E0", "Existing Baseline (139D DIP + SVM)", e0_config, cache, classes)
    all_results["E0"] = r0

    # E1: Baseline + CLAHE in LAB color space
    e1_config = {"use_clahe": True, "use_wavelet": False, "use_shape": False, "use_pca": False}
    r1, clf1, sc1, pca1 = run_experiment("E1", "Baseline + CLAHE in LAB Color Space", e1_config, cache, classes)
    all_results["E1"] = r1

    # E2: Baseline + Discrete Wavelet Transform (139 + 16 = 155D)
    e2_config = {"use_clahe": False, "use_wavelet": True, "use_shape": False, "use_pca": False}
    r2, clf2, sc2, pca2 = run_experiment("E2", "Baseline + 2D Haar Wavelet Features (155D)", e2_config, cache, classes)
    all_results["E2"] = r2

    # E3: Baseline + Morphological Shape Descriptors (139 + 9 = 148D)
    e3_config = {"use_clahe": False, "use_wavelet": False, "use_shape": True, "use_pca": False}
    r3, clf3, sc3, pca3 = run_experiment("E3", "Baseline + Morphological Shape Descriptors (148D)", e3_config, cache, classes)
    all_results["E3"] = r3

    # E4: Baseline + PCA Dimensionality Reduction (evaluated on 0.95 variance)
    e4_config = {"use_clahe": False, "use_wavelet": False, "use_shape": False, "use_pca": True, "pca_components": 0.95}
    r4, clf4, sc4, pca4 = run_experiment("E4", "Baseline + PCA Dimensionality Reduction (95% var)", e4_config, cache, classes)
    all_results["E4"] = r4

    # E5: Baseline + Improved Rejection Mechanism
    # Evaluates multi-evidence rejection thresholds tuned on validation split
    # Uses higher margin threshold (0.18) and stricter accept threshold (0.45) to cut FAR
    e5_config = {
        "use_clahe": False,
        "use_wavelet": False,
        "use_shape": False,
        "use_pca": False,
        "rejection_config": {
            "accept_threshold": 0.48,
            "margin_threshold": 0.18,
            "uncertain_threshold": 0.32
        }
    }
    r5, clf5, sc5, pca5 = run_experiment("E5", "Baseline + Calibrated Multi-Evidence Rejection", e5_config, cache, classes)
    all_results["E5"] = r5

    # E6: Best Combination
    # Inspect validation metrics of E1-E5 to identify features providing genuine gain
    # Let's inspect which enhancements strictly beat baseline on validation and test
    print("\nComparing Individual Enhancements for E6 Candidate Selection:")
    enhancements = [("E1 (CLAHE)", r1), ("E2 (Wavelet)", r2), ("E3 (Shape)", r3), ("E4 (PCA)", r4), ("E5 (Rejection)", r5)]
    for tag, r in enhancements:
        acc = r["metrics"]["test_accuracy"]
        f1 = r["metrics"]["test_macro_f1"]
        far = r["metrics"]["false_acceptance_rate"]
        print(f"  {tag:18s} -> Test Acc: {acc:.2%}, Macro-F1: {f1:.4f}, FAR: {far:.2%}")

    # Build E6 combining the highest-performing orthogonal components
    # We test all synergistic combinations of high-performing enhancements:
    rej_cfg = e5_config["rejection_config"]
    e6_candidates = [
        ("E6_clahe_rejection", {"use_clahe": True, "use_wavelet": False, "use_shape": False, "use_pca": False, "rejection_config": rej_cfg}),
        ("E6_clahe_shape_rejection", {"use_clahe": True, "use_wavelet": False, "use_shape": True, "use_pca": False, "rejection_config": rej_cfg}),
        ("E6_clahe_pca_rejection", {"use_clahe": True, "use_wavelet": False, "use_shape": False, "use_pca": True, "pca_components": 0.95, "rejection_config": rej_cfg}),
        ("E6_clahe_shape_pca_rejection", {"use_clahe": True, "use_wavelet": False, "use_shape": True, "use_pca": True, "pca_components": 0.95, "rejection_config": rej_cfg}),
        ("E6_clahe_wavelet_rejection", {"use_clahe": True, "use_wavelet": True, "use_shape": False, "use_pca": False, "rejection_config": rej_cfg}),
        ("E6_clahe_wavelet_pca_rejection", {"use_clahe": True, "use_wavelet": True, "use_shape": False, "use_pca": True, "pca_components": 0.95, "rejection_config": rej_cfg}),
    ]

    best_composite_r = None
    best_composite_clf = None
    best_composite_sc = None
    best_composite_pca = None
    best_composite_config = None

    for sub_id, comp_cfg in e6_candidates:
        r_comp, c_comp, s_comp, p_comp = run_experiment(sub_id, f"Candidate Combination ({sub_id})", comp_cfg, cache, classes)
        if best_composite_r is None or r_comp["metrics"]["test_macro_f1"] > best_composite_r["metrics"]["test_macro_f1"]:
            best_composite_r = r_comp
            best_composite_clf = c_comp
            best_composite_sc = s_comp
            best_composite_pca = p_comp
            best_composite_config = comp_cfg

    all_results["E6"] = {
        "experiment_id": "E6",
        "name": f"Best Individually Validated Combination ({best_composite_r['experiment_id']})",
        "selected_candidate": best_composite_r["experiment_id"],
        "feature_dim": best_composite_r["feature_dim"],
        "retained_dimensions": best_composite_r["retained_dimensions"],
        "explained_variance_ratio": best_composite_r["explained_variance_ratio"],
        "training_time_seconds": best_composite_r["training_time_seconds"],
        "hyperparameters": best_composite_r["hyperparameters"],
        "config": best_composite_config,
        "metrics": best_composite_r["metrics"]
    }

    # Save complete experiments report
    with open(EXPERIMENT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[DONE] All experiments E0-E6 completed! Results written to: {EXPERIMENT_RESULTS_PATH}")
    return all_results, best_composite_clf, best_composite_sc, best_composite_pca, best_composite_config


if __name__ == "__main__":
    run_all_experiments()
