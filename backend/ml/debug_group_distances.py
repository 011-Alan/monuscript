import sys
from pathlib import Path
import numpy as np
import joblib

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from image_processing.feature_extractor import extract_features, FEATURE_GROUPS

pipeline = joblib.load(BACKEND_DIR / "ml" / "model_artifacts" / "svm_pipeline.joblib")
scaler = pipeline["scaler"]
clf = pipeline["classifier"]
classes = pipeline["classes"]
use_clahe = pipeline.get("use_clahe", False)

fpath = BACKEND_DIR / "data" / "dataset" / "test" / "india_gate" / "india_gate_test_01.jpg"
feats, prep = extract_features(str(fpath), use_clahe=use_clahe)
sc = scaler.transform(feats.reshape(1, -1))[0]

# Inspect feature groups distance to India Gate centroid vs Qutub Minar centroid
ig_c = np.array(pipeline["class_centroids"]["india_gate"])
qm_c = np.array(pipeline["class_centroids"]["qutub_minar"])

print("=== FEATURE GROUP DISTANCE DECOMPOSITION for india_gate_test_01.jpg ===")
for gname, ginfo in FEATURE_GROUPS.items():
    s, e = ginfo["start"], ginfo["end"]
    dist_ig = np.linalg.norm(sc[s:e] - ig_c[s:e])
    dist_qm = np.linalg.norm(sc[s:e] - qm_c[s:e])
    print(f"Group {gname:10s} [{s:3d}:{e:3d}]: dist to IndiaGate={dist_ig:6.2f} | dist to QutubMinar={dist_qm:6.2f} | diff (IG - QM)={dist_ig - dist_qm:+6.2f}")

# Also check without CLAHE
feats_noclahe, _ = extract_features(str(fpath), use_clahe=False)
print("\nIs it CLAHE causing the issue?")
# Test both with and without CLAHE across all feature groups
print("With CLAHE top 5 edge features:", feats[40:45])
print("Without CLAHE top 5 edge features:", feats_noclahe[40:45])
