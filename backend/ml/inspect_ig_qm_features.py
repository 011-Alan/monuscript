import sys
from pathlib import Path
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from image_processing.feature_extractor import extract_features
from image_processing.preprocessing import preprocess_image_for_features

DATASET_ROOT = BACKEND_DIR / "data" / "dataset"

def get_avg_feats(cls):
    feats_list = []
    for f in (DATASET_ROOT / "train" / cls).glob("*.*"):
        feats, _ = extract_features(str(f), use_clahe=True)
        feats_list.append(feats)
    return np.mean(feats_list, axis=0)

ig_avg = get_avg_feats("india_gate")
qm_avg = get_avg_feats("qutub_minar")

f_test01 = DATASET_ROOT / "test" / "india_gate" / "india_gate_test_01.jpg"
test_feats, prep = extract_features(str(f_test01), use_clahe=True)

print("=== COLOR CHANNELS (HSV) [107:139] ===")
print("Hue diff to IndiaGate:", np.linalg.norm(test_feats[107:123] - ig_avg[107:123]))
print("Hue diff to QutubMinar:", np.linalg.norm(test_feats[107:123] - qm_avg[107:123]))
print("Saturation diff to IndiaGate:", np.linalg.norm(test_feats[123:131] - ig_avg[123:131]))
print("Saturation diff to QutubMinar:", np.linalg.norm(test_feats[123:131] - qm_avg[123:131]))

print("\n=== EDGE ORIENTATION [56:72] ===")
print("Orientation diff to IndiaGate:", np.linalg.norm(test_feats[56:72] - ig_avg[56:72]))
print("Orientation diff to QutubMinar:", np.linalg.norm(test_feats[56:72] - qm_avg[56:72]))

print("\n=== ANISOTROPY & DENSITY [72:76] ===")
print("Test image [density, mean_mag, std_mag, anisotropy]:", test_feats[72:76])
print("IndiaGate avg [density, mean_mag, std_mag, anisotropy]:", ig_avg[72:76])
print("QutubMinar avg [density, mean_mag, std_mag, anisotropy]:", qm_avg[72:76])

# Let's inspect image size and aspect ratio
from PIL import Image
im = Image.open(f_test01)
print(f"\nImage size of india_gate_test_01.jpg: {im.size} (aspect ratio={im.size[0]/im.size[1]:.2f})")
