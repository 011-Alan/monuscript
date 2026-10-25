import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.inference import MonumentPredictor

predictor = MonumentPredictor.get_instance()
uploads = BACKEND_DIR / "uploads"

print("=== CHECKING ALL UPLOADED IMAGES ===")
for f in sorted(uploads.glob("*.*")):
    if f.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        continue
    res = predictor.predict_image(str(f))
    top_c = res.get("top_class")
    score = res.get("top_score")
    acc = res.get("accepted")
    unc = res.get("is_uncertain")
    msg = res.get("message")
    print(f"[{f.name[:16]}] Top: {top_c:12s} ({score:.2f}) | Accepted: {acc} | Uncertain: {unc}")
    if unc or not acc:
        print(f"   -> Message: {msg}")
