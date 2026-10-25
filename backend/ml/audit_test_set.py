import os
from pathlib import Path
from PIL import Image

test_dir = Path(__file__).resolve().parent.parent / "data" / "dataset" / "test"

print("=================================================================")
print(" AUDITING TEST SET IMAGES")
print("=================================================================")

for cls_dir in sorted(test_dir.iterdir()):
    if not cls_dir.is_dir(): continue
    cls_name = cls_dir.name
    print(f"\n--- {cls_name.upper()} ---")
    for img_path in sorted(cls_dir.iterdir()):
        try:
            im = Image.open(img_path)
            # Basic stats
            stat = f"size={im.size}, mode={im.mode}"
            print(f"  {img_path.name}: {stat}")
        except Exception as e:
            print(f"  {img_path.name}: ERROR {e}")
