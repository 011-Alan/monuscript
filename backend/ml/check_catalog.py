import json
from pathlib import Path

catalog_path = Path(__file__).resolve().parent.parent / "data" / "dataset_catalog.json"
with open(catalog_path, "r", encoding="utf-8") as f:
    cat = json.load(f)

for monument, splits in cat.get("monuments", {}).items():
    print(f"=== {monument.upper()} ===")
    for split_name in ["train", "val", "test"]:
        items = splits.get(split_name, [])
        print(f"  [{split_name.upper()}] ({len(items)} items):")
        for it in items:
            print(f"    - {it['filename']}: {it['title']}")
