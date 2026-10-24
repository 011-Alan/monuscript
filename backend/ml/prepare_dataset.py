"""
MonuScript Dataset Preparation Script
Fetches authentic, Creative Commons / Public Domain photographs of supported Indian monuments
and negative/rejection test cases from Wikimedia Commons.
Partitions them cleanly into train, val, and test sets without data leakage.
"""

import os
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from PIL import Image

USER_AGENT = "MonuScriptAcademic/1.0 (academic research project; student@university.edu)"
DATASET_ROOT = Path(__file__).resolve().parent.parent / "data" / "dataset"
CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset_catalog.json"

MONUMENT_QUERIES = {
    "taj_mahal": [
        "Taj Mahal Agra India",
        "Taj Mahal front view",
        "Taj Mahal dome minaret",
        "Taj Mahal marble architecture",
    ],
    "qutub_minar": [
        "Qutub Minar Delhi",
        "Qutb Minar tower Mehrauli",
        "Qutb Minar architecture balcony",
    ],
    "red_fort": [
        "Red Fort Delhi Lahori Gate",
        "Lal Qila Delhi sandstone",
        "Red Fort ramparts Delhi",
    ],
    "india_gate": [
        "India Gate New Delhi arch",
        "India Gate memorial Delhi",
        "India Gate Rajpath evening",
    ],
    "charminar": [
        "Charminar Hyderabad monument",
        "Charminar four minarets Telangana",
        "Charminar Hyderabad arch",
    ],
    "hawa_mahal": [
        "Hawa Mahal Jaipur facade",
        "Palace of Winds Jaipur windows",
        "Hawa Mahal pink sandstone",
    ],
}

NEGATIVE_QUERIES = [
    "Modern glass skyscraper facade",
    "Eiffel Tower Paris France",
    "Colosseum Rome Italy",
    "Himalayas mountain landscape nature",
    "Modern office room interior",
]

def search_commons_files(query, limit=15):
    """Search Wikimedia Commons for image file titles matching a query."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": "6",  # File namespace
        "srlimit": str(limit),
        "format": "json"
    }
    url = f"https://commons.wikimedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            items = data.get("query", {}).get("search", [])
            return [it["title"] for it in items if it.get("title")]
    except Exception as e:
        print(f"Error searching '{query}': {e}")
        return []

def get_image_urls(titles):
    """Retrieve direct 800px thumbnail URLs and licenses for given file titles."""
    if not titles:
        return []
    
    results = []
    # Batch query titles (up to 20 at a time)
    batch_size = 15
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|thumburl|size|extmetadata",
            "iiurlwidth": "800",
            "format": "json"
        }
        url = f"https://commons.wikimedia.org/w/api.php?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                pages = data.get("query", {}).get("pages", {})
                for pid, p in pages.items():
                    infos = p.get("imageinfo", [])
                    if not infos:
                        continue
                    info = infos[0]
                    thumb = info.get("thumburl") or info.get("url")
                    orig = info.get("url")
                    width = info.get("thumbwidth") or info.get("width", 0)
                    height = info.get("thumbheight") or info.get("height", 0)
                    meta = info.get("extmetadata", {})
                    license_name = meta.get("LicenseShortName", {}).get("value", "CC / Public Domain")
                    artist = meta.get("Artist", {}).get("value", "Unknown")
                    if thumb and (thumb.lower().endswith(('.jpg', '.jpeg', '.png')) or 'jpg' in thumb.lower() or 'png' in thumb.lower()):
                        results.append({
                            "title": p.get("title"),
                            "url": thumb,
                            "original_url": orig,
                            "width": width,
                            "height": height,
                            "license": license_name,
                            "artist": artist
                        })
            time.sleep(0.5)
        except Exception as e:
            print(f"Error fetching metadata for batch: {e}")
    return results

def download_and_verify(url, target_path):
    """Download image and verify it is a valid, readable image with minimum dimensions."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            content = resp.read()
            with open(target_path, "wb") as f:
                f.write(content)
        # Verify with PIL
        with Image.open(target_path) as img:
            img.verify()
        with Image.open(target_path) as img:
            w, h = img.size
            if w < 100 or h < 100:
                target_path.unlink(missing_ok=True)
                return False
        return True
    except Exception as e:
        if target_path.exists():
            target_path.unlink(missing_ok=True)
        return False

def prepare_all(images_per_class=20):
    catalog = {"monuments": {}, "negatives": []}
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        for cls in MONUMENT_QUERIES.keys():
            (DATASET_ROOT / split / cls).mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "negatives").mkdir(parents=True, exist_ok=True)

    print("=== Downloading authentic monument images from Wikimedia Commons ===")
    for cls, queries in MONUMENT_QUERIES.items():
        print(f"\nProcessing class: {cls}")
        seen_titles = set()
        all_titles = []
        for q in queries:
            t = search_commons_files(q, limit=12)
            for item in t:
                if item not in seen_titles:
                    seen_titles.add(item)
                    all_titles.append(item)
        
        print(f"Found {len(all_titles)} distinct image candidates for {cls}. Fetching URLs...")
        image_meta = get_image_urls(all_titles)
        
        # Download verified images
        verified_images = []
        for idx, meta in enumerate(image_meta):
            if len(verified_images) >= images_per_class:
                break
            temp_file = DATASET_ROOT / f"temp_{cls}_{idx}.jpg"
            if download_and_verify(meta["url"], temp_file):
                verified_images.append((temp_file, meta))
                print(f"  [OK] ({len(verified_images)}/{images_per_class}) {meta['title'][:50]}")
            else:
                print(f"  [SKIP] Could not verify {meta['title'][:40]}")
            time.sleep(0.3)

        # Split into train (70%), val (15%), test (15%)
        # For ~20 images: 14 train, 3 val, 3 test
        total = len(verified_images)
        n_train = max(1, int(total * 0.70))
        n_val = max(1, int(total * 0.15))
        
        splits = {
            "train": verified_images[:n_train],
            "val": verified_images[n_train:n_train + n_val],
            "test": verified_images[n_train + n_val:]
        }
        
        catalog["monuments"][cls] = {"train": [], "val": [], "test": []}
        for split_name, items in splits.items():
            for i, (tmp_path, meta) in enumerate(items):
                final_name = f"{cls}_{split_name}_{i+1:02d}.jpg"
                final_path = DATASET_ROOT / split_name / cls / final_name
                tmp_path.rename(final_path)
                catalog["monuments"][cls][split_name].append({
                    "filename": final_name,
                    "title": meta["title"],
                    "url": meta["url"],
                    "original_url": meta["original_url"],
                    "license": meta["license"],
                    "artist": meta["artist"]
                })
        print(f"  --> {cls}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

    # Now download negative rejection images
    print("\n=== Downloading Negative / Rejection images ===")
    seen_neg = set()
    neg_titles = []
    for q in NEGATIVE_QUERIES:
        t = search_commons_files(q, limit=6)
        for item in t:
            if item not in seen_neg:
                seen_neg.add(item)
                neg_titles.append(item)
    neg_meta = get_image_urls(neg_titles)
    neg_count = 0
    for idx, meta in enumerate(neg_meta):
        if neg_count >= 15:
            break
        target_name = f"negative_{neg_count+1:02d}.jpg"
        target_path = DATASET_ROOT / "negatives" / target_name
        if download_and_verify(meta["url"], target_path):
            neg_count += 1
            catalog["negatives"].append({
                "filename": target_name,
                "title": meta["title"],
                "url": meta["url"],
                "original_url": meta["original_url"],
                "license": meta["license"],
                "artist": meta["artist"]
            })
            print(f"  [OK] Negative ({neg_count}) {meta['title'][:50]}")
        time.sleep(0.3)

    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(f"\n[DONE] Dataset preparation complete! Catalog saved to {CATALOG_PATH}")

if __name__ == "__main__":
    prepare_all(images_per_class=18)
