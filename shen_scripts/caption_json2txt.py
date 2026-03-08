import os
import json
from pathlib import Path

# -------- paths -------- #
TRAIN_JSON = "/ssd0/shenzhen/Datasets/depth/workzone_segm/annotations/instances_train_gps_split.json"
VAL_JSON   = "/ssd0/shenzhen/Datasets/depth/workzone_segm/annotations/instances_val_gps_split.json"

IMAGE_DIR = "/ssd0/shenzhen/Datasets/depth/workzone_segm/boston/image"
 # NOTE: not trained on boston, caption useless for now, but keep it for future use
CAPTION_DIR = "/ssd0/shenzhen/Datasets/depth/workzone_segm/boston/caption"
os.makedirs(CAPTION_DIR, exist_ok=True)


def add_captions_from_coco(coco_json_path: str, name_to_caption: dict):
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    added = 0
    overwritten = 0
    for img in coco.get("images", []):
        fname = img.get("file_name", "")
        caption = img.get("scene_description", "").strip()
        if not fname:
            continue

        # normalize to basename to match img_path.name
        key = os.path.basename(fname)

        if key in name_to_caption:
            overwritten += 1
        name_to_caption[key] = caption
        added += 1

    return added, overwritten


# build mapping: file_name(basename) -> caption
name_to_caption = {}

a1, o1 = add_captions_from_coco(TRAIN_JSON, name_to_caption)
a2, o2 = add_captions_from_coco(VAL_JSON, name_to_caption)

print(f"Loaded captions into map: {len(name_to_caption)} unique filenames")
print(f"  train added={a1}, overwrote={o1}")
print(f"  val   added={a2}, overwrote={o2}")

# -------- iterate images -------- #
image_paths = list(Path(IMAGE_DIR).glob("*"))

written = 0
missing = 0

for img_path in image_paths:
    fname = img_path.name
    stem = img_path.stem

    if fname not in name_to_caption:
        missing += 1
        continue

    caption = name_to_caption[fname]

    out_path = os.path.join(CAPTION_DIR, f"{stem}.txt")
    with open(out_path, "w") as f:
        f.write(caption)

    written += 1

print(f"✅ Written: {written}")
print(f"⚠️ Missing captions: {missing}")
print("Done.")