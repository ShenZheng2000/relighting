import os
from PIL import Image
import numpy as np

def apply_mask_and_save(raw_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".png") or "_groundedsam2" in fname:
            continue  # skip masks and non-png

        raw_path = os.path.join(raw_dir, fname)
        mask_path = os.path.join(raw_dir, fname.replace(".png", "_groundedsam2.png"))
        output_path = os.path.join(output_dir, fname)

        if not os.path.exists(mask_path):
            print(f"[SKIP] No mask for {fname}")
            continue

        img = Image.open(raw_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_np = np.array(img)
        mask_np = np.array(mask)

        img_np[mask_np == 255] = [255, 255, 255]  # set background to white
        Image.fromarray(img_np).save(output_path)
        print(f"[DONE] {output_path}")

# Run for both train and test
base = "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_23"
apply_mask_and_save(os.path.join(base, "train_A_raw"), os.path.join(base, "train_A"))
apply_mask_and_save(os.path.join(base, "test_A_raw"), os.path.join(base, "test_A"))
