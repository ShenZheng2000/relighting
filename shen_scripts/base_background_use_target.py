import os
from PIL import Image
import numpy as np

def apply_mask_and_replace(raw_dir, bg_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".png") or "_groundedsam2" in fname:
            continue  # skip masks and non-png

        raw_path = os.path.join(raw_dir, fname)
        mask_path = os.path.join(raw_dir, fname.replace(".png", "_groundedsam2.png"))
        bg_path = os.path.join(bg_dir, fname)
        output_path = os.path.join(output_dir, fname)

        if not os.path.exists(mask_path) or not os.path.exists(bg_path):
            print(f"[SKIP] No mask or background for {fname}")
            continue

        img = Image.open(raw_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        bg = Image.open(bg_path).convert("RGB")

        img_np = np.array(img)
        mask_np = np.array(mask)
        bg_np = np.array(bg)

        # Replace background pixels
        img_np[mask_np == 255] = bg_np[mask_np == 255]

        Image.fromarray(img_np).save(output_path)
        print(f"[DONE] {output_path}")

        # # Save the mask for debugging
        # exit()

# Run for both train and test
base = "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24_use_target_bg"
apply_mask_and_replace(
    os.path.join(base, "train_A_raw"),
    os.path.join(base, "train_B"),
    os.path.join(base, "train_A")
)
apply_mask_and_replace(
    os.path.join(base, "test_A_raw"),
    os.path.join(base, "test_B"),
    os.path.join(base, "test_A")
)
