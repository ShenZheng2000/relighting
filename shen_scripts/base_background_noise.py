import os
from PIL import Image
import numpy as np

def apply_mask_and_save(raw_dir, output_dir, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Fix the random seed for reproducibility

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

        # NOTE: use Gaussian noise 
        # Generate Gaussian noise (standard normal distribution)
        noise = np.random.randn(*img_np.shape).astype(np.float32)

        # Normalize noise: zero mean, unit variance -> scale to [0, 255]
        noise = ((noise - noise.mean()) / (noise.std() + 1e-8)) * 64 + 128
        noise = np.clip(noise, 0, 255).astype(np.uint8)

        # Apply noise where mask is 255 (background)
        img_np[mask_np == 255] = noise[mask_np == 255]

        Image.fromarray(img_np).save(output_path)
        print(f"[DONE] {output_path}")

        # Exit for debugging
        # exit()

# Run for both train and test
base = "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_24_use_noise_bg"

# You can set a fixed seed (like 0) for reproducibility
apply_mask_and_save(os.path.join(base, "train_A_raw"), os.path.join(base, "train_A"), seed=0)
apply_mask_and_save(os.path.join(base, "test_A_raw"), os.path.join(base, "test_A"), seed=0)