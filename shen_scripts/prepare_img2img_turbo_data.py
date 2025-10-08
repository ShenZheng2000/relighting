import os
import random
from PIL import Image
import json

# Skip known bad images
skip_list = {"Boden_Girl_Shorts_002.png", "Eres_women_swimwear_303.png"}

# Define the root input and output directories
root_dir = "/home/shenzhen/Relight_Projects/relighting/outputs"
relight_type = "candlelight_1" # NOTE: change this everytime to avoid overwriting previous data
output_dir = "/scratch/shenzhen/exp_10_2"
output_dir = os.path.join(output_dir, relight_type)

# Define prompt
prompt = "Relit with warm candlelight in a dimly lit indoor setting, casting soft, flickering shadows and enveloping the subject in golden-orange tones to create a cozy, nostalgic mood."

# Save cropped base and relight images
def save_crops(image_paths, base_folder, relight_folder, start_idx=0, img_dim=784):
    idx = start_idx
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            w, h = img.size
            base_crop = img.crop((w - 2 * img_dim, 0, w - img_dim, img_dim))
            relight_crop = img.crop((w - img_dim, 0, w, img_dim))
            fname = f"{idx}.png"
            base_crop.save(os.path.join(base_folder, fname))
            relight_crop.save(os.path.join(relight_folder, fname))
            idx += 1
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    return idx

def create_json(num_images, prompt, output_path):
    data = {f"{i}.png": prompt for i in range(num_images)}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved JSON to {output_path}")

# Output subdirectories
save_base_train_A = os.path.join(output_dir, "train_A")
save_relight_train_B = os.path.join(output_dir, "train_B")
save_base_test_A = os.path.join(output_dir, "test_A")
save_relight_test_B = os.path.join(output_dir, "test_B")

# Create output directories
for d in [save_base_train_A, save_relight_train_B, save_base_test_A, save_relight_test_B]:
    os.makedirs(d, exist_ok=True)

# Collect valid image paths
valid_image_paths = []

# Iterate over folder for image filtering and selection.
for exp_folder in sorted(os.listdir(root_dir)):
    subfolder = os.path.join(root_dir, exp_folder, relight_type)

    if not os.path.isdir(subfolder):
        continue

    invalid_path = os.path.join(subfolder, "invalid.txt")
    invalid_files = set()

    if os.path.exists(invalid_path):
        with open(invalid_path, "r") as f:
            invalid_files = set(line.strip() for line in f.readlines())

    for fname in sorted(os.listdir(subfolder)):
        if (
            fname.lower().endswith(".png")
            and fname not in invalid_files
            and fname not in skip_list      # <-- add this line
        ):
            valid_image_paths.append(os.path.join(subfolder, fname))

# Shuffle and split 80/20 (NOTE: use seed=0 for reproducibility)
random.seed(0)
random.shuffle(valid_image_paths)
split_idx = int(0.8 * len(valid_image_paths))
train_files = valid_image_paths[:split_idx]
test_files = valid_image_paths[split_idx:]

# # Save crops for train and test sets
save_crops(train_files, save_base_train_A, save_relight_train_B)
save_crops(test_files, save_base_test_A, save_relight_test_B)

# Save JSONs
num_train = len(train_files)
num_test = len(test_files)
create_json(num_train, prompt, os.path.join(output_dir, "train_prompts.json"))
create_json(num_test, prompt, os.path.join(output_dir, "test_prompts.json"))