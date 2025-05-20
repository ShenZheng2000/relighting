import os
import random
import json
import pandas as pd
from PIL import Image

# Input config
sheet_path = "Seed_Direction_4_23.xlsx"
base_dir = "/home/shenzhen/Relight_Projects/relighting-comparison/outputs"
relight_type = "golden_hour_back_1"
save_ = "/home/shenzhen/Relight_Projects/img2img-turbo/data/Seed_Direction_4_23"
save_base_train_A = os.path.join(save_, "train_A_raw")
save_relight_train_B = os.path.join(save_, "train_B")
save_base_test_A = os.path.join(save_, "test_A_raw")
save_relight_test_B = os.path.join(save_, "test_B")

# Create folders
for folder in [save_base_train_A, save_relight_train_B, save_base_test_A, save_relight_test_B]:
    os.makedirs(folder, exist_ok=True)

# Load sheet
df = pd.read_excel(sheet_path)

# Initialize
grand_total = 0
grand_filtered = 0
grand_valid = 0
random.seed(0)
save_idx = 0

# Prompt to be assigned to each saved image
relight_prompt = "Relit with golden-hour sunlight from behind, softly outlining the subject in amber tones, casting long, fading shadows, and creating a calm, atmospheric glow."

# Dictionaries for saving prompts
train_prompts = {}
test_prompts = {}

for idx, row in df.iterrows():
    seed = row["Seed"]
    candidates = row[2:].dropna().tolist()

    if all(str(c).strip().lower() in ["", "start here!", "end here!"] for c in candidates):
        continue

    skip_filenames = [f for f in candidates if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    folder_name = f"exp_4_13_v{seed}/{relight_type}"
    folder_path = os.path.join(base_dir, folder_name)

    if not os.path.isdir(folder_path):
        print(f"[Folder missing] {folder_path}")
        continue

    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    valid_files = [f for f in all_files if f not in skip_filenames]
    total = len(all_files)
    filtered = len(all_files) - len(valid_files)
    valid = len(valid_files)

    print(f"[{folder_name}] Total: {total}, Filtered: {filtered}, Valid: {valid}")
    grand_total += total
    grand_filtered += filtered
    grand_valid += valid



    # Shuffle and split 80/20
    random.shuffle(valid_files)
    split_idx = int(0.8 * len(valid_files))
    train_files = valid_files[:split_idx]
    test_files = valid_files[split_idx:]

    for file_list, base_folder, relight_folder, prompt_dict in [
        (train_files, save_base_train_A, save_relight_train_B, train_prompts),
        (test_files, save_base_test_A, save_relight_test_B, test_prompts)
    ]:
        for file in file_list:
            img_path = os.path.join(folder_path, file)
            try:
                img = Image.open(img_path)
                w, h = img.size
                base_crop = img.crop((w - 2 * 784, 0, w - 784, 784))
                relight_crop = img.crop((w - 784, 0, w, 784))
                filename = f"{save_idx}.png"
                base_crop.save(os.path.join(base_folder, filename))
                relight_crop.save(os.path.join(relight_folder, filename))
                prompt_dict[filename] = relight_prompt
                save_idx += 1
            except Exception as e:
                print(f"Failed to process image: {img_path}, Error: {e}")

# Save the prompt dictionaries to JSON
with open(os.path.join(save_, "train_prompts.json"), "w") as f:
    json.dump(train_prompts, f, indent=4)

with open(os.path.join(save_, "test_prompts.json"), "w") as f:
    json.dump(test_prompts, f, indent=4)

print(f"\n[SUMMARY] Total: {grand_total}, Filtered: {grand_filtered}, Valid: {grand_valid}")
print(f"Saved prompt JSONs to:\n - train_prompts.json\n - test_prompts.json")

# after that, run groundedsam2 to generate mask, and then base_background_white.py to apply mask