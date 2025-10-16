import os
import random
from PIL import Image
import json
from tqdm import tqdm  # ✅ progress bar

# Skip known bad images
skip_list = {
            # from debug_100
            "Boden_Girl_Shorts_002.png", 
             "Eres_women_swimwear_303.png",

            # from debug 1000
             "174_WOMEN_TOPS_333.png",
             "annaoctober_women_suiting_001.png",
             "Badtaste_Women_Tops_175.png",
             "Crycry_paradise_Women_Bottoms_001.png",
             "houseofcb_women_dresses_033.png"

            # from big face 1000
            "acnestudios_women_outerwear_039.png",
            "balmain_women_t_shirts_034.png",
            "Bananarepublic_R2_Women_Sweaters_Cardigan_064.png",
            "baobabswim_women_resortwear_018.png",
            "baobabswim_women_resortwear_163.png",
            "casall_women_jackets_025.png",
            "casall_women_longsleeved-tops_020.png",
            "Cigaberry_Women_Tops_010.png",
            "DIDU_Women_Tops_025.png",
            "Diotima_Women_Trousers_003.png",
            "DistrictVision_Women_Tops_009.png",
            "dodobaror_women_tops_003.png",
            "finisterre_menyulexwetsuits_004.png",
            "houseofcb_women_tops_181.png",
            "isabelle_quinn_women_sets_027.png",
            "candlelight_1/kan_women_skirt_015.png",
            "candlelight_1/Lagence_Women_Denim_109.png",
            "candlelight_1/Lagence_Women_Denim_148.png",
            "candlelight_1/lichi_women_sets_200.png",
            "candlelight_1/maisonmargiela_R2_men_shirts_040.png",
            "candlelight_1/MOSSANDSPY_WOMEN_DRESS_052.png",
            "candlelight_1/Nanagotti_Women_Dresses_022.png",
            "candlelight_1/Oakandfort_R2_Men_Bottoms_015.png",
            "candlelight_1/Oakandfort_R2_Women_Outerwear_002.png",
            "candlelight_1/Oakandfort_R2_Women_Outerwear_003.png",
            "candlelight_1/oakandfort_womenknitwear_036.png",
            "candlelight_1/TAGEECHITA_WOMEN_TOPS_006.png",
            "candlelight_1/ZARA_R2_WOMEN_TROUSERS_004.png",
            "candlelight_1/acnestudios_women_outerwear_039.png",
            "candlelight_1/balmain_women_t_shirts_034.png",
            "candlelight_1/Bananarepublic_R2_Women_Sweaters_Cardigan_064.png",
            "candlelight_1/baobabswim_women_resortwear_018.png",
            "candlelight_1/baobabswim_women_resortwear_163.png",
            "candlelight_1/casall_women_jackets_025.png",
            "candlelight_1/casall_women_longsleeved-tops_020.png",
            "candlelight_1/CELTICANDCO_WOMEN_KNITWEARS_131.png",
            "candlelight_1/Cigaberry_Women_Tops_010.png",
            "candlelight_1/dandydelmar_women_sets_007.png",
            "candlelight_1/dandydelmar_women_sets_009.png",
            "candlelight_1/DIDU_Women_Tops_025.png",
            "candlelight_1/Diotima_Women_Trousers_003.png",
            "candlelight_1/DistrictVision_Women_Tops_009.png",
            "candlelight_1/dodobaror_women_tops_003.png",
            "candlelight_1/finisterre_menyulexwetsuits_004.png",
            "candlelight_1/houseofcb_women_tops_181.png",
            "candlelight_1/Lagence_Women_Denim_109.png",
            "candlelight_1/Lagence_Women_Denim_148.png",
            "candlelight_1/lichi_women_sets_200.png",
            "candlelight_1/maisonmargiela_R2_men_shirts_040.png",
            "candlelight_1/MOSSANDSPY_WOMEN_DRESS_052.png",
            "candlelight_1/Nanagotti_Women_Dresses_022.png",
            "candlelight_1/Oakandfort_R2_Men_Bottoms_015.png",
            "candlelight_1/Oakandfort_R2_Women_Outerwear_002.png",
            "candlelight_1/Oakandfort_R2_Women_Outerwear_003.png",
            "candlelight_1/oakandfort_womenknitwear_036.png",
            "candlelight_1/paxphilomena_men_short_sleeves_shirts_003.png",
            "candlelight_1/TAGEECHITA_WOMEN_TOPS_006.png",
            "candlelight_1/ZARA_R2_WOMEN_DRESS_173.png",
            "candlelight_1/ZARA_R2_WOMEN_TROUSERS_004.png"
        }

# Configuration
target_prefix = "exp_10_11"  # only process folders with this prefix
relight_type = "candlelight_1" # NOTE: change this everytime to avoid overwriting previous data

# Directories
root_dir = "/home/shenzhen/Relight_Projects/relighting/outputs"
output_dir = os.path.join("/scratch/shenzhen/relighting", target_prefix, relight_type)

# Define prompt
prompt = "Relit with warm candlelight in a dimly lit indoor setting, casting soft, flickering shadows and enveloping the subject in golden-orange tones to create a cozy, nostalgic mood."

# Save cropped base and relight images
def save_crops(image_paths, base_folder, relight_folder, start_idx=0, img_dim=784):
    idx = start_idx

    for img_path in tqdm(image_paths, desc=f"Cropping {os.path.basename(base_folder)}"):
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

    if not exp_folder.startswith(target_prefix):
        continue  # skip others

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