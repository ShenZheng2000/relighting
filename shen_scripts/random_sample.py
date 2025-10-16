import os
import random
import shutil

# ----------------------------
# Settings
# ----------------------------
src_dir = "/home/shenzhen/Datasets/dataset_with_garment"   # directory with 23136 folders
dst_dir = "/home/shenzhen/Datasets/dataset_with_garment_debug_1000"          # destination directory
num_sample = 1000
seed = 0
# ----------------------------

# Ensure reproducibility
random.seed(seed)

# Get all top-level folders
all_folders = [f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, f))]
print(f"Total folders found: {len(all_folders)}")

# Sample 1000 folders
sampled_folders = random.sample(all_folders, num_sample)
print(f"Sampling {len(sampled_folders)} folders...")

# Create destination root
os.makedirs(dst_dir, exist_ok=True)

# Copy sampled folders
for i, folder_name in enumerate(sampled_folders, 1):
    src_path = os.path.join(src_dir, folder_name)
    dst_path = os.path.join(dst_dir, folder_name)
    print(f"[{i}/{num_sample}] Copying: {folder_name}")
    shutil.copytree(src_path, dst_path)

print("✅ Done! Sampled folders copied to:", dst_dir)