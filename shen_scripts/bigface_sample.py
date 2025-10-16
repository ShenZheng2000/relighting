import os
import shutil

# ----------------------------
# Settings
# ----------------------------

src_dir = "/home/shenzhen/Datasets/dataset_with_garment"        # original dataset

# num_sample = 1000
num_sample = 100

# dst_dir = "/home/shenzhen/Datasets/dataset_with_garment_bigface_1000"  # output subset
dst_dir = f"/home/shenzhen/Datasets/dataset_with_garment_bigface_{num_sample}"  # output subset

ranking_file = f"/home/shenzhen/Relight_Projects/face_detection/retinaface/face_area_ranking.txt"  # file like "1,Qnigirls_Women_Tops_163,80"

# ----------------------------

# Read ranking file
ranked_folders = []
with open(ranking_file, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            folder_name = parts[1].strip()
            ranked_folders.append(folder_name)

# Limit to top-N
ranked_folders = ranked_folders[:num_sample]
print(f"Selecting top {len(ranked_folders)} folders from ranking file.")

# Create destination root
os.makedirs(dst_dir, exist_ok=True)

# Copy folders
for i, folder_name in enumerate(ranked_folders, 1):
    src_path = os.path.join(src_dir, folder_name)
    dst_path = os.path.join(dst_dir, folder_name)

    if not os.path.exists(src_path):
        print(f"[Skip] Folder not found: {folder_name}")
        continue

    print(f"[{i}/{len(ranked_folders)}] Copying: {folder_name}")
    shutil.copytree(src_path, dst_path)

print(f"\n✅ Done! Copied top {len(ranked_folders)} folders to {dst_dir}")