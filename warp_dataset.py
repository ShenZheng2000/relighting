# import os
# os.environ["INSIGHTFACE_FORCE_TORCH"] = "1"
# import cv2
# import shutil
# import torch
# from insightface.app import FaceAnalysis
# from warp_utils.warping_layers import PlainKDEGrid, warp, load_img_warp, save_img_warp, invert_grid

# # ---- NEW: import shared config & load ----
# from utils import parse_arguments, load_config

# import argparse
# import sys


# # ============================================================
# # ✅ Add a local-only --bw arg without touching your parser
# #    (strip it from sys.argv so parse_arguments() won't see it)
# # ============================================================
# _extra = argparse.ArgumentParser(add_help=False)
# _extra.add_argument(
#     "--bw",
#     type=int,
#     default=512,
#     help="Override bandwidth_scale for KDE warp (does not touch config)",
# )

# # Parse only --bw and remove it from argv before calling parse_arguments()
# _extra_args, _remaining = _extra.parse_known_args(sys.argv[1:])
# sys.argv = [sys.argv[0]] + _remaining   # <-- hide --bw from your parser

# args = parse_arguments()                 # your existing function (unchanged)
# setattr(args, "bw", _extra_args.bw)      # attach bw to args so downstream code can use it

# config = load_config(args)
# bandwidth_scale = args.bw

# print(f"🔧 Using bandwidth_scale = {bandwidth_scale} for this run")


# target_prefix = config.output_dir     # e.g. exp_10_9
# relight_type  = config.relight_type   # e.g. candlelight_1

# # ---- auto input/output ----
# input_root  = f"/ssd1/shenzhen/relighting/{target_prefix}/{relight_type}"
# output_root = f"/ssd1/shenzhen/relighting/{target_prefix}_warped_{bandwidth_scale}/{relight_type}"

# subfolders_to_warp = ["train_A", "test_A"]
# subfolders_to_copy = ["train_B", "test_B"]


# # Initialize face detector ONCE (not per image)
# face_app = FaceAnalysis(name='buffalo_l')
# face_app.prepare(ctx_id=0)  # GPU = 0

# def process_image(img_path, warped_dir):
#     img_cv2 = cv2.imread(img_path)
#     if img_cv2 is None:
#         print(f"⚠️ Cannot read: {img_path}")
#         return

#     height, width = img_cv2.shape[:2]
#     img = load_img_warp(img_path).to("cuda")

#     # ----------------------------------------
#     # Face detection (InsightFace RetinaFace)
#     # ----------------------------------------
#     faces = face_app.get(img_cv2)
#     if len(faces) > 0:
#         faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
#         x1, y1, x2, y2 = map(int, faces[0].bbox)
#         # TODO: think if we need to clamp here
#         # TODO: how to incorporate with warp_pipeline.py in another repo? 
#         x1 = max(0, min(x1, width - 1))
#         y1 = max(0, min(y1, height - 1))
#         x2 = max(0, min(x2, width - 1))
#         y2 = max(0, min(y2, height - 1))
#     else:
#         print(f"❌ No face detected — fallback ({img_path})")
#         x1, y1, x2, y2 = 0, 0, width, height

#     bboxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32).unsqueeze(0).to("cuda")

#     # ----------------------------------------
#     # Compute forward grid + warp
#     # ----------------------------------------
#     grid_net = PlainKDEGrid(
#         input_shape=(height, width),
#         output_shape=(height, width),
#         separable=True,
#         bandwidth_scale=bandwidth_scale,   # <<< CHANGED (formerly 512)
#         amplitude_scale=1.0,
#     ).to("cuda")

#     grid = grid_net(img, gt_bboxes=bboxes)     # forward warp grid
#     warped_img = warp(grid, img)               # warped image

#     # ----------------------------------------
#     # Compute inverse grid NOW
#     # ----------------------------------------
#     inverse_grid = invert_grid(grid, (1, 3, height, width), separable=True)

#     # ----------------------------------------
#     # Save warped image + inverse grid ONLY
#     # ----------------------------------------
#     base = os.path.splitext(os.path.basename(img_path))[0]
#     warped_path = os.path.join(warped_dir, f"{base}.png")
#     inv_grid_path = os.path.join(warped_dir, f"{base}.inv.pth")

#     save_img_warp(warped_img, warped_path)
#     torch.save(inverse_grid.cpu(), inv_grid_path)

#     print(f"💾 Saved warped → {warped_path}")
#     print(f"💾 Saved inverse grid → {inv_grid_path}")

# def process_and_warp_folder(sub_folder):
#     src_dir = os.path.join(input_root, sub_folder)
#     dst_dir = os.path.join(output_root, sub_folder)
#     os.makedirs(dst_dir, exist_ok=True)

#     imgs = [f for f in os.listdir(src_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
#     imgs.sort()

#     for fname in imgs:
#         src_img = os.path.join(src_dir, fname)
#         process_image(src_img, dst_dir)

# def copy_folder(sub_folder):
#     src_dir = os.path.join(input_root, sub_folder)
#     dst_dir = os.path.join(output_root, sub_folder)
#     os.makedirs(dst_dir, exist_ok=True)

#     imgs = [f for f in os.listdir(src_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
#     imgs.sort()

#     for fname in imgs:
#         src_img = os.path.join(src_dir, fname)
#         dst_img = os.path.join(dst_dir, fname)
#         shutil.copy2(src_img, dst_img)
#     print(f"📁 Copied {sub_folder} → {dst_dir} (hard copy)")

# if __name__ == "__main__":
#     os.makedirs(output_root, exist_ok=True)

#     # Warp train_A + test_A
#     for sub in subfolders_to_warp:
#         process_and_warp_folder(sub)

#     # Hard-copy train_B + test_B
#     for sub in subfolders_to_copy:
#         copy_folder(sub)

#     # ✅ copy JSON prompt files (must exist for PairedDataset)
#     for json_name in ["train_prompts.json", "test_prompts.json"]:
#         src = os.path.join(input_root, json_name)
#         dst = os.path.join(output_root, json_name)
#         if os.path.exists(src):
#             shutil.copy2(src, dst)
#             print(f"📄 Copied {json_name} → {dst}")
#         else:
#             print(f"⚠️ {json_name} not found in {input_root}, skipped")
#     print("✅ Warp + Copy complete. Warped dataset is now self-contained ✅")
