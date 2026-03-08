# NOTE: this is the NEW inference version based on this format. 
# TODO later: rewrite this code in a nicer way to merge with the inference.py

# ========================= DATA LAYOUT =========================
# INPUT_FOLDER  (config.input_dir)
#
# $dataset_name/
# ├── image/
# │   ├── 00000_00.jpg
# │   ├── 00001_00.jpg
# │   └── ...
# ├── caption/
# │   ├── 00000_00.txt
# │   ├── 00001_00.txt
# │   └── ...
#
#
# OUTPUT_FOLDER  (relative to project root)
#
# outputs/
# └── <exp_name>_seedX/
#     └── <relight_type>/
#         ├── 00000_00.png
#         ├── 00001_00.png
#         └── ...
#

# ===============================================================
import os
import glob
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from diffusers import FluxControlPipeline
from image_gen_aux import DepthPreprocessor
from transformers import pipeline
from utils import relighting_prompt_versions  # make sure it's imported

# Import your utilities and pipelines.
from diffusers import FluxFillPipeline
from utils import (
    concat_images_side_by_side,
    parse_arguments,
    load_config,
    resize_mask_to_canvas,
    resolve_flat_paths,
    tile_2x1_pil,
    center_crop_pil
)


# ---------------- Inference functions ---------------- #
def process_subfolder_inference(subfolder_path, config, pipe_inference, prompts, depth_model, use_v2):
    """
    Processes a subfolder by running T2I inference.
    The final output is a concatenated image saved in the outputs directory.
    """

    # Flat layout: subfolder_path is actually a stem like "00000_00"
    stem = subfolder_path
    source_image_path, annotation_path, _ = resolve_flat_paths(config, stem)

    if (source_image_path is None) or (not os.path.exists(annotation_path)):
        print(f"Skipping inference for {stem} due to missing image/caption.")
        return

    source_image = Image.open(source_image_path).convert('RGB')

    if config.center_crop:
        source_image = center_crop_pil(source_image, config.width, config.height)

    with open(annotation_path, "r") as f:
        base_prompt = f.read().strip()

    # NOTE: MUST BE! no background override for now!
    # base_prompt = apply_background_override(base_prompt, config)

    # Build the final prompt.
    relight_id = config.relight_type

    if relight_id not in prompts:
        raise KeyError(f"relight_type '{relight_id}' not found in prompt_version={config.prompt_version}")

    relight_prompt = prompts.get(relight_id, "")

    # TODO: move this phrase into the config file later (let's call it light overwrite?)
    phrase = " in neutral daytime lighting"
    final_prompt = (
        "A 2x1 image grid; "
        f"On the left, {base_prompt}{phrase}. "
        f"On the right, the same scene {relight_prompt}."
    )
    output_width = config.width * 2  # Grid: double the width.

    # ---- compute depth from source image ----
    if use_v2:
        depth = depth_model(source_image)["depth"].convert("RGB")
    else:
        depth = depth_model(source_image)[0].convert("RGB")

    # tile to 2x1
    depth_map_2x1 = tile_2x1_pil(depth)
    depth_map_2x1 = depth_map_2x1.resize((output_width, config.height), Image.BILINEAR)

    # Run inference (T2I).
    image = pipe_inference(
        prompt=final_prompt,
        control_image=depth_map_2x1,
        height=config.height,
        width=output_width,
        guidance_scale=config.cfg,
        num_inference_steps=config.num_steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(config.seed),
    ).images[0]

    # Save the final concatenated result.
    output_dir_final = os.path.join("outputs", config.output_dir, relight_id)
    os.makedirs(output_dir_final, exist_ok=True)

    # ---- save depth to mirrored folder: depth/<exp>/<relight>/<stem>.png ----
    depth_dir_final = os.path.join("depth", config.output_dir, relight_id)
    os.makedirs(depth_dir_final, exist_ok=True)
    depth_save_path = os.path.join(depth_dir_final, f"{stem}.png")
    depth.save(depth_save_path)

    output_filename = f"{stem}.png"
    output_path = os.path.join(output_dir_final, output_filename)

    concatenated_image = concat_images_side_by_side(source_image, image)
    concatenated_image.save(output_path)
    print(f"Saved final inference image: {output_path}")


def run_inference_loop(config, pipe_inference, prompts, depth_model, use_v2):
    """
    Flat layout: loop over root/image/* and use stem as sample id.
    """
    count = 0
    image_dir = os.path.join(config.input_dir, "image")
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))

    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        process_subfolder_inference(stem, config, pipe_inference, prompts, depth_model, use_v2)

        count += 1
        if config.max_images and count >= config.max_images:
            print(f"Reached max_images limit: {config.max_images}. Stopping inference.")
            return


# ---------------- Main function ---------------- #
def main():
    args = parse_arguments()
    base_config = load_config(args)  # Load once, then override seed each time
    device = f"cuda:{base_config.gpu}" if torch.cuda.is_available() else "cpu"

    # Set up depth model once  (KEEP IT HERE)
    if base_config.use_depthanythingv2:
        depth_model = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
        use_v2 = True
    else:
        depth_model = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
        use_v2 = False

    for i in range(args.num_seeds):
        seed = args.seed_offset + i
        config = load_config(args)
        config.seed = seed
        config.output_dir = f"{os.path.splitext(os.path.basename(args.exp_config))[0]}_seed{seed}"

        print(f"\n=== Running seed {seed} ===")

        # ---- Inference ----
        print("Loading inference pipeline...")
        pipe_inference = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Depth-dev",
            torch_dtype=torch.bfloat16
        ).to(device)
        pipe_inference.set_progress_bar_config(disable=True)

        prompts = relighting_prompt_versions[str(config.prompt_version)]
        run_inference_loop(config, pipe_inference, prompts, depth_model, use_v2)

        del pipe_inference
        torch.cuda.empty_cache()

    print("✅ All seeds finished.")


if __name__ == "__main__":
    main()