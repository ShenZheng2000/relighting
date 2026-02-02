# NOTE: this is the legacy inference version based on this format. 

# dataset_with_garment_bigface_100/
# ├── 8seconds_men_shirts_034/
# │   ├── pre_processing/
# │       ├── black_fg_mask_groundedsam2.png
# │   ├── bdy_2.jpg
# │   └── gar_0.jpg
# │   └── gpt_annotation__bdy_2.txt
# ├── 09WOMEN_WOMEN_BLOUSE_167/
# ├── 09WOMEN_WOMEN_PANTS_418/
# ├── Adidas_R2_Men_Jackets_216/
# └── Adsb_Women_Skirts_008/
# └── ...


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
    # relighting_prompts,              # For inference (t2i) prompts.
    # relighting_prompts_2,            # For outpainting prompts.
    concat_images_side_by_side,
    parse_arguments,
    load_config,
    load_depth_map,
    prepare_canvas_and_mask,
    process_body_mask,
    extract_background,
    extract_foreground,
    process_depth_map,
    resize_mask_to_canvas,
    apply_background_override
)

'''
cd outputs/
python -m http.server 8000
'''

# ---------------- Outpainting functions ---------------- #
def run_outpainting(subfolder_path, config, pipe_outpaint, outpaint_prompts, depth_model, use_v2):
    """
    Performs outpainting on a given subfolder.
    Saves two versions:
      - Base outpaint (without fg mask) using the base prompt.
      - Relight outpaint (with fg mask) using the selected relight prompt.
    Optionally saves depth maps if config.save_depth_maps is True.
    Returns paths to the base and relight outpainted images.
    """
    # Collect required files.
    annotation_files = glob.glob(os.path.join(subfolder_path, "*.txt"))
    image_files = glob.glob(os.path.join(subfolder_path, "bdy_*"))
    mask_files = glob.glob(os.path.join(subfolder_path, "pre_processing/body_mask/*.png"))

    if len(annotation_files) == 0 or len(image_files) == 0 or len(mask_files) == 0:
        print(f"Skipping {subfolder_path} due to missing annotation/image/mask.")
        return None, None

    annotation_path = annotation_files[0]
    source_image_path = image_files[0]
    body_mask_rgba_path = mask_files[0]

    # NOTE: decide to use original or grounded sam2 mask
    if config.use_groundedsam2:
        black_mask_path = os.path.join(subfolder_path, "pre_processing/black_fg_mask_groundedsam2.png")
    else:
        black_mask_path = os.path.join(subfolder_path, "pre_processing/black_fg_mask.png")

    if os.path.exists(black_mask_path):
        print("Skipping mask processing; using existing masks.")
        body_mask = Image.open(black_mask_path).convert("L")
    else:
        # _, black_mask_path = process_body_mask(body_mask_rgba_path, white_mask_path)
        black_mask_path = process_body_mask(body_mask_rgba_path, black_mask_path)
        body_mask = Image.open(black_mask_path).convert("L")

    # Load source image and annotation.
    source_image = Image.open(source_image_path).convert("RGB")

    with open(annotation_path, "r") as f:
        base_prompt = f.read().strip()
    
    base_prompt = apply_background_override(base_prompt, config)
    
    if config.extract_bg_from_base_prompt:
        base_prompt = extract_background(base_prompt)
        print("Extracted base_prompt:", base_prompt)

    # Use configuration parameters.
    target_width = config.width
    target_height = config.height
    cfg_outpaint = config.cfg_outpaint
    num_inference_steps = config.num_inference_steps  # outpainting steps
    crop_to_foreground = config.crop_to_foreground
    upper_crop = config.upper_crop  # Whether to keep upper half only for outpainting

    # Outpainting without rectangle (not fg mask) (base version).
    if not config.relight_image_only:
        canvas_base_no, mask_base_no, _, _, _ = prepare_canvas_and_mask(
            source_image, target_width, target_height,
            apply_fg_mask=False,
            body_mask=body_mask,
            crop_to_foreground=crop_to_foreground,
            upper_crop=upper_crop,
        )
        img_out_base_no = pipe_outpaint(
            prompt=base_prompt,
            image=canvas_base_no,
            mask_image=mask_base_no,
            height=target_height,
            width=target_width,
            guidance_scale=cfg_outpaint,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(42)
        ).images[0]

    # Outpainting with fg mask (relight version).
    # TODO: check if outpaint based on masked, or entire rectangle regions!!!!!!!!!!!!!!!!
    relight_id = config.relight_type
    relight_prompt = outpaint_prompts.get(relight_id, "")
    canvas_relight, mask_relight, relight_scale, relight_x_offset, relight_y_offset = prepare_canvas_and_mask(
        source_image, target_width, target_height,
        apply_fg_mask=True, 
        body_mask=body_mask,
        crop_to_foreground=crop_to_foreground,
        upper_crop=upper_crop,
    )

    img_out_relight = pipe_outpaint(
        prompt=relight_prompt,
        image=canvas_relight,
        mask_image=mask_relight,
        height=target_height,
        width=target_width,
        guidance_scale=cfg_outpaint,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(config.seed),
    ).images[0]

    # Instead of saving inside the input subfolder, we save to the output directory.
    outpaint_folder = os.path.join("outpaint", config.output_dir, relight_id, os.path.basename(subfolder_path))
    os.makedirs(outpaint_folder, exist_ok=True)
    base_no_path = os.path.join(outpaint_folder, "img_out_base.png")
    relight_path = os.path.join(outpaint_folder, "img_out_relight.png")

    if not config.relight_image_only:
        img_out_base_no.save(base_no_path)

    img_out_relight.save(relight_path)
    print(f"Saved outpaint images in {outpaint_folder}")

    # Optionally save depth maps.
    if config.save_depth_maps:
        depth_base_path = os.path.join(outpaint_folder, "depth_base.png")
        depth_relight_path = os.path.join(outpaint_folder, "depth_relight.png")

        if not config.relight_image_only:
            process_depth_map(base_no_path, depth_base_path, depth_model, use_v2=config.use_depthanythingv2)

        process_depth_map(relight_path, depth_relight_path, depth_model, use_v2=config.use_depthanythingv2)
        # print("Saved depth maps.")

        # Replace relight foreground depth with base foreground depth if enabled.
        if config.copy_fg_depth:
            raise NotImplementedError("copy_fg_depth is not implemented yet.")

    return base_no_path, relight_path



def run_outpainting_loop(config, pipe_outpaint, outpaint_prompts, depth_model, use_v2):
    """
    Loops over subfolders in the input directory to run outpainting.
    """
    count = 0
    for subfolder in sorted(os.listdir(config.input_dir)):
        subfolder_path = os.path.join(config.input_dir, subfolder)
        if os.path.isdir(subfolder_path):
            run_outpainting(subfolder_path, config, pipe_outpaint, outpaint_prompts, depth_model, use_v2)
            count += 1
            if config.max_images and count >= config.max_images:
                print(f"Reached max_images limit: {config.max_images}. Stopping outpainting.")
                return

# ---------------- Inference functions ---------------- #
def process_subfolder_inference(subfolder_path, config, pipe_inference, prompts):
    """
    Processes a subfolder by running T2I inference.
    The final output is a concatenated image saved in the outputs directory.
    """
    # Load the source image and annotation.
    annotation_files = glob.glob(os.path.join(subfolder_path, "*.txt"))
    image_files = glob.glob(os.path.join(subfolder_path, "bdy_*"))
    
    if len(annotation_files) == 0 or len(image_files) == 0:
        print(f"Skipping inference for {subfolder_path} due to missing annotation/image.")
        return

    annotation_path = annotation_files[0]
    source_image_path = image_files[0]
    source_image = Image.open(source_image_path).convert('RGB')
    
    with open(annotation_path, "r") as f:
        base_prompt = f.read().strip()
    
    base_prompt = apply_background_override(base_prompt, config)

    if config.extract_fg_from_base_prompt_for_generation:
        base_prompt = extract_foreground(base_prompt)
        # print("Extracted base_prompt foreground for generation:", base_prompt)

    # Build the final prompt.
    relight_id = config.relight_type

    if relight_id not in prompts:
        raise KeyError(f"relight_type '{relight_id}' not found in prompt_version={config.prompt_version}")

    relight_prompt = prompts.get(relight_id, "")

    if config.relight_image_only:
        # Use only the relight prompt.
        final_prompt = relight_prompt
        output_width = config.width  # Single image width.
    else:
        # Use the 2x1 grid prompt.
        final_prompt = (
            f"A photo of a person in a 2 by 1 grid. "
            f"On the left, {base_prompt} "
            f"On the right, {relight_prompt}."
        )
        if config.final_prompt_2:
            print("Using alternate final prompt.")
            final_prompt = (
                f"A 2x1 image grid; "
                f"On the left, {base_prompt} "
                f"On the right, the same person {relight_prompt}."
            )
        output_width = config.width * 2  # Grid: double the width.

    # Load or compute the depth map.
    depth_map_2x1 = load_depth_map(subfolder_path, config, relight_id)

    # Decide on image dimensions.
    if config.match_source_resolution:
        raise NotImplementedError("match_source_resolution is not implemented yet.")

    # Run inference (T2I).
    image = pipe_inference(
        prompt=final_prompt,
        control_image=depth_map_2x1,
        # height=height,
        # width=width * 2,  # 2x width for the grid
        height=config.height,
        width=output_width,
        guidance_scale=config.cfg,
        num_inference_steps=config.num_steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(config.seed),
    ).images[0]

    if config.resize_generated:
        raise NotImplementedError("resize_generated is not implemented yet.")

    # Save the final concatenated result.
    output_dir_final = os.path.join("outputs", config.output_dir, relight_id)
    os.makedirs(output_dir_final, exist_ok=True)
    output_filename = f"{os.path.basename(subfolder_path)}.png"
    output_path = os.path.join(output_dir_final, output_filename)
    concatenated_image = concat_images_side_by_side(source_image, image)
    concatenated_image.save(output_path)
    print(f"Saved final inference image: {output_path}")

def run_inference_loop(config, pipe_inference, prompts):
    """
    Loops over subfolders in the input directory to run inference.
    """
    count = 0
    for subfolder in sorted(os.listdir(config.input_dir)):
        subfolder_path = os.path.join(config.input_dir, subfolder)
        if os.path.isdir(subfolder_path):
            process_subfolder_inference(subfolder_path, config, pipe_inference, prompts)

            count += 1
            if config.max_images and count >= config.max_images:
                print(f"Reached max_images limit: {config.max_images}. Stopping inference.")
                return

# ---------------- Main function ---------------- #
def main():
    args = parse_arguments()
    base_config = load_config(args)  # Load once, then override seed each time
    device = f"cuda:{base_config.gpu}" if torch.cuda.is_available() else "cpu"

    # Set up depth model once
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

        # ---- Outpainting ----
        print("Loading outpainting pipeline...")
        pipe_outpaint = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16
        ).to(device)
        pipe_outpaint.set_progress_bar_config(disable=True)

        outpaint_prompts = relighting_prompt_versions[str(config.prompt_version)]
        run_outpainting_loop(config, pipe_outpaint, outpaint_prompts, depth_model, use_v2)
        del pipe_outpaint
        torch.cuda.empty_cache()

        # ---- Inference ----
        print("Loading inference pipeline...")
        pipe_inference = FluxControlPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Depth-dev",
            torch_dtype=torch.bfloat16
        ).to(device)
        pipe_inference.set_progress_bar_config(disable=True)

        prompts = relighting_prompt_versions[str(config.prompt_version)]
        run_inference_loop(config, pipe_inference, prompts)
        del pipe_inference
        torch.cuda.empty_cache()

    print("✅ All seeds finished.")

if __name__ == "__main__":
    main()
