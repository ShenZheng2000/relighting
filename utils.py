from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
from omegaconf import OmegaConf
import os
import torch
import numpy as np
from image_gen_aux import DepthPreprocessor
from PIL import ImageOps

# depth_processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")

# NOTE: hardcoded prompts
relighting_prompts = {
    "golden_hour": "Relit by warm, golden-hour sunlight filtering through the trees, casting long, soft-edged shadows and creating a dreamy, atmospheric glow.",
    "moonlight": "Relit by soft, bluish moonlight streaming through an open window, casting gentle, diffused shadows and creating a serene, nighttime ambiance.",
    "noon_sunlight": "Relit by bright, overhead noon sunlight, creating strong, well-defined shadows with high contrast and a crisp, sharp atmosphere.",
    "neon_lights": "Relit by vibrant neon signs reflecting off wet pavement, casting colorful, dynamic glows in shades of pink, blue, and purple, creating a futuristic cyberpunk mood.",
    "candlelight": "Relit by flickering candlelight, casting soft, warm, golden hues with gentle, moving shadows, creating an intimate and cozy ambiance.",
    "spotlight": "Relit by a harsh, focused spotlight, creating extreme contrast with bright highlights and deep, sharp-edged shadows.",
    "thunderstorm": "Relit by flashes of lightning in a dark storm, creating dramatic, high-contrast illumination with deep shadows and eerie blue highlights.",
    "meteor_shower": "Relit by streaking meteors across the night sky, casting fleeting, dynamic glows with shifting highlights and deep cosmic shadows.",
    "volcanic_glow": "Relit by the fiery red-orange glow of molten lava, casting intense, flickering shadows with deep contrast and an apocalyptic atmosphere.",
    "foggy_morning": "Relit by soft, diffused morning light filtering through thick fog, muting colors and softening edges to create an ethereal, mysterious ambiance.",
}

# NOTE: I make the prompt longer (more details and more concret objects), and use relit instead of Relit (though it should not matter) => suitable for outpainting

relighting_prompts_2 = {
    "golden_hour": "relit by warm, golden-hour sunlight streaming through tall oak trees in a tranquil park, highlighting patches of wildflowers and casting long, soft-edged shadows across the grassy ground, creating a dreamy, atmospheric glow.",
    "noon_sunlight": "relit by bright, overhead noon sunlight blazing over a lively urban plaza, sharply defining every corner with crisp shadows and vivid highlights on modern glass and concrete structures, creating a dynamic and energetic daytime scene.",
    "neon_lights": "relit by vibrant neon lights reflecting off rain-slicked city streets, where electric hues of pink, blue, and purple burst from storefronts and billboards, bathing the surroundings in a futuristic, cyberpunk glow.",
    "candlelight": "relit by the gentle flicker of candlelight in an intimate setting, where warm amber tones softly dance over rustic wooden surfaces and delicate fabrics, creating a cozy, nostalgic ambiance filled with quiet charm.",
    "foggy_morning": "relit by the soft, diffused light of an early foggy morning in a quiet countryside, where gentle rays pierce through a thick mist over dew-covered fields and ancient trees, creating a serene, dreamlike atmosphere.",
    "moonlight": "relit by soft, bluish moonlight filtering through an open window framed by gently swaying curtains, casting pale, silvery light across worn wooden floorboards, scattered books, and the edge of a cozy armchair, creating a serene, nighttime glow filled with quiet stillness.",
}

# NOTE: indoor scenes
relighting_prompts_3 = {
    "golden_hour": "relit by warm, golden-hour sunlight streaming through a living room window, casting long, soft-edged shadows across wooden floors and gently illuminating cozy furniture, creating a calm, atmospheric glow.",
    "noon_sunlight": "relit by bright noon sunlight pouring through large apartment windows, creating sharp, well-defined shadows on white walls and highlighting indoor plants and shelves, adding energy to the quiet space.",
    "neon_lights": "relit by colorful neon lights from signs outside a downtown apartment, casting pink, blue, and purple glows across a modern interior with glass tables and framed artwork, creating a futuristic, urban ambiance.",
    "candlelight": "relit by the gentle flicker of candlelight in a dimly lit room, where warm amber tones dance over bookshelves, soft cushions, and old wooden furniture, creating a cozy and nostalgic atmosphere.",
    "foggy_morning": "relit by soft, diffused morning light seeping through sheer curtains in a quiet bedroom, muting colors and softening edges of the bed, rug, and potted plants, creating a peaceful, dreamlike indoor scene.",
    "moonlight": "relit by soft, bluish moonlight filtering through a bedroom window, casting pale shadows across the bed, nightstand, and curtains, filling the space with a serene and quiet nighttime mood.",
}

# NOTE: extremely simplified 
relighting_prompts_4 = {
    "golden_hour": "relit by golden-hour sunlight.",
    "noon_sunlight": "relit by bright noon sunlight.",
    "neon_lights": "relit by colorful neon lights.",
    "candlelight": "relit by warm candlelight.",
    "foggy_morning": "relit by diffused foggy morning light.",
    "moonlight": "relit by soft moonlight.",
}

# NOTE: explicitly mention the relighting direction => not working
relighting_prompts_5 = {
    "golden_hour_front": "Relit by golden-hour sunlight shining directly on the subject's face, illuminating the front evenly.",
    "golden_hour_side": "Relit by golden-hour sunlight coming from the side, casting soft shadows across the subject's face.",
    "golden_hour_back": "Relit by golden-hour sunlight coming from behind the subject, creating a warm rim light around the hair and shoulders.",
}

relighting_prompts_6 = {
    # NOTE: these prompts looks good!!!
    # "golden_hour_back": "relit with golden-hour sunlight from behind, face in shadow, rim glow",
    # "golden_hour_side": "relit with golden-hour sunlight from the side, one side lit, one side shadowed",
    # "golden_hour_front": "relit with golden-hour sunlight from the front, face fully lit, no shadows",

    "noon_sunlight_1": "Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood.",
    "golden_sunlight_1": "Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood.",
    "foggy_1": "Relit with dense fog in a muted outdoor setting, casting soft diffused shadows and surrounding the subject in pale gray light to create a quiet, atmospheric mood.",
    "moonlight_1": "Relit with cold moonlight in a minimalist nighttime scene, casting crisp soft shadows and bathing the subject in icy blue highlights to create a tranquil, distant mood.",
    "dusk_backlit_1": "Relit with dramatic dusk backlighting after sunset, casting the subject into a dark silhouette while the sky fades from pale blue to deep indigo.",
    
    # "twilight_sky_1": "Relit with gentle blue-hour twilight after sunset, casting smooth diffused shadows and enveloping the subject in soft desaturated blue-gray tones to create a quiet, serene mood.", # TODO: think this later
    # "morning_sunlight_1": "Relit with soft early-morning sunlight in a fresh outdoor setting, casting gentle short shadows and surrounding the subject in pale yellow-white tones to create a clean, lightweight mood.", # TODO: think this later

    "candlelight_1": "Relit with warm candlelight in a dimly lit indoor setting, casting soft, flickering shadows and enveloping the subject in golden-orange tones to create a cozy, nostalgic mood.",
    "spotlight_1": "Relit with a concentrated bright beam in an indoor stage scene, casting smooth directional shadows and fully lighting the subject to create a sharp, center-highlighted mood.",
    "neon_streetlight_1": "Relit with vibrant neon streetlights in a lively outdoor setting, casting colorful pink and blue reflections and surrounding the subject with soft glowing edges to create a modern, cyberpunk mood.",
    "dappled_sunlight_1": "Relit with dappled sunlight softened by humid air, casting diffused warm blotches of light and surrounding the subject in hazy golden tones to create a gentle, atmospheric mood.",
    
    # "dusk_backlit_2": "Relit with dusk backlighting just before sunset, placing the subject in deep shadow against a softly illuminated blue-gradient evening sky.",
    # "dusk_backlit_3": "Relit with strong backlighting at dusk, rendering the subject mostly as a silhouette against a cool blue twilight sky with a faint horizon glow.",
    # "dusk_backlit_4": "Relit with dramatic dusk backlighting after sunset, casting the subject into a dark silhouette while the sky fades from pale blue to deep indigo.",
    # "dusk_backlit_5":"Relit with extreme dusk backlighting, fully silhouetting the subject against a fading blue-to-black twilight sky.",

    # "twilight_sky_2": "Relit with early evening twilight shortly after sunset, casting very soft low-contrast shadows and surrounding the subject in muted cool-gray tones to create a calm, transitional mood.",
    # "twilight_sky_3": "Relit with gentle blue-hour twilight after sunset, casting smooth diffused shadows and enveloping the subject in soft desaturated blue-gray tones to create a quiet, serene mood.",
    # "twilight_sky_4": "Relit with late twilight as the sky darkens, casting subtle directional shadows and surrounding the subject in cool slate-blue tones to create a subdued, contemplative mood.",
    # "twilight_sky_5": "Relit with hazy twilight under thin clouds after sunset, casting softly glowing diffused shadows and enveloping the subject in pale lavender-gray tones to create a gentle, dreamy mood.",

    # "morning_sunlight_2": "Relit with soft early-morning sunlight in a fresh outdoor setting, casting gentle short shadows and surrounding the subject in pale yellow-white tones to create a clean, lightweight mood.",
    # "morning_sunlight_3": "Relit with cool early-morning sunlight after dawn, casting crisp bluish-tinted shadows and enveloping the subject in clear cool-white tones to create a brisk, awakened mood.",
    # "morning_sunlight_4": "Relit with bright rising-morning sunlight in an open outdoor scene, casting clear forward shadows and surrounding the subject in neutral pale-yellow light to create a vibrant, energetic mood.",
    # "morning_sunlight_5": "Relit with hazy morning sunlight through thin mist, casting diffused glowing shadows and enveloping the subject in soft creamy-white tones to create a gentle, dreamy mood.",
    # "morning_sunlight_6": "Relit with angled low-morning sunlight just above the horizon, casting long cool-edged shadows and surrounding the subject in fresh pale-gold highlights to create a dynamic, crisp mood.",
    # "morning_sunlight_7": "Relit with filtered morning sunlight through light foliage, casting delicate dappled cool-warm patches and surrounding the subject in soft neutral tones to create a natural, refreshing mood."

    # "dappled_sunlight_2": "Relit with gentle dappled sunlight from early-morning foliage, casting soft rounded patches of light and enveloping the subject in warm subdued tones to create a calm, natural mood.",
    # "dappled_sunlight_3": "Relit with dappled sunlight under sparse foliage, casting broad warm patches of light and surrounding the subject in softly shifting golden tones to create an airy, textured mood.",
    # "dappled_sunlight_4": "Relit with dappled sunlight through dense summer leaves, casting small bright flecks of light and enveloping the subject in rich warm tones to create a vivid, patterned mood.",
    # "dappled_sunlight_5": "Relit with dappled sunlight filtered by gently moving branches, casting dynamic drifting highlights and bathing the subject in warm natural tones to create an animated, lively mood.",
    # "dappled_sunlight_7": "Relit with dappled sunlight from low afternoon sun, casting angled warm fragments of light and enveloping the subject in glowing orange tones to create a dramatic, textured mood.",
    # "dappled_sunlight_8": "Relit with dappled sunlight softened by humid air, casting diffused warm blotches of light and surrounding the subject in hazy golden tones to create a gentle, atmospheric mood.",

    # "overcast_daylight_2": "Relit with diffused overcast daylight in a quiet outdoor setting, casting subtle soft shadows and enveloping the subject in cool grayish light to create a smooth, balanced mood.",
    # "overcast_daylight_3": "Relit with diffused overcast daylight in a quiet outdoor setting, casting gentle muted shadows and surrounding the subject in soft grayish tones to create a calm, balanced mood.",
    # "overcast_daylight_4": "Relit with diffused overcast daylight in a quiet outdoor setting, casting faint natural shadows and enveloping the subject in cool neutral light to create a smooth, understated mood.",
    # "overcast_daylight_5": "Relit with diffused overcast daylight in a quiet outdoor setting, casting subtle even shadows and bathing the subject in cool muted tones to create a gentle, natural mood.",
    # "overcast_daylight_6": "Relit with diffused overcast daylight in a quiet outdoor setting, casting soft uniform shadows and enveloping the subject in light gray illumination to create a smooth, tranquil mood.",
    # "overcast_daylight_7": "Relit with diffused overcast daylight in a quiet outdoor setting, casting barely visible shadows and enveloping the subject in cool gentle tones to create a clean, balanced mood.",

    # "window_light_2": "Relit with soft morning window light in a calm indoor setting, casting gentle angled shadows and surrounding the subject in cool pale highlights to create a fresh, peaceful mood.",
    # "window_light_3": "Relit with warm afternoon window light in a cozy indoor setting, casting smooth directional shadows and enveloping the subject in soft golden tones to create a relaxed, inviting mood.",
    # "window_light_4": "Relit with bright midday window light in a clean indoor setting, casting crisp natural shadows and wrapping the subject in clear white highlights to create a vivid, balanced mood.",
    # # "window_light_5": "Relit with diffused cloudy window light in a muted indoor setting, casting faint soft shadows and surrounding the subject in gentle grayish tones to create a quiet, subdued mood.",
    # "window_light_6": "Relit with sharp low-angle window light in a minimalist indoor setting, casting long defined shadows and bathing the subject in warm orange highlights to create a dramatic, intimate mood.",
    # "window_light_7": "Relit with filtered curtain window light in a delicate indoor setting, casting soft patterned shadows and enveloping the subject in pale diffused glow to create a tender, dreamy mood.",
    # "window_light_8": "Relit with cool twilight window light in a dim indoor setting, casting subtle soft shadows and surrounding the subject in gentle bluish tones to create a calm, contemplative mood.",
    # "window_light_9": "Relit with bright snowy window light in a serene indoor setting, casting crisp diffused shadows and bathing the subject in clean cool highlights to create a pure, tranquil mood.",

    # "moonlight_2": "Relit with cool moonlight in a calm outdoor night setting, casting soft directional shadows and surrounding the subject in faint bluish tones to create a serene, contemplative mood.",
    # "moonlight_3": "Relit with pale moonlight under a clear night sky, casting subtle elongated shadows and enveloping the subject in gentle silvery tones to create a quiet, dreamlike mood.",
    # "moonlight_4": "Relit with cold moonlight in a minimalist nighttime scene, casting crisp soft shadows and bathing the subject in icy blue highlights to create a tranquil, distant mood.",
    # "moonlight_5": "Relit with soft lunar glow through thin clouds, casting diffused shadows and wrapping the subject in misty bluish-gray tones to create a calm, ethereal mood.",
    # "moonlight_6": "Relit with moonlight reflected off water in a nocturnal setting, casting shimmering highlights and surrounding the subject in cool silver tones to create a poetic, reflective mood.",
    # "moonlight_7": "Relit with high moonlight above open terrain, casting sharp angled shadows and illuminating the subject with pale spectral light to create a cinematic, mysterious mood.",
    # "moonlight_8": "Relit with moonlight filtered through forest leaves, casting delicate dappled shadows and surrounding the subject in muted twilight tones to create a natural, intimate mood.",
    # "moonlight_9": "Relit with intense full moon glow on a clear night, casting deep contrasting shadows and enveloping the subject in luminous icy-blue light to create a dramatic, haunting mood."

    # "foggy_2": "Relit with dense fog in a muted outdoor setting, casting soft diffused shadows and surrounding the subject in pale gray light to create a quiet, atmospheric mood.",
    # "foggy_3": "Relit with gentle fog in an open outdoor scene, blurring distant details and enveloping the subject in cool misty tones to create a calm, subdued mood.",
    # "foggy_4": "Relit with low-hanging fog in a still outdoor setting, softening edges and surrounding the subject in flat white light to create a muted, tranquil mood.",
    # "foggy_5": "Relit with drifting fog in a quiet exterior space, reducing contrast and covering the subject in pale neutral tones to create a soft, contemplative mood.",
    # "foggy_6": "Relit with light fog in a natural outdoor environment, smoothing highlights and bathing the subject in diffuse gray light to create a relaxed, dreamy mood.",
    # "foggy_7": "Relit with cool fog in a calm outdoor scene, fading background clarity and surrounding the subject with subdued gray haze to create an understated, immersive mood.",
    # "foggy_8": "Relit with heavy mist in a muted outdoor landscape, flattening shadows and enveloping the subject in soft diffused light to create a serene, atmospheric mood.",
    # "foggy_9": "Relit with rolling fog in a quiet open setting, dimming distant shapes and surrounding the subject in soft pale tones to create a peaceful, hazy mood."

    # "spotlight_2": "Relit with a bright centered spotlight in an indoor stage setting, casting focused contour shadows and lighting the subject with strong front highlights to create a vivid, performance-like mood.",
    # "spotlight_3": "Relit with a concentrated studio spotlight in an indoor setting, casting crisp directional shadows and isolating the subject with clear front illumination to create a bold, stage-focused mood.",
    # "spotlight_4": "Relit with a high-impact stage spotlight in a controlled indoor scene, casting defined contour shadows and brightly highlighting the subject to create a powerful, center-stage mood.",
    # "spotlight_5": "Relit with a focused performance spotlight in an indoor venue, casting clean directional shadows and strongly illuminating the subject to create a sharp, concert-style mood.",
    # "spotlight_6": "Relit with a single-beam center spotlight in an indoor stage setting, casting gentle contour shadows and bright front lighting to keep the subject clearly visible in a dramatic, show-like mood."

    # "spotlight_7": "Relit with a bright center-beam spotlight in an indoor stage setting, casting soft contour shadows and fully illuminating the subject's face and body to create a vivid, show-like mood.",
    # "spotlight_8": "Relit with a strong front-facing spotlight in a performance-style indoor scene, casting clean directional shadows and lighting the subject with full, even highlights to create a clear, center-stage mood.",
    # "spotlight_9": "Relit with an intense centered spotlight in an indoor venue, casting natural contour shadows and brightly revealing the subject to create a bold, high-visibility stage mood.",
    # "spotlight_10": "Relit with a bright focused beam in an indoor stage environment, casting gentle shaping shadows and lighting the subject with strong frontal illumination to create a polished, spotlight-centered mood.",
    # "spotlight_11": "Relit with a powerful front spotlight in an indoor concert-style setting, casting subtle contour shadows and clearly lighting the subject's features to create a crisp, high-impact mood.",
    # "spotlight_12": "Relit with a concentrated bright beam in an indoor stage scene, casting smooth directional shadows and fully lighting the subject to create a sharp, center-highlighted mood.",

}


# Register available prompt versions
relighting_prompt_versions = {
    "1": relighting_prompts, # default one. no need to specify in config
    "2": relighting_prompts_2,
    "3": relighting_prompts_3,
    "4": relighting_prompts_4, 
    "5": relighting_prompts_5,
    "6": relighting_prompts_6,
    # Add future versions like "4": relighting_prompts_4 here
}


def concat_images_side_by_side(image1, image2):
    """
    Concatenates two images side-by-side, resizing both to the height of image2,
    
    Args:
        image1 (PIL.Image): The first image (left side).
        image2 (PIL.Image): The second image (right side).
    Returns:
        PIL.Image: The concatenated image
    """
    # Resize both images to the height of image2, preserving aspect ratio
    target_height = image2.height
    
    # Resize image1
    aspect_ratio1 = image1.width / image1.height
    new_width1 = int(target_height * aspect_ratio1)
    image1 = image1.resize((new_width1, target_height))
    
    # Create a new blank image with combined width
    concatenated_image = Image.new('RGB', (image1.width + image2.width, target_height))
    
    # Paste the images side-by-side
    concatenated_image.paste(image1, (0, 0))
    concatenated_image.paste(image2, (image1.width, 0))
    
    return concatenated_image


def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="configs/base.yaml", help="Path to the base configuration file")
    parser.add_argument("--exp_config", type=str, required=True, help="Path to the experiment-specific configuration file")
    parser.add_argument("--relight_type", type=str, required=True, help="Specify relighting type")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID to use")
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--seed_offset', type=int, default=0, help='Starting seed value (default is 0)')

    # ✅ ADD THIS (dataset version selector)
    parser.add_argument("--dataset_tag", type=str, default="", help="controls skip_list version and (optionally) output folder naming")

    return parser.parse_args()

def load_config(args):
    """Load and merge configuration files and inject CLI arguments."""
    base_cfg = OmegaConf.load(args.base_config)
    exp_cfg = OmegaConf.load(args.exp_config)
    config = OmegaConf.merge(base_cfg, exp_cfg)
    config.relight_type = args.relight_type
    config.gpu = args.gpu
    config.output_dir = os.path.splitext(os.path.basename(args.exp_config))[0]
    config.dataset_tag = args.dataset_tag
    print(OmegaConf.to_yaml(config))
    return config

def load_depth_map(subfolder_path, config, relight_id):
    # Only allow outpaint depth mode
    if "outpaint" not in str(config.depth_mode):
        raise ValueError(
            f"depth_mode={config.depth_mode} not supported. "
            "Only modes containing 'outpaint' are allowed."
        )

    # subfolder_path can be:
    #  - flat: "00000_00" (already a stem)
    #  - spreeai: "/.../dataset/Adidas_R2_Men_Jackets_216" (folder path)
    #  - spreeai: "Adidas_R2_Men_Jackets_216" (folder name)
    stem = os.path.basename(subfolder_path.rstrip("/"))

    outpaint_folder = os.path.join("outpaint", config.output_dir, relight_id, stem)

    if config.relight_image_only:
        relight_path = os.path.join(outpaint_folder, "depth_relight.png")
        return Image.open(relight_path).convert("RGB")
    else:
        base_path = os.path.join(outpaint_folder, "depth_base.png")
        relight_path = os.path.join(outpaint_folder, "depth_relight.png")

        base = Image.open(base_path).convert("RGB")
        relight = Image.open(relight_path).convert("RGB")
        assert base.size == relight.size, "Depth maps must have the same size!"
        return Image.fromarray(np.hstack([np.array(base), np.array(relight)]))

def extract_background(prompt: str) -> str:
    # Find the index where "background" starts, ignoring case.
    index = prompt.lower().find("background")
    if index != -1:
        return prompt[index:]
    return ""

def extract_foreground(prompt: str) -> str:
    index = prompt.lower().find("background")
    if index != -1:
        return prompt[:index].strip(",. ")
    return prompt.strip(",. ")


def process_depth_map(input_path, output_path, depth_model, use_v2=False):
    """
    Process an image to generate its depth map using the provided depth model.
    
    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the generated depth map.
        depth_model: A pre-loaded depth estimation model (pipeline or processor).
        use_v2 (bool): Flag indicating if the new pipeline is used.
    """
    image = Image.open(input_path).convert("RGB")
    
    if use_v2:
        depth_map = depth_model(image)["depth"]
    else:
        control_image = load_image(input_path)
        depth_map = depth_model(control_image)[0].convert("RGB")
    
    depth_map.save(output_path)
    print(f"Saved depth map to {output_path}")


def process_body_mask(input_path, black_mask_output_path):
    """
    Convert an RGBA image to a black foreground mask.
    Foreground will be black (0), background white (255).

    Args:
        input_path (str): Path to the input RGBA body mask.
        black_mask_output_path (str): Path to save the black foreground mask.

    Returns:
        str: Path to the saved black foreground mask.
    """
    img = Image.open(input_path).convert("RGBA")
    alpha_channel = np.array(img)[:, :, 3]

    black_fg_mask = np.where(alpha_channel > 0, 0, 255).astype(np.uint8)
    Image.fromarray(black_fg_mask).save(black_mask_output_path)

    print(f"Saved black foreground mask: {black_mask_output_path}")
    return black_mask_output_path

def prepare_canvas_and_mask(image, target_width, target_height, apply_fg_mask=False, body_mask=None, crop_to_foreground=False, upper_crop=False):
    '''
    Resizes and centers an image on a fixed-size canvas with black padding, optionally creating a matching mask.
    If crop_to_foreground is True and a body_mask is provided, the image is tightly cropped to the foreground.
    '''

    # --- Crop to foreground if enabled ---
    if crop_to_foreground and body_mask is not None:
        # Invert, since the foreground is black
        inverted_mask = ImageOps.invert(body_mask)
        bbox = inverted_mask.getbbox()
        if bbox is not None:
            left, upper, right, lower = bbox

            if upper_crop:
                # keep only top 50% height
                mid = upper + (lower - upper) // 2
                lower = mid

            image = image.crop((left, upper, right, lower))
            body_mask = body_mask.crop((left, upper, right, lower))

    # --- Resize image and paste to canvas ---
    orig_w, orig_h = image.size
    scale = min(target_width / orig_w, target_height / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    canvas = Image.new("RGB", (target_width, target_height), color=(0, 0, 0))
    canvas.paste(image.resize((new_w, new_h), Image.LANCZOS), (x_offset, y_offset))

    # --- Create the mask ---
    if apply_fg_mask and body_mask is not None:
        body_mask_resized = body_mask.resize((new_w, new_h), Image.NEAREST)
        mask_canvas = Image.new("L", (target_width, target_height), color=255)
        mask_canvas.paste(body_mask_resized, (x_offset, y_offset))
    else:
        mask_canvas = Image.new("L", (target_width, target_height), color=255)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask_canvas)
        draw.rectangle((x_offset, y_offset, x_offset + new_w, y_offset + new_h), fill=0)

    return canvas, mask_canvas, scale, x_offset, y_offset


def resize_mask_to_canvas(mask, target_width, target_height):
    # Get original size
    orig_w, orig_h = mask.size
    # Compute scale preserving aspect ratio
    # scale = min(target_width / orig_w, target_height / orig_h, 1.0)
    scale = min(target_width / orig_w, target_height / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    # Resize the mask using NEAREST (to preserve binary values)
    resized_mask = mask.resize((new_w, new_h), Image.NEAREST)
    # Create a blank canvas of target size (fill with white, assuming white is background)
    canvas = Image.new("L", (target_width, target_height), color=255)
    # Center the resized mask onto the canvas
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas.paste(resized_mask, (x_offset, y_offset))
    return canvas


def resolve_flat_paths(config, stem):
    root = config.input_dir
    image_dir = os.path.join(root, "image")
    caption_dir = os.path.join(root, "caption")
    mask_dir = os.path.join(root, "fg_masks")

    img_candidates = [
        os.path.join(image_dir, f"{stem}.jpg"),
        os.path.join(image_dir, f"{stem}.jpeg"),
        os.path.join(image_dir, f"{stem}.png"),
    ]
    source_image_path = next((p for p in img_candidates if os.path.exists(p)), None)

    annotation_path = os.path.join(caption_dir, f"{stem}.txt")

    mask_candidates = [
        os.path.join(mask_dir, f"{stem}.png"),
        os.path.join(mask_dir, f"{stem}.jpg"),
        os.path.join(mask_dir, f"{stem}.jpeg"),
    ]
    black_mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)

    return source_image_path, annotation_path, black_mask_path