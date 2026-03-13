import os
import io
import sys
import base64
from pathlib import Path

import openai
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import relighting_prompts_6

client = openai.OpenAI()

# ----------------------------
# Config
# ----------------------------
# exp_1_1_warped_128_eyes
# exp_1_10_1
# exp_1_10_1_warped_128_eyes

# golden_sunlight_1
# noon_sunlight_1
# foggy_1
# moonlight_1


relight_type = "golden_sunlight_1" 
source_dir = "/home/shenzhen/Datasets/VITON/test_sample_100/image"

# model_type = "exp_1_1_warped_128_eyes"
# result_suffix = "_warp_relight_unwarp"  # set to None if you want flexible matching
# result_dir = f"/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/{model_type}/{relight_type}/VITON/test_sample_100/image"

result_suffix = None
result_dir = f"/home/shenzhen/Relight_Projects/IC-Light/outputs/512x512/{relight_type}/VITON/test/image"

model_name = "gpt-4o"
prompt_description = relighting_prompts_6[relight_type]

valid_exts = (".png", ".jpg", ".jpeg")


def encode_image_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def query_chatgpt_with_two_images(src_b64, tgt_b64, description_prompt):
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are given two images:\n"
                            "- first image: source/original image\n"
                            "- second image: generated relit result\n\n"
                            "Instructions:\n"
                            "1. Does the second image match the following lighting description?\n"
                            f"\"{description_prompt}\"\n"
                            "(Yes/No)\n"
                            "2. Does the second image contain exactly one person?\n"
                            "(Yes/No)\n"
                            "3. Same person as the first image?\n"
                            "(Yes/No)\n"
                            "4. Same clothing as the first image?\n"
                            "(Yes/No)\n"
                            "5. Same pose as the first image?\n"
                            "(Yes/No)\n\n"
                            "**FINAL ANSWER:** Yes (only if all five answers are Yes). Otherwise write No."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{src_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{tgt_b64}"},
                    },
                ],
            }
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content


def find_result_file_for_source(source_stem, result_files, suffix=None):
    candidates = []

    for fname in result_files:
        stem = Path(fname).stem

        if suffix is not None:
            if stem == source_stem + suffix:
                candidates.append(fname)
        else:
            if stem == source_stem or stem.startswith(source_stem + "_"):
                candidates.append(fname)

    if len(candidates) == 0:
        return None

    if len(candidates) == 1:
        return candidates[0]

    print(f"⚠️ Multiple matches for {source_stem}: {candidates}")
    return sorted(candidates)[0]


def evaluate_pairs(source_dir, result_dir, result_suffix=None):
    passed = 0
    failed = 0
    missing = 0
    unreadable = 0

    fail_q1 = 0
    fail_q2 = 0
    fail_q3 = 0
    fail_q4 = 0
    fail_q5 = 0

    missing_examples = []
    unreadable_examples = []

    source_files = sorted(
        [f for f in os.listdir(source_dir) if f.lower().endswith(valid_exts)]
    )
    result_files = sorted(
        [f for f in os.listdir(result_dir) if f.lower().endswith(valid_exts)]
    )

    if not source_files:
        print(f"No source images found in: {source_dir}")
        return

    if not result_files:
        print(f"No result images found in: {result_dir}")
        return

    for src_fname in source_files:
        src_path = os.path.join(source_dir, src_fname)
        src_stem = Path(src_fname).stem

        tgt_fname = find_result_file_for_source(src_stem, result_files, result_suffix)

        if tgt_fname is None:
            print("====================================================")
            print(f"[SOURCE] {src_fname}")
            print("⚠️ No matched result image found")
            missing += 1
            missing_examples.append(src_fname)
            continue

        tgt_path = os.path.join(result_dir, tgt_fname)

        try:
            src_img = Image.open(src_path).convert("RGB")
            tgt_img = Image.open(tgt_path).convert("RGB")
        except Exception as e:
            print("====================================================")
            print(f"[SOURCE] {src_fname}")
            print(f"[RESULT] {tgt_fname}")
            print(f"⚠️ Failed to open image: {e}")
            unreadable += 1
            unreadable_examples.append((src_fname, tgt_fname, str(e)))
            continue

        src_b64 = encode_image_base64(src_img)
        tgt_b64 = encode_image_base64(tgt_img)

        result = query_chatgpt_with_two_images(src_b64, tgt_b64, prompt_description)

        result_lower = result.lower()

        if "unable to view" in result_lower or "can't view" in result_lower:
            unreadable += 1

        elif "yes" in result_lower.split("final answer")[-1]:
            passed += 1

        else:
            failed += 1

            print("====================================================")
            print(f"[SOURCE] {src_fname}")
            print(f"[RESULT] {tgt_fname}")
            print(result)

            lines = result_lower.splitlines()

            if len(lines) >= 5:
                if "no" in lines[0]: fail_q1 += 1
                if "no" in lines[1]: fail_q2 += 1
                if "no" in lines[2]: fail_q3 += 1
                if "no" in lines[3]: fail_q4 += 1
                if "no" in lines[4]: fail_q5 += 1


    total = len(source_files)

    print("\n================ FINAL SUMMARY ================")
    print(f"Total source images : {total}")
    print(f"Passed              : {passed}")
    print(f"Rejected            : {failed}")
    print(f"Missing result      : {missing}")
    print(f"Unreadable          : {unreadable}")
    print(f"Pass rate           : {passed / max(1, total):.2%}")

    print("\nSuccess rate breakdown:")
    print(f"Lighting match      : {(total - fail_q1) / max(1, total):.2%}")
    print(f"Single person       : {(total - fail_q2) / max(1, total):.2%}")
    print(f"Same person         : {(total - fail_q3) / max(1, total):.2%}")
    print(f"Same clothes        : {(total - fail_q4) / max(1, total):.2%}")
    print(f"Same pose           : {(total - fail_q5) / max(1, total):.2%}")

    if missing_examples:
        print("\n---------------- Missing Results ----------------")
        for src_fname in missing_examples:
            print(src_fname)

    if unreadable_examples:
        print("\n---------------- Unreadable Examples ----------------")
        for item in unreadable_examples:
            print(item)


if __name__ == "__main__":
    evaluate_pairs(source_dir, result_dir, result_suffix)