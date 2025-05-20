import os
import openai
from PIL import Image
import io
import base64

# Set your OpenAI API key
client = openai.OpenAI(api_key="xxx")

# Folder of images to evaluate
image_folder = "/home/shenzhen/Relight_Projects/relighting-comparison/outputs/exp_4_13_v0/candlelight_1"
prompt_description = (
    "Relit with warm candlelight in a dimly lit indoor setting, casting soft, flickering shadows and enveloping the subject in golden-orange tones to create a cozy, nostalgic mood."
)

 
def query_chatgpt_with_image(img_b64, description_prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,  # NOTE: for consistent results
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This is a horizontal strip of 3 images:\n"
                            "- left: original image\n"
                            "- middle: generated base image\n"
                            "- right: generated relit image\n"
                            "Instructions:\n"
                            "1. Does the **right** image match the following lighting description?\n"
                            f"\"{description_prompt}\"\n"
                            "(Yes/No)\n"
                            "2. Do both the **middle** and **right** images contain exactly one person?\n"
                            "(Yes/No)\n"
                            "3. Same person?\n"
                            "(Yes/No)\n"
                            "4. Same clothing?\n"
                            "(Yes/No)\n"
                            "5. Same pose?\n"
                            "(Yes/No)\n"
                            "**FINAL ANSWER:** Yes (only if all five answers are Yes). Otherwise, write No."
                        )
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ],
            }
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content


def encode_image_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def evaluate_all_images(folder, prompt_desc):
    # Clear the invalid.txt file if it exists
    invalid_txt_path = os.path.join(folder, "invalid.txt")
    if os.path.exists(invalid_txt_path):
        os.remove(invalid_txt_path)

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(folder, fname)
        full_img = Image.open(path).convert("RGB")
        img_b64 = encode_image_base64(full_img)
        result = query_chatgpt_with_image(img_b64, prompt_desc)

        print("====================================================")
        print(f"\n[{fname}]")
        print(result)

        # Decision filter
        if "unable to view" in result.lower():
            print("⚠️ Skipped — image not readable by model")
        elif "FINAL ANSWER: Yes" in result or "**FINAL ANSWER:** Yes" in result:
            print("✅ Good image")
        else:
            print("❌ Reject image")
            with open(invalid_txt_path, "a") as f:
                f.write(f"{fname}\n")

# Run it
evaluate_all_images(image_folder, prompt_description)