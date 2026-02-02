# NOTE: bad script, skip for now! 
# import os
# import io
# import base64
# from PIL import Image
# import openai

# # --------- CONFIG ----------
# MODEL_NAME = "gpt-4o"
# IMG_GLOB_DIR = "/home/shenzhen/Datasets/VITON/train_debug_100/image"
# OUT_CAP_DIR  = "/home/shenzhen/Datasets/VITON/train_debug_100/caption"
# TEMPERATURE = 0
# MAX_TOKENS = 200

# TEMPLATE = (
#     'Describe the person in the image using this exact format: '
#     'Woman/Man, <pose>, wearing <top description>, paired with <bottom description if any>, '
#     '<accessories if any>, <hair>, <expression if visible>, background of <scene and lighting>. '
# )

# # --------- OPENAI CLIENT ----------
# client = openai.OpenAI()

# def encode_image_base64_pil(img: Image.Image) -> str:
#     buf = io.BytesIO()
#     img.save(buf, format="JPEG", quality=95)
#     return base64.b64encode(buf.getvalue()).decode("utf-8")

# def caption_one_image(img_path: str) -> str:
#     img = Image.open(img_path).convert("RGB")
#     img_b64 = encode_image_base64_pil(img)

#     resp = client.chat.completions.create(
#         model=MODEL_NAME,
#         temperature=TEMPERATURE,
#         max_tokens=MAX_TOKENS,
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": TEMPLATE},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
#                 ],
#             }
#         ],
#     )
#     return resp.choices[0].message.content.strip()

# def main():
#     os.makedirs(OUT_CAP_DIR, exist_ok=True)

#     exts = (".jpg", ".jpeg", ".png", ".webp")
#     img_files = [f for f in os.listdir(IMG_GLOB_DIR) if f.lower().endswith(exts)]
#     img_files = sorted(img_files)  # VSCode-like order

#     for fname in img_files:
#         in_path = os.path.join(IMG_GLOB_DIR, fname)

#         # caption filename: same stem, .txt
#         stem, _ = os.path.splitext(fname)
#         out_path = os.path.join(OUT_CAP_DIR, f"{stem}.txt")

#         try:
#             cap = caption_one_image(in_path)
#         except Exception as e:
#             cap = f"ERROR: {e}"

#         print(f"[{fname}] {cap}")
#         with open(out_path, "w", encoding="utf-8") as f:
#             f.write(cap + "\n")

#         # NOTE: use this for debug now, comment later
#         exit()

# if __name__ == "__main__":
#     main()