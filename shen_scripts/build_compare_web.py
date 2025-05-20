import os

'''
cd /scratch1/shenzhen/img2img-turbo
python3 -m http.server 8080
'''

# --- Config ---
base_dir = "/scratch1/shenzhen/img2img-turbo"
relight_type = "candlelight_1_may4"

base_folder = f"{base_dir}/data/{relight_type}/test_A"
relight_folder = f"{base_dir}/output/pix2pix_turbo/{relight_type}/eval/fid_13301"
output_html = os.path.join(base_dir, f"{relight_type}.html")

# --- Relative paths (from HTML to image folders) ---
rel_base = os.path.relpath(base_folder, base_dir)
rel_relight = os.path.relpath(relight_folder, base_dir)

# --- Available relight image indices ---
available_indices = sorted(
    int(f.split("_")[1].split(".")[0])
    for f in os.listdir(relight_folder)
    if f.startswith("val_") and f.endswith(".png")
)

# --- Write HTML ---
with open(output_html, "w") as f:
    f.write("<html><head><title>Side-by-Side Comparison</title></head><body>\n")
    f.write("<table border='1' cellspacing='10'>\n")
    f.write("<tr><th>Base Image</th><th>Relight Image</th></tr>\n")

    for i in available_indices:
        base_img = f"{rel_base}/{i}.png"
        relight_img = f"{rel_relight}/val_{i}.png"
        f.write(f"<tr><td style='text-align:center'><img src='{base_img}' width='256'></td><td style='text-align:center'><img src='{relight_img}' width='256'></td></tr>\n")

    f.write("</table>\n</body></html>\n")
print(f"HTML file created at: {output_html}")