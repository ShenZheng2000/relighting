import os
import sys
import webbrowser
import subprocess

# NOTE: use ctrl + shift + r to refresh the browser!!!!!!!!!!!!!!!!!!!!!!!!!!

exp_num = sys.argv[1]
folder = f"/home/shenzhen/Relight_Projects/relighting-comparison/outputs/exp_4_13_v{exp_num}/golden_hour_back_1"
output_html = os.path.join(folder, "index.html")

# Generate HTML
images = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
with open(output_html, "w") as f:
    f.write("<html><head><title>Image Viewer</title></head><body>\n")
    for img in images:
        f.write(f"<div style='margin-bottom:20px;'>\n")
        f.write(f"<p style='font-size:18px; font-weight:bold;'>{img}</p>\n")
        f.write(f"<img src='{img}' style='width:512px; height:auto;'><br>\n")
        f.write("</div>\n")
    f.write("</body></html>")

print(f"HTML saved to: {output_html}")

# Kill port 8000 if in use
os.system("fuser -k 8000/tcp")

# Start server in background
subprocess.Popen(["python3", "-m", "http.server", "8000"], cwd=folder)

# Open in browser
webbrowser.open("http://localhost:8000/index.html")