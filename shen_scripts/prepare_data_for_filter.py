import os
import re
import shutil
from pathlib import Path
from tqdm import tqdm

SRC_ROOT = Path("/home/shenzhen/Relight_Projects/relighting/outputs")
DST_ROOT = Path("/home/shenzhen/Datasets/relighting/exp_10_16_flat")

EXP_PREFIX = "exp_10_16_seed"
DO_SYMLINK = False

RELIGHT_TYPES = {
    "golden_sunlight_1",
    "noon_sunlight_1",
    "foggy_1",
    "moonlight_1",
}

seed_re = re.compile(rf"^{re.escape(EXP_PREFIX)}(\d+)$")
DST_ROOT.mkdir(parents=True, exist_ok=True)

seed_dirs = [d for d in SRC_ROOT.iterdir() if d.is_dir() and seed_re.match(d.name)]

for seed_dir in tqdm(sorted(seed_dirs), desc="Seeds"):
    seed = int(seed_re.match(seed_dir.name).group(1))
    seed_tag = f"seed{seed:02d}"

    for relight_dir in sorted(seed_dir.iterdir()):
        if not relight_dir.is_dir():
            continue

        relight_type = relight_dir.name
        if relight_type not in RELIGHT_TYPES:
            continue

        out_dir = DST_ROOT / relight_type
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs = list(relight_dir.glob("*.png"))
        for img_path in tqdm(imgs, desc=f"{relight_type}/{seed_tag}", leave=False):
            dst_path = out_dir / f"{seed_tag}-{img_path.name}"
            if dst_path.exists():
                continue
            shutil.copy2(img_path, dst_path)

print("Done.")