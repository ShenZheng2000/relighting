

# Introduction

**Flux Outpainting**  
We use the FLUX Fill pipeline to outpaint each source image twice: once using the base prompt (from dataset annotations) and once using a relighting prompt (e.g., golden_sunlight_1). This reshapes the image to the desired resolution and adds rich background context for the relit version.

**Depth Estimation**  
We then run depth estimation on both outpainted images to obtain depth maps for the base and relit views.

**Flux 2x1 Generation**

Next, we use the FLUX Control pipeline with the paired depth maps as control input. Using a 2×1 grid prompt, we generate two side-by-side images: the left shows the person under base lighting, and the right shows the same person with the same pose and clothing under the relighting prompt.

**(Optional) GPT Image Filtering**

We use the ChatGPT API to filter out low-quality images before training.

**Image Warping on Detected Face Regions**

We detect facial and eye regions and apply saliency-guided warping to enlarge them, increasing effective resolution and enabling finer details to be represented and reconstructed in the latent space.

**Image-to-Image Model Training**  
Finally, we train an image-to-image translation model (e.g., Pix2Pix-Turbo) on the synthesized pairs to distill the relighting behavior into a lightweight, fast-inference network.


# Run FLUX to Generate Base–Relit Image Pairs


## 1. Install Environment

```
cd $HOME && git clone https://github.com/black-forest-labs/flux
cd $HOME/flux
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## 2. Run Grounded SAM 2 to get body mask

Clone this repo: https://github.com/ShenZheng2000/Grounded-SAM-2

Install the env based on the instruction, setup path, and run `run.py`, setting `--input-dir` to be the input dataset folder like `dataset_with_garment`.


## 3. Specify relighting prompt
Edit `utils.py` and modify or add entries in `relighting_prompt_?`. 

For example:

```
relighting_prompts_6 = {
    "noon_sunlight_1": "Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood."
    "golden_sunlight_1": "Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood.",
    # add more as needed.
}
```


## 4. Prepare dataset
In `configs/base.yaml`, set `input_dir` to your dataset path. 
Example:
```
input_dir: /home/shenzhen/Datasets/dataset_with_garment_bigface_100
```

You can use `shen_scripts/bigface_100.txt` to get these 100 images

Expected dataset folder structure (LEGACY / SpreeAI-style):
```
dataset_with_garment_bigface_100/
├── 8seconds_men_shirts_034/
│   ├── pre_processing/
│       ├── black_fg_mask_groundedsam2.png
│   ├── bdy_2.jpg
│   └── gar_0.jpg
│   └── gpt_annotation__bdy_2.txt
├── 09WOMEN_WOMEN_BLOUSE_167/
├── 09WOMEN_WOMEN_PANTS_418/
├── Adidas_R2_Men_Jackets_216/
└── Adsb_Women_Skirts_008/
└── ...
```

Expected dataset folder structure (NEW / flat-style):
```
$dataset_name/
├── caption/
│   ├── 00000_00.txt
├── fg_masks/
│   ├── 00000_00.png
├── image/
│   ├── 00000_00.jpg
└── ...
```


## 5. Prepare Config
In your experiment config (e.g., `configs/exp_10_16.yaml`):
* Set `prompt_version` (default: 6)
* Set `max_images` (2 for quick debugging, null for full experiments)
* Set `upper_crop` (default: true. recommended for warping experiments: false)


## 6. Run inference 

For example, to run relighting using `golden_sunlight_1` across 10 GPUs, each running 1 seeds:

Run in terminal (LEGACY / SpreeAI-style):
```
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type golden_sunlight_1 --gpu 0 --seed_offset 0 --num_seeds 1
```

OR, Run in terminal (NEW / flat-style):
```
python inference_spreeai.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type golden_sunlight_1 --gpu 0 --seed_offset 0 --num_seeds 1
```

All images will be saved in `outputs/`



## 7. Prepare Dataset in Pix2Pix-Turbo's Format


Run the following command in the terminal 

NOTE: same config as above, but please update `root_dir` and `output_dir` in `prepare_data.py`
```
python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type golden_sunlight_1 --gpu 0
```

The script above will:
* Format images into the structure required by Pix2Pix-Turbo
* Automatically split the dataset into train/test sets
* Skip any images listed in `invalid.txt` or `skip_list` (you can modify or remove these files if needed)

### Example Dataset Structure
```
/home/shenzhen/Datasets/relighting/exp_10_16/golden_sunlight_1
├── train_A
│ ├── 0.png
│ ├── 1.png
│ └── ...
├── train_B
│ ├── 0.png
│ ├── 1.png
│ └── ...
├── test_A
│ ├── 0.png
│ ├── 1.png
│ └── ...
├── test_B
│ ├── 0.png
│ ├── 1.png
│ └── ...
├── train_prompts.json
└── test_prompts.json
```



<details>
<summary><strong> (Optional, ) Filter out bad images using GPT-API</strong></summary>

Install the OpenAI client: 
```
pip install openai
```

Edit the script: `shen_scripts/gpt_api_decide.py`

* Set your API key: 
    ```
    client = openai.OpenAI(api_key="xxx")
    ```

* Set your root directory. For example: 
    ```
    root_dir = "/home/shenzhen/Relight_Projects/relighting/outputs"
    ```

Run the script. For each subfolder with images, a corresponding `invalid.txt` will be generated listing the filtered-out images.
</details>





# Train Pix2Pix-Turbo with Synthesized Images


## 1. Setup Repo and Install Env
```
git clone https://github.com/ShenZheng2000/img2img-turbo
cd img2img-turbo
conda env create -f environment.yaml
conda activate img2img-turbo

pip install huggingface_hub==0.25.0
pip install peft==0.10.0
pip install wandb
pip install vision_aided_loss

pip install insightface==0.7.3
pip install opencv-python pillow
pip install numpy==1.26.4
pip install onnxruntime-gpu==1.17.1
```


## 2. Image Warping on Detected Face Regions

Run the following commands inside the `img2img-turbo` repository.

NOTE: A bandwidth (`--bw`) of 128 is recommended, and using `--include-eyes` improves face and eye detail quality.

Run in terminal:
```
python warp_dataset.py \
    --input_root /home/shenzhen/Datasets/relighting \
    --target_prefix exp_10_16 \
    --relight_type golden_sunlight_1 \
    --bw 128 \
    --include-eyes
```

### Example Dataset Structure (Warped Images)
```
/home/shenzhen/Datasets/relighting/exp_10_16_warped_128/golden_sunlight_1
├── train_A
│ ├── 0.png
│ ├── 0.inv.pth
│ ├── 1.png
│ ├── 1.inv.pth
│ └── ...
├── train_B
│ ├── 0.png
│ ├── 0.inv.pth
│ ├── 1.png
│ ├── 1.inv.pth
│ └── ...
├── test_A
│ ├── 0.png
│ ├── 0.inv.pth
│ ├── 1.png
│ ├── 1.inv.pth
│ └── ...
├── test_B
│ ├── 0.png
│ ├── 0.inv.pth
│ ├── 1.png
│ ├── 1.inv.pth
│ └── ...
├── train_prompts.json
└── test_prompts.json
```


## 3. Model Training 

For training details, see the [official guide](https://github.com/GaParmar/img2img-turbo/blob/main/docs/training_pix2pix_turbo.md)

Example training with 4 GPUs, each at least having 48GB of memory

Run in terminal
```
bash run.sh
```

## 4. Model Testing

Example testing with 1 GPU
```
bash test2.sh
```