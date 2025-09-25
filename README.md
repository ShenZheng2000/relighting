

# Introduction

**Flux Outpainting**  
We use the FLUX Fill pipeline to outpaint each source image twice: once using the base prompt (from dataset annotations) and once using a relighting prompt (e.g., candlelight). This reshapes the image to the desired resolution and adds rich background context for the relit version.

**Depth Estimation**  
We then run depth estimation on both outpainted images to obtain depth maps for the base and relit views.

**Flux 2x1 Generation**

Next, we use the FLUX Control pipeline with the paired depth maps as control input. Using a 2×1 grid prompt, we generate two side-by-side images: the left shows the person under base lighting, and the right shows the same person with the same pose and clothing under the relighting prompt.

**(Optional) GPT Image Filtering**

Optionally, we use the GPT API to filter out low-quality images before training.

**Image-to-Image Model Training**  
Finally, we train an image-to-image translation model (e.g., Pix2Pix-Turbo) on the synthesized pairs to distill the relighting behavior into a lightweight, fast-inference network.


# Run FLUX to Generate Base–Relit Image Pairs


## Install Environment

```
cd $HOME && git clone https://github.com/black-forest-labs/flux
cd $HOME/flux
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```


## Specify relighting prompt
Edit `utils.py` and modify or add entries in `relighting_prompt_?`. 

For example:

```
"candlelight_1": "Relit with warm candlelight in a dimly lit indoor setting, casting soft, flickering shadows and enveloping the subject in golden-orange tones to create a cozy, nostalgic mood."
"goldenhour_1": "Relit with xxxxxxxxxxxxxxxxxxxxx"
```


## Prepare dataset
In `configs/base.yaml`, set `input_dir` to your dataset path. 
Example:
```
input_dir: /home/shenzhen/Datasets/dataset_with_garment_debug_100
```

Expected dataset folder structure:
```
dataset_with_garment_debug_100/
├── 8seconds_men_shirts_034/
│   ├── pre_processing/
│   ├── bdy_2.jpg
│   └── gar_0.jpg
├── 09WOMEN_WOMEN_BLOUSE_167/
├── 09WOMEN_WOMEN_PANTS_418/
├── Adidas_R2_Men_Jackets_216/
└── Adsb_Women_Skirts_008/
```


## Prepare Config
In your experiment config (e.g., `configs/exp_4_13.yaml`):
* Set `prompt_version` (default: 6)
* Set `max_images` (2 for quick debugging, 100 or more for full experiments)


## Run inference 

For example, to run relighting using `candlelight_1` across 2 GPUs (0 and 1), each running 2 seeds:

Run in terminal
```
python inference.py --exp_config configs/exp_4_13.yaml --relight_type candlelight_1 --gpu 0 --seed_offset 0 --num_seeds 2 &
python inference.py --exp_config configs/exp_4_13.yaml --relight_type candlelight_1 --gpu 1 --seed_offset 2 --num_seeds 2 &
wait
```

All images will be saved in `outputs/`


# (Optional) Filter out bad images using GPT-API

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



# Train Pix2Pix-Turbo with Synthesized Images

## Prepare Dataset in Correct Format

Edit: `shen_scripts/prepare_img2img_turbo_data.py`

* Set input directory (e.g.): `root_dir = "/home/shenzhen/Relight_Projects/relighting/outputs"`
* Set output directory (e.g.): `output_dir = "/home/shenzhen/Relight_Projects/img2img-turbo/data"`
* Set relight type: `relight_type = "candlelight_1"`

Run the script. It will:
* Format images into the structure required by Pix2Pix-Turbo
* Automatically split into train/test
* Skip any images listed in `invalid.txt` (if present)

## Example Dataset Structure
```
/data/candlelight_1_may4
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


## Setup Repo
```
git clone https://github.com/GaParmar/img2img-turbo
cd img2img-turbo
conda env create -f environment.yaml
conda activate img2img-turbo
```

## Install Environment
```
pip install huggingface_hub==0.25.0
pip install peft==0.10.0
pip install wandb
pip install vision_aided_loss
```

## Start Training 

Sample command:
```
accelerate launch src/train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="/home/shenzhen/Relight_Projects/img2img-turbo/output/pix2pix_turbo/candlelight_1" \
    --dataset_folder="/home/shenzhen/Relight_Projects/img2img-turbo/data/candlelight_1" \
    --resolution=512 \
    --train_batch_size=1 \
    --enable_xformers_memory_efficient_attention --viz_freq 25 \
    --track_val_fid \
    --report_to "wandb" --tracker_project_name "pix2pix_turbo_candlelight_1" \
    --train_image_prep "no_resize" \
    --test_image_prep "no_resize"
```

For more training details, see the [official guide](https://github.com/GaParmar/img2img-turbo/blob/main/docs/training_pix2pix_turbo.md)