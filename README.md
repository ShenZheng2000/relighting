

# Introduction

**Flux Outpainting**  
We use the FLUX Fill pipeline to outpaint each source image twice: once using the base prompt (from dataset annotations) and once using a relighting prompt (e.g., candlelight). This reshapes the image to the desired resolution and adds rich background context for the relit version.

**Depth Estimation**  
We then run depth estimation on both outpainted images to obtain depth maps for the base and relit views.

**Flux 2x1 Generation**

Next, we use the FLUX Control pipeline with the paired depth maps as control input. Using a 2×1 grid prompt, we generate two side-by-side images: the left shows the person under base lighting, and the right shows the same person with the same pose and clothing under the relighting prompt.

**(Optional) GPT Image Filtering**

We use the ChatGPT API to filter out low-quality images before training.

**Image Warping on Detected Face Regions**

We detect face regions and apply saliency-guided warping to enlarge them, which increases the effective resolution and allows finer facial details to be represented and reconstructed in the latent space.

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
    "candlelight_1": "Relit with warm candlelight in a dimly lit indoor setting, casting soft, flickering shadows and enveloping the subject in golden-orange tones to create a cozy, nostalgic mood."
    "noon_sunlight_1": "Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood."
    # add more as needed.
}
```


## 4. Prepare dataset
In `configs/base.yaml`, set `input_dir` to your dataset path. 
Example:
```
input_dir: /home/shenzhen/Datasets/dataset_with_garment_debug_100
```

Randomly sampled 100 images folder name in `shen_scripts/debug_100.txt`

Expected dataset folder structure:
```
dataset_with_garment_debug_100/
├── 8seconds_men_shirts_034/
│   ├── pre_processing/
│   ├── bdy_2.jpg
│   └── gar_0.jpg
│   └── gpt_annotation__bdy_2.txt
├── 09WOMEN_WOMEN_BLOUSE_167/
├── 09WOMEN_WOMEN_PANTS_418/
├── Adidas_R2_Men_Jackets_216/
└── Adsb_Women_Skirts_008/
└── ...
```

Alteratively, randomly sample 1000 images using `shen_scripts/random_sample.py` to get `dataset_with_garment_debug_1000`

Or the 100 big face images in  `shen_scripts/bigface_100.txt`



## 5. Prepare Config
In your experiment config (e.g., `configs/exp_10_16.yaml`):
* Set `prompt_version` (default: 6)
* Set `max_images` (2 for quick debugging, null for full experiments)


## 6. Run inference 

For example, to run relighting using `candlelight_1` across 10 GPUs, each running 1 seeds:

Run in terminal
```
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type candlelight_1 --gpu 0 --seed_offset 0 --num_seeds 1 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type candlelight_1 --gpu 1 --seed_offset 1 --num_seeds 1 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type candlelight_1 --gpu 2 --seed_offset 2 --num_seeds 1 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type candlelight_1 --gpu 3 --seed_offset 3 --num_seeds 1 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type candlelight_1 --gpu 4 --seed_offset 4 --num_seeds 1 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type candlelight_1 --gpu 5 --seed_offset 5 --num_seeds 1 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type candlelight_1 --gpu 6 --seed_offset 6 --num_seeds 1 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type candlelight_1 --gpu 7 --seed_offset 7 --num_seeds 1 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type candlelight_1 --gpu 8 --seed_offset 8 --num_seeds 1 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type candlelight_1 --gpu 9 --seed_offset 9 --num_seeds 1 &
wait
```

All images will be saved in `outputs/`

<details>
<summary><strong> (Optional) Filter out bad images using GPT-API</strong></summary>

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

## 1. Prepare Dataset in Pix2Pix-Turbo's Format

For example, to prepare the dataset in Pix2Pix-Turbo's format:

Go to the relighting repo. 

Run in terminal
```
python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_9.yaml --relight_type candlelight_1 --gpu 0
```

The script will:
* Format images into the structure required by Pix2Pix-Turbo
* Automatically split into train/test
* Skip any images listed in `invalid.txt` or `skip_list`

### Example Dataset Structure
```
/data/candlelight_1
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




## 2. Setup Repo and Install Env
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


## 3. Image Warping on Detected Face Regions

Stay in img2img-turbo repo. (NOTE: 128 is the recommended bandwidth)

Run in terminal:
```
python warp_dataset.py --target_prefix exp_10_16 --relight_type noon_sunlight_1 --bw 128
```

### Example Dataset Structure (Warped Images)
```
/data/candlelight_1_warped_{bandwidth_scale}
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


## 4. Model Training 

For training details, see the [official guide](https://github.com/GaParmar/img2img-turbo/blob/main/docs/training_pix2pix_turbo.md)

Example training with 4 GPUs, each at least having 48GB of memory

Run in terminal
```
bash run.sh
```

## 5. Model Testing

Example testing with 1 GPU
```
bash test2.sh
```