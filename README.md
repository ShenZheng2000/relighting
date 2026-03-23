# Introduction
We consider two tasks: **human relighting** and **driving scene relighting**, which share a similar generation pipeline but differ in how depth is obtained.

**Flux Outpainting**

For human relighting ONLY, we use `FLUX.1-Fill-dev` to outpaint each source image twice: once with a base prompt (from dataset annotations or generated prompts) and once with a relighting prompt. This expands the image to the target resolution (e.g., 784×784) and introduces rich background context.

**Depth Estimation**

For human relighting, depth maps are computed from the outpainted images.

For driving scenes, depth is directly estimated from the input image without outpainting.

**Flux 2x1 Generation**

We use `FLUX.1-Depth-dev` with depth as control input to generate a 2×1 image grid. The left image corresponds to the base lighting, and the right image shows the same subject or scene under the target relighting condition, while preserving structure and content.

**(Optional) GPT Image Filtering**

We use the ChatGPT API to filter out low-quality images before training. 


# Run FLUX to Generate Base–Relit Image Pairs

## 1. Environment Setup

```
conda create -n flux_diffusers python=3.10 -y
conda activate flux_diffusers
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -U diffusers
pip install git+https://github.com/asomoza/image_gen_aux.git
```

## 2. Download Dataset

For our human relighting data generation pipeline, you only need 100 images with GPT-4o generated caption and Grounded SAM2-generated fg masks from VITON-HD. Please download them from [here](https://drive.google.com/drive/folders/1LIVq0SKuvAoJwTvFzFoHXrt-bgaxAFoI?usp=drive_link).

Similarly, for roadwork, please download the images used in the data generation pipeline from [here](https://drive.google.com/file/d/1vpwWWIM7QSqYe3LveqC1DnRt2FNoJvNI/view?usp=drive_link).


## 3. Specify Relighting Prompts

Edit `utils.py` to define or modify relighting prompts.

For example:

```
relighting_prompts_6 = {
    "noon_sunlight_1": "Relit with bright noon sunlight in a clear outdoor setting, casting soft natural shadows and surrounding the subject in crisp white light to create a clean, vibrant daytime mood.",
    "golden_sunlight_1": "Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood.",
    "foggy_1": "Relit with dense fog in a muted outdoor setting, casting soft diffused shadows and surrounding the subject in pale gray light to create a quiet, atmospheric mood.",
    "moonlight_1": "Relit with cold moonlight in a minimalist nighttime scene, casting crisp soft shadows and bathing the subject in icy blue highlights to create a tranquil, distant mood.",
    # add more if needed
}
```


## 4. Prepare Dataset

In the YAML config file, set `input_dir` to your dataset path.

Expected dataset structure:
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


## 5. Run inference & Prepare train-test splits

See `inf.sh` for example commands.

Make sure to specify `inference_mode` in the YAML config file:
- `inference_mode: human` → outpainting + T2I
- `inference_mode: driving` → T2I


### Example Dataset Structure
```
/home/shenzhen/Datasets/relighting/exp_1_10_1/golden_sunlight_1
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
<summary><strong> (Optional) Filter Out Bad Images Using GPT API</strong></summary>

Install the OpenAI client: 
```
pip install openai
```

Edit the script: `shen_scripts/gpt_api_decide.py`

* Set your API key: 
    ```
    client = openai.OpenAI(api_key="xxx")
    ```

* Set your root directory and relight type. For example: 
    ```
    root_dir = "/home/shenzhen/Relight_Projects/relighting/outputs/exp_1_10_1_seed0"
    relight_type = "golden_sunlight_1"
    ```

Run the script. For each subfolder with images, a corresponding `invalid.txt` will be generated listing the filtered-out images.
</details>