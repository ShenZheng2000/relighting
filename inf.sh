# # Use flux to generate 2x1 images
# python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_9.yaml --relight_type noon_sunlight_1 --gpu 4 --seed_offset 60 --num_seeds 10

# # break into img2img-turbo format
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_9.yaml --relight_type noon_sunlight_1 --gpu 0

# # warp dataset to enlarge faces
# python warp_dataset.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_9.yaml --relight_type candlelight_1 --gpu 0