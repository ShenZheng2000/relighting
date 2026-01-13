# # Use flux to generate 2x1 images
# python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type golden_sunlight_1 --gpu 4 --seed_offset 0 --num_seeds 25 &
# python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type golden_sunlight_1 --gpu 5 --seed_offset 25 --num_seeds 25 &
# python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type golden_sunlight_1 --gpu 6 --seed_offset 50 --num_seeds 25 &
# python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml  --relight_type golden_sunlight_1 --gpu 7 --seed_offset 75 --num_seeds 25 &
# wait

# # convert into img2img-turbo format
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_1_10_1.yaml --relight_type golden_sunlight_1 --gpu 0
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_1_10_1.yaml --relight_type noon_sunlight_1 --gpu 1

# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_17.yaml --relight_type foggy_1 --gpu 0
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_17.yaml --relight_type moonlight_1 --gpu 1

# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type foggy_1 --gpu 1 --dataset_tag v2
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type moonlight_1 --gpu 2 --dataset_tag v2
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type dusk_backlit_1 --gpu 3 --dataset_tag v2
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type golden_sunlight_1 --gpu 3 --dataset_tag v2

for s in $(seq 0 6); do python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_1_10_1.yaml --relight_type noon_sunlight_1 --gpu 0 --seed_offset $s --num_seeds 1; done &
for s in $(seq 7 13); do python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_1_10_1.yaml --relight_type noon_sunlight_1 --gpu 1 --seed_offset $s --num_seeds 1; done &
for s in $(seq 14 19); do python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_1_10_1.yaml --relight_type noon_sunlight_1 --gpu 2 --seed_offset $s --num_seeds 1; done &
for s in $(seq 20 25); do python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_1_10_1.yaml --relight_type noon_sunlight_1 --gpu 3 --seed_offset $s --num_seeds 1; done &
for s in $(seq 26 31); do python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_1_10_1.yaml --relight_type noon_sunlight_1 --gpu 4 --seed_offset $s --num_seeds 1; done &
for s in $(seq 32 37); do python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_1_10_1.yaml --relight_type noon_sunlight_1 --gpu 5 --seed_offset $s --num_seeds 1; done &
for s in $(seq 38 43); do python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_1_10_1.yaml --relight_type noon_sunlight_1 --gpu 6 --seed_offset $s --num_seeds 1; done &
for s in $(seq 44 49); do python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_1_10_1.yaml --relight_type noon_sunlight_1 --gpu 7 --seed_offset $s --num_seeds 1; done &
wait

# TODO: try ONE or many longer background text prompt (see utils.py!) to regenerate diverse background for VITON?

# after all these, extend to more relight prompts, and probably (later) expand the data size used for training 