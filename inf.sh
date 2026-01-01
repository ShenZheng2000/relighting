# # Use flux to generate 2x1 images
# python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type golden_sunlight_1 --gpu 4 --seed_offset 0 --num_seeds 25 &
# python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type golden_sunlight_1 --gpu 5 --seed_offset 25 --num_seeds 25 &
# python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type golden_sunlight_1 --gpu 6 --seed_offset 50 --num_seeds 25 &
# python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml  --relight_type golden_sunlight_1 --gpu 7 --seed_offset 75 --num_seeds 25 &
# wait

# # convert into img2img-turbo format
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type dusk_backlit_1 --gpu 0
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_12_7.yaml --relight_type dusk_backlit_1 --gpu 1

# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type foggy_1 --gpu 1 --dataset_tag v2
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type moonlight_1 --gpu 2 --dataset_tag v2
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type dusk_backlit_1 --gpu 3 --dataset_tag v2
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type golden_sunlight_1 --gpu 3 --dataset_tag v2

# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_12_7.yaml --relight_type foggy_1 --gpu 4 --dataset_tag v2
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_12_7.yaml --relight_type moonlight_1 --gpu 5 --dataset_tag v2
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_12_7.yaml --relight_type dusk_backlit_1 --gpu 6 --dataset_tag v2
# python prepare_data.py --base_config configs/base_10_2.yaml --exp_config configs/exp_12_7.yaml --relight_type golden_sunlight_1 --gpu 7 --dataset_tag v2