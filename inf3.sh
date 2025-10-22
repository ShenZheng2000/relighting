python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type noon_sunlight_1 --gpu 0 --seed_offset 50 --num_seeds 13 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type noon_sunlight_1 --gpu 1 --seed_offset 63 --num_seeds 13 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type noon_sunlight_1 --gpu 2 --seed_offset 76 --num_seeds 12 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type noon_sunlight_1 --gpu 3 --seed_offset 88 --num_seeds 12 &
wait