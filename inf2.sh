python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type noon_sunlight_1 --gpu 4 --seed_offset 0  --num_seeds 13 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type noon_sunlight_1 --gpu 5 --seed_offset 13 --num_seeds 13 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type noon_sunlight_1 --gpu 6 --seed_offset 26 --num_seeds 12 &
python inference.py --base_config configs/base_10_2.yaml --exp_config configs/exp_10_16.yaml --relight_type noon_sunlight_1 --gpu 7 --seed_offset 38 --num_seeds 12 &
wait