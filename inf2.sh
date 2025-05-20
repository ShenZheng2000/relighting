python inference.py --exp_config configs/exp_4_13.yaml --relight_type candlelight_1 --gpu 0 --seed_offset 0 --num_seeds 2 &
python inference.py --exp_config configs/exp_4_13.yaml --relight_type candlelight_1 --gpu 1 --seed_offset 2 --num_seeds 2 &
wait