# # Each GPU gets ~4–5 seeds
# python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 1  --seed_offset 5  --num_seeds 5 &
# python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 2  --seed_offset 10 --num_seeds 5 &
# python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 3  --seed_offset 22 --num_seeds 4 &
# python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 4  --seed_offset 26 --num_seeds 4 &
# python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 5  --seed_offset 30 --num_seeds 3 &
# python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 6  --seed_offset 39 --num_seeds 4 &
# python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 7  --seed_offset 43 --num_seeds 4 &
# wait

python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 0 --seed_offset 15 --num_seeds 1 &
python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 2 --seed_offset 16 --num_seeds 1 &
python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 3 --seed_offset 33 --num_seeds 1 &
python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 4 --seed_offset 50 --num_seeds 1 &
python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 5 --seed_offset 81 --num_seeds 1 &
python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 6 --seed_offset 82 --num_seeds 1 &
python inference.py --exp_config configs/exp_10_2.yaml --relight_type candlelight_1 --gpu 7 --seed_offset 83 --num_seeds 1 &
wait