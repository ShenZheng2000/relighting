# DONE: choose good seeds
# Take the best prompt to run on selected seeds with 100 images!
# Manually filter out bad images
# Train pix2pix-turbo on these good image pairs 

# NOTE: let's use golden_hour_back_1 for now (simple, but provide enough background context, and lighting instructions)
# NOTE: these are manuualy filtered "good" seeds that generate backlight images 

# # NOTE: run across 100 seeds (0-99) with candlelight_1 => all seeds looks reasonable (good quality + match text prompt)
# DOING: run inference on 100 images
for i in {0..16}; do python inference.py --exp_config configs/seeds/exp_4_13_v${i}.yaml --relight_type candlelight_1 --gpu 0; done &
for i in {17..33}; do python inference.py --exp_config configs/seeds/exp_4_13_v${i}.yaml --relight_type candlelight_1 --gpu 1; done &
for i in {34..49}; do python inference.py --exp_config configs/seeds/exp_4_13_v${i}.yaml --relight_type candlelight_1 --gpu 2; done &
for i in {50..66}; do python inference.py --exp_config configs/seeds/exp_4_13_v${i}.yaml --relight_type candlelight_1 --gpu 3; done &
for i in {67..83}; do python inference.py --exp_config configs/seeds/exp_4_13_v${i}.yaml --relight_type candlelight_1 --gpu 4; done &
for i in {84..99}; do python inference.py --exp_config configs/seeds/exp_4_13_v${i}.yaml --relight_type candlelight_1 --gpu 5; done &
wait
