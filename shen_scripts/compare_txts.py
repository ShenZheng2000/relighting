# a_txt = "/home/shenzhen/Relight_Projects/relighting/outputs/exp_10_16_seed0/moonlight_1/gpt-4.1_invalid.txt"
# a_txt = "/home/shenzhen/Relight_Projects/relighting/outputs/exp_10_16_seed0/moonlight_1/gpt-4o_invalid.txt"
a_txt = "/home/shenzhen/Relight_Projects/relighting/outputs/exp_10_16_seed0/moonlight_1/gpt-4.1-mini_invalid.txt"
b_txt = "/home/shenzhen/Datasets/relighting/exp_10_16_flat/moonlight_1_bad_toy.txt"

a = set(line.strip() for line in open(a_txt) if line.strip())
b = set(line.strip() for line in open(b_txt) if line.strip())

only_a = a - b   # GPT flagged but you didn't
only_b = b - a   # You flagged but GPT didn't
both = a & b     # Agreement

print("GPT invalid count:", len(a))
print("Your manual bad count:", len(b))
print("Agreement (GPT ∩ You):", len(both))
print("GPT-only (GPT \\ You):", len(only_a))
print("You-only (You \\ GPT):", len(only_b))