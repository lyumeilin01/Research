import random
import matplotlib.pyplot as plt

target_prob_list = [0.3, 0.1, 0.4, 0.2]
values = [5,10,15,20]
target_distribution_map = {v:p for (v,p) in zip(values, target_prob_list)}

origin_probs_list = [0.1, 0.7, 0.1, 0.1] # Must correspond to 'values' order
origin_distribution_map = {v:p for (v,p) in zip(values, origin_probs_list)}

# Calculate M for Rejection Sampling
M = 0
for v in values:
    if origin_distribution_map[v] == 0: # Avoid division by zero if a value has 0 proposal prob
        if target_distribution_map[v] > 0:
            print(f"Warning: Target prob for {v} is > 0 but proposal prob is 0. Cannot sample this value.")
            M = float('inf') # This makes rejection sampling impossible
            break
        else:
            continue # target is 0, proposal is 0, fine.
    ratio = target_distribution_map[v] / origin_distribution_map[v]
    if ratio > M:
        M = ratio

print(f"Calculated M: {M}")
if M == float('inf'):
    print("Cannot proceed with Rejection Sampling due to M being infinite.")
    exit()

# It's good practice to make M slightly larger for floating point reasons,
# or ensure your M is theoretically sound. Here M=4 seems exact from calculation.

accepted_samples_counts = {v:0 for v in values}
num_samples_to_generate = 10000 # How many accepted samples you want
generated_count = 0
total_proposals = 0

while generated_count < num_samples_to_generate:
    total_proposals += 1
    # 1. Sample val from the proposal distribution (origin_probs)
    val = random.choices(values, weights=origin_probs_list)[0]

    # 2. Generate a random number u ~ U(0,1)
    u = random.random()

    # 3. Acceptance condition
    # Accept if u <= p(val) / (M * q(val))
    # p(val) is target_distribution_map[val]
    # q(val) is origin_distribution_map[val]
    acceptance_threshold = target_distribution_map[val] / (M * origin_distribution_map[val])

    if u <= acceptance_threshold:
        accepted_samples_counts[val] += 1
        generated_count += 1

# Normalize to get the generated distribution
total_accepted = sum(accepted_samples_counts.values())
generated_distribution_final = {v: count / total_accepted for v, count in accepted_samples_counts.items() }

print("Target Distribution:", target_distribution_map)
print("Generated Distribution (Rejection Sampling):", generated_distribution_final)
print(f"Total proposals made: {total_proposals}")
print(f"Acceptance rate: {total_accepted / total_proposals:.4f}")

# Optional: Plotting
fig, ax = plt.subplots()
bar_width = 0.35
index = range(len(values))

bar1 = ax.bar([i - bar_width/2 for i in index], target_prob_list, bar_width, label='Target')
bar2 = ax.bar([i + bar_width/2 for i in index], [generated_distribution_final.get(v, 0) for v in values], bar_width, label='Generated')

ax.set_xlabel('Values')
ax.set_ylabel('Probability')
ax.set_title('Target vs Generated Distribution (Rejection Sampling)')
ax.set_xticks(index)
ax.set_xticklabels(values)
ax.legend()
plt.show()