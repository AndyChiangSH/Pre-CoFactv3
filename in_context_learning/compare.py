import json

# Load benchmark.json
with open('data.json', 'r') as benchmark_file:
    benchmark_data = json.load(benchmark_file)

# Load val.json
with open('train.json', 'r') as val_file:
    val_data = json.load(val_file)[:len(benchmark_data)]

# Extract values associated with the "label" key from val.json
val_labels = [item.get('label') for item in val_data]

# Count instances where val_label is "Neutral"
num_val_support = val_labels.count("Support")

# Count instances where val_label and benchmark_data are both "Neutral"
num_both_support = sum(1 for val, benchmark in zip(val_labels, benchmark_data) if val == benchmark == "Support")

# Count instances where val_label is "Neutral"
num_val_neutral = val_labels.count("Neutral")

# Count instances where val_label and benchmark_data are both "Neutral"
num_both_neutral = sum(1 for val, benchmark in zip(val_labels, benchmark_data) if val == benchmark == "Neutral")

# Count instances where val_label is "Refute"
num_val_refute = val_labels.count("Refute")

# Count instances where val_label and benchmark_data are both "Refute"
num_both_refute = sum(1 for val, benchmark in zip(val_labels, benchmark_data) if val == benchmark == "Refute")

print(f"Number of instances where val_label is 'Support': {num_val_support}")
print(f"Number of instances where val_label and benchmark_data are both 'Support': {num_both_support}")

print(f"Number of instances where val_label is 'Neutral': {num_val_neutral}")
print(f"Number of instances where val_label and benchmark_data are both 'Neutral': {num_both_neutral}")

print(f"Number of instances where val_label is 'Refute': {num_val_refute}")
print(f"Number of instances where val_label and benchmark_data are both 'Refute': {num_both_refute}")
