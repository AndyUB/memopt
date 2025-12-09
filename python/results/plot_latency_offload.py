#!/usr/bin/env python3
"""
Visualize latency benchmarks comparing offload vs no-offload for different model sizes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Find the most recent latency benchmark CSV
csv_files = glob.glob("latency_benchmark_*.csv")
if not csv_files:
    print("No latency benchmark CSV files found!")
    exit(1)

latest_csv = max(csv_files, key=os.path.getctime)
print(f"Loading data from: {latest_csv}")

# Load data
df = pd.read_csv(latest_csv)

# Create single figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.suptitle('Latency Breakdown: CPU Offload vs No Offload', fontsize=14, fontweight='bold')

# Model names for x-axis - create pairs for each model (no offload, with offload)
models = df['model_name'].tolist()
num_models = len(models)

# Create x positions for grouped bars
x_positions = []
x_labels = []
for i, model in enumerate(models):
    x_positions.append(i * 3)      # No offload
    x_positions.append(i * 3 + 1)  # With offload
    x_labels.append('No\nOffload')
    x_labels.append('With\nOffload')

x = np.array(x_positions)

# Prepare data for stacked bars
forward_no = df['forward_no_offload_sec'].tolist()
backward_no = df['backward_no_offload_sec'].tolist()
optimizer_no = df['optimizer_no_offload_sec'].tolist()

forward_with = df['forward_with_offload_sec'].tolist()
backward_with = df['backward_with_offload_sec'].tolist()
optimizer_with = df['optimizer_with_offload_sec'].tolist()

# Interleave no-offload and with-offload data
forward_data = []
backward_data = []
optimizer_data = []
for i in range(num_models):
    forward_data.append(forward_no[i])
    forward_data.append(forward_with[i])
    backward_data.append(backward_no[i])
    backward_data.append(backward_with[i])
    optimizer_data.append(optimizer_no[i])
    optimizer_data.append(optimizer_with[i])

width = 0.8

# Create stacked bars
p1 = ax.bar(x, forward_data, width, label='Forward', color='#2ecc71', alpha=0.85)
p2 = ax.bar(x, backward_data, width, bottom=forward_data,
           label='Backward', color='#f39c12', alpha=0.85)
bottom_optimizer = [f + b for f, b in zip(forward_data, backward_data)]
p3 = ax.bar(x, optimizer_data, width, bottom=bottom_optimizer,
           label='Optimizer', color='#9b59b6', alpha=0.85)

# Add total time labels on top of each bar
totals = [f + b + o for f, b, o in zip(forward_data, backward_data, optimizer_data)]
for i, (pos, total) in enumerate(zip(x, totals)):
    ax.text(pos, total, f'{total:.3f}s',
           ha='center', va='bottom', fontsize=8, fontweight='bold')

# Add model name separators and labels
for i, model in enumerate(models):
    mid_x = i * 3 + 0.5
    ax.text(mid_x, -0.05 * max(totals), model,
           ha='center', va='top', fontsize=11, fontweight='bold',
           transform=ax.get_xaxis_transform())
    
    # Add vertical separator line after each model (except the last)
    if i < num_models - 1:
        ax.axvline(x=i * 3 + 2, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Adjust y-axis to make room for model labels
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min - 0.05 * y_max, y_max * 1.1)

plt.tight_layout()

# Save figure
output_file = 'latency_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("LATENCY SUMMARY")
print("="*60)
for idx, row in df.iterrows():
    print(f"\n{row['model_name']}:")
    print(f"  Total latency (no offload):   {row['total_no_offload_sec']:.4f} sec")
    print(f"  Total latency (with offload): {row['total_with_offload_sec']:.4f} sec")
    if row['overall_speedup'] >= 1:
        print(f"  Speedup:                      {row['overall_speedup']:.2f}x faster")
    else:
        print(f"  Slowdown:                     {1/row['overall_speedup']:.2f}x slower")
    print(f"  Forward speedup:              {row['forward_speedup']:.2f}x")
    print(f"  Backward speedup:             {row['backward_speedup']:.2f}x")
    print(f"  Optimizer speedup:            {row['optimizer_speedup']:.2f}x")
