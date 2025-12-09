#!/usr/bin/env python3
"""
Visualize latency benchmarks comparing FP32 vs BF16 mixed precision for different model sizes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Find the most recent latency mixprec benchmark CSV
csv_files = glob.glob("latency_mixprec_benchmark_*.csv")
if not csv_files:
    print("No latency mixprec benchmark CSV files found!")
    exit(1)

latest_csv = max(csv_files, key=os.path.getctime)
print(f"Loading data from: {latest_csv}")

# Load data
df = pd.read_csv(latest_csv)

# Create single figure
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.suptitle('Latency: FP32 vs Mixed Precision Configurations', fontsize=14, fontweight='bold')

# Model names for x-axis
models = df['model_name'].tolist()
num_models = len(models)

x = np.arange(num_models)
width = 0.2  # Width of each bar

# Total latency for each configuration
total_fp32 = df['total_fp32_sec'].tolist()
total_autocast = df['total_autocast_sec'].tolist()
total_no_master = df['total_no_master_sec'].tolist()
total_default = df['total_default_sec'].tolist()

bars1 = ax.bar(x - 1.5*width, total_fp32, width, label='FP32', color='#e74c3c', alpha=0.85)
bars2 = ax.bar(x - 0.5*width, total_autocast, width, label='FP32-FP16 (autocast, master param)', color='#3498db', alpha=0.85)
bars3 = ax.bar(x + 1.5*width, total_default, width, label='FP32-FP16 (no autocast, master param)', color='#2ecc71', alpha=0.85)
bars4 = ax.bar(x + 0.5*width, total_no_master, width, label='FP16', color='#f39c12', alpha=0.85)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Adjust y-axis
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min, y_max * 1.1)

plt.tight_layout()

# Save figure
output_file = 'latency_mixprec_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("LATENCY SUMMARY: FP32 vs Mixed Precision Configurations")
print("="*60)
for idx, row in df.iterrows():
    print(f"\n{row['model_name']}:")
    print(f"  Total latency FP32:       {row['total_fp32_sec']:.4f} sec")
    print(f"  Total latency autocast:   {row['total_autocast_sec']:.4f} sec ({row['speedup_autocast']:.2f}x)")
    print(f"  Total latency no_master:  {row['total_no_master_sec']:.4f} sec ({row['speedup_no_master']:.2f}x)")
    print(f"  Total latency default:    {row['total_default_sec']:.4f} sec ({row['speedup_default']:.2f}x)")
