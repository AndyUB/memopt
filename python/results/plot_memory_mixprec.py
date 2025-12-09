#!/usr/bin/env python3
"""
Visualize memory benchmarks comparing FP32 vs BF16 mixed precision for different model sizes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Find the most recent memory mixprec benchmark CSV
csv_files = glob.glob("memory_mixprec_benchmark_*.csv")
if not csv_files:
    print("No memory mixprec benchmark CSV files found!")
    exit(1)

latest_csv = max(csv_files, key=os.path.getctime)
print(f"Loading data from: {latest_csv}")

# Load data
df = pd.read_csv(latest_csv)

# Create single figure
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.suptitle('Peak Memory Usage: FP32 vs Mixed Precision Configurations', fontsize=14, fontweight='bold')

# Model names for x-axis
models = df['model_name'].tolist()
num_models = len(models)

# ====================
# Peak memory comparison for 4 configurations
# ====================

x = np.arange(num_models)
width = 0.2  # Width of each bar

# Peak memory for each configuration
peak_fp32 = df['peak_mem_fp32_gb'].tolist()
peak_autocast = df['peak_mem_autocast_gb'].tolist()
peak_no_master = df['peak_mem_no_master_gb'].tolist()
peak_default = df['peak_mem_default_gb'].tolist()

bars1 = ax.bar(x - 1.5*width, peak_fp32, width, label='FP32', color='#e74c3c', alpha=0.85)
bars2 = ax.bar(x - 0.5*width, peak_autocast, width, label='FP32-FP16 (autocast, master param)', color='#3498db', alpha=0.85)
bars3 = ax.bar(x + 1.5*width, peak_default, width, label='FP32-FP16 (no autocast, master param)', color='#2ecc71', alpha=0.85)
bars4 = ax.bar(x + 0.5*width, peak_no_master, width, label='FP16', color='#f39c12', alpha=0.85)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Peak Memory (GB)', fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# Adjust y-axis
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min, y_max * 1.1)

plt.tight_layout()

# Save figure
output_file = 'memory_mixprec_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("MEMORY SUMMARY: FP32 vs Mixed Precision Configurations")
print("="*60)
for idx, row in df.iterrows():
    print(f"\n{row['model_name']}:")
    print(f"  Peak memory FP32:        {row['peak_mem_fp32_gb']:.3f} GB")
    print(f"  Peak memory autocast:    {row['peak_mem_autocast_gb']:.3f} GB ({(1 - row['peak_mem_autocast_gb']/row['peak_mem_fp32_gb'])*100:.1f}% saved)")
    print(f"  Peak memory no_master:   {row['peak_mem_no_master_gb']:.3f} GB ({(1 - row['peak_mem_no_master_gb']/row['peak_mem_fp32_gb'])*100:.1f}% saved)")
    print(f"  Peak memory default:     {row['peak_mem_default_gb']:.3f} GB ({(1 - row['peak_mem_default_gb']/row['peak_mem_fp32_gb'])*100:.1f}% saved)")
