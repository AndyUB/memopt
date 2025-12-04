#!/usr/bin/env python3
"""
Visualize memory benchmarks comparing offload vs no-offload for different model sizes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Find the most recent memory benchmark CSV
csv_files = glob.glob("memory_benchmark_*.csv")
if not csv_files:
    print("No memory benchmark CSV files found!")
    exit(1)

latest_csv = max(csv_files, key=os.path.getctime)
print(f"Loading data from: {latest_csv}")

# Load data
df = pd.read_csv(latest_csv)

# Create single figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.suptitle('Peak Allocated GPU Memory: CPU Offload vs No Offload', fontsize=14, fontweight='bold')

# Model names for x-axis
models = df['model_name'].tolist()
x = np.arange(len(models))
width = 0.35

# Peak Allocated Memory Comparison
bars1 = ax.bar(x - width/2, df['peak_allocated_no_offload_gb'], width,
               label='No Offload', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, df['peak_allocated_with_offload_gb'], width,
               label='With Offload', color='#3498db', alpha=0.8)

ax.set_ylabel('Memory (GB)', fontsize=12, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right', fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}GB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()

# Save figure
output_file = 'memory_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_file}")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("MEMORY SUMMARY")
print("="*60)
for idx, row in df.iterrows():
    print(f"\n{row['model_name']}:")
    print(f"  Peak allocated (no offload):   {row['peak_allocated_no_offload_gb']:.2f} GB")
    print(f"  Peak allocated (with offload): {row['peak_allocated_with_offload_gb']:.2f} GB")
    print(f"  Memory saved:                  {row['memory_saved_allocated_gb']:.2f} GB ({row['memory_reduction_allocated_pct']:.1f}%)")
    print(f"  Peak reserved (no offload):    {row['peak_reserved_no_offload_gb']:.2f} GB")
    print(f"  Peak reserved (with offload):  {row['peak_reserved_with_offload_gb']:.2f} GB")
    print(f"  Reserved saved:                {row['memory_saved_reserved_gb']:.2f} GB ({row['memory_reduction_reserved_pct']:.1f}%)")
