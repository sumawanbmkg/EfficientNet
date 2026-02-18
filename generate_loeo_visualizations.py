#!/usr/bin/env python3
"""Generate LOEO validation visualizations"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
with open('loeo_validation_results/loeo_final_results.json') as f:
    results = json.load(f)

folds = [r['fold'] for r in results['per_fold_results']]
mag_accs = [r['magnitude_accuracy'] for r in results['per_fold_results']]
azi_accs = [r['azimuth_accuracy'] for r in results['per_fold_results']]

# Figure 1: Per-fold accuracy bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Magnitude
ax1 = axes[0]
bars1 = ax1.bar(folds, mag_accs, color='steelblue', alpha=0.8, edgecolor='navy')
ax1.axhline(y=results['magnitude_accuracy']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {results['magnitude_accuracy']['mean']:.2f}%")
ax1.axhline(y=94.37, color='orange', linestyle=':', linewidth=2, label='Random Split: 94.37%')
ax1.set_xlabel('Fold', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Magnitude Classification - LOEO 10-Fold', fontsize=14, fontweight='bold')
ax1.set_ylim(90, 100)
ax1.set_xticks(folds)
ax1.legend(loc='lower right')
ax1.grid(axis='y', alpha=0.3)

# Azimuth
ax2 = axes[1]
bars2 = ax2.bar(folds, azi_accs, color='forestgreen', alpha=0.8, edgecolor='darkgreen')
ax2.axhline(y=results['azimuth_accuracy']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {results['azimuth_accuracy']['mean']:.2f}%")
ax2.axhline(y=57.39, color='orange', linestyle=':', linewidth=2, label='Random Split: 57.39%')
ax2.set_xlabel('Fold', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Azimuth Classification - LOEO 10-Fold', fontsize=14, fontweight='bold')
ax2.set_ylim(50, 90)
ax2.set_xticks(folds)
ax2.legend(loc='lower right')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('loeo_validation_results/loeo_per_fold_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Comparison chart
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(2)
width = 0.35

random_split = [94.37, 57.39]
loeo = [results['magnitude_accuracy']['mean'], results['azimuth_accuracy']['mean']]
loeo_std = [results['magnitude_accuracy']['std'], results['azimuth_accuracy']['std']]

bars1 = ax.bar(x - width/2, random_split, width, label='Random Split', color='coral', alpha=0.8)
bars2 = ax.bar(x + width/2, loeo, width, yerr=loeo_std, label='LOEO (10-Fold)', color='steelblue', alpha=0.8, capsize=5)

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Validation Method Comparison: Random Split vs LOEO', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Magnitude', 'Azimuth'], fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars1, random_split):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
for bar, val, std in zip(bars2, loeo, loeo_std):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2, f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add improvement arrows
ax.annotate('', xy=(0.175, loeo[0]), xytext=(0.175, random_split[0]),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(0.35, (loeo[0] + random_split[0])/2, '+3.2%', color='green', fontsize=10, fontweight='bold')

ax.annotate('', xy=(1.175, loeo[1]), xytext=(1.175, random_split[1]),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(1.35, (loeo[1] + random_split[1])/2, '+12.1%', color='green', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('loeo_validation_results/loeo_comparison_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Box plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

ax1 = axes[0]
bp1 = ax1.boxplot([mag_accs], patch_artist=True)
bp1['boxes'][0].set_facecolor('steelblue')
bp1['boxes'][0].set_alpha(0.7)
ax1.axhline(y=94.37, color='orange', linestyle='--', linewidth=2, label='Random Split')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Magnitude - Distribution', fontsize=12, fontweight='bold')
ax1.set_xticklabels(['LOEO'])
ax1.legend()
ax1.set_ylim(90, 100)

ax2 = axes[1]
bp2 = ax2.boxplot([azi_accs], patch_artist=True)
bp2['boxes'][0].set_facecolor('forestgreen')
bp2['boxes'][0].set_alpha(0.7)
ax2.axhline(y=57.39, color='orange', linestyle='--', linewidth=2, label='Random Split')
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Azimuth - Distribution', fontsize=12, fontweight='bold')
ax2.set_xticklabels(['LOEO'])
ax2.legend()
ax2.set_ylim(50, 90)

plt.tight_layout()
plt.savefig('loeo_validation_results/loeo_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Visualizations saved:")
print("   - loeo_per_fold_accuracy.png")
print("   - loeo_comparison_chart.png")
print("   - loeo_boxplot.png")
