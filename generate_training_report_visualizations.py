#!/usr/bin/env python3
"""
Generate comprehensive visualizations for training report

Creates publication-quality figures for scientific paper
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for publication
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# Load training history
exp_dir = Path('experiments_v3/exp_v3_20260131_172406')
history = pd.read_csv(exp_dir / 'training_history.csv')

# Create output directory
output_dir = Path('training_report_figures')
output_dir.mkdir(exist_ok=True)

print("🎨 Generating training report visualizations...")

# Figure 1: Loss Curves
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history['epoch'] + 1, history['train_loss'], 'o-', label='Training Loss', linewidth=2, markersize=6)
ax.plot(history['epoch'] + 1, history['val_loss'], 's-', label='Validation Loss', linewidth=2, markersize=6)
ax.axvline(x=7, color='gray', linestyle='--', alpha=0.5, label='112→168 pixels')
ax.axvline(x=14, color='gray', linestyle=':', alpha=0.5, label='168→224 pixels')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'fig1_loss_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 1: Loss curves saved")


# Figure 2: Magnitude F1 Score
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history['epoch'] + 1, history['train_magnitude_f1'], 'o-', 
        label='Training Magnitude F1', linewidth=2, markersize=6, color='#2ecc71')
ax.plot(history['epoch'] + 1, history['val_magnitude_f1'], 's-', 
        label='Validation Magnitude F1', linewidth=2, markersize=6, color='#e74c3c')
ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Target (0.85)')
ax.axvline(x=13, color='gold', linestyle='--', alpha=0.7, label='Best Model (Epoch 13)')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Magnitude Classification F1 Score', fontsize=14, fontweight='bold')
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(output_dir / 'fig2_magnitude_f1.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 2: Magnitude F1 score saved")

# Figure 3: Azimuth F1 Score
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history['epoch'] + 1, history['train_azimuth_f1'], 'o-', 
        label='Training Azimuth F1', linewidth=2, markersize=6, color='#3498db')
ax.plot(history['epoch'] + 1, history['val_azimuth_f1'], 's-', 
        label='Validation Azimuth F1', linewidth=2, markersize=6, color='#e67e22')
ax.axhline(y=0.70, color='green', linestyle='--', alpha=0.5, label='Target (0.70)')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Azimuth Classification F1 Score (Challenge)', fontsize=14, fontweight='bold')
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 0.1])
plt.tight_layout()
plt.savefig(output_dir / 'fig3_azimuth_f1.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 3: Azimuth F1 score saved")


# Figure 4: Class Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Magnitude distribution
magnitude_data = {
    'Medium': 1036,
    'Normal': 888,
    'Large': 28,
    'Moderate': 20
}
colors_mag = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
ax1.bar(magnitude_data.keys(), magnitude_data.values(), color=colors_mag, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax1.set_title('Magnitude Class Distribution', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
for i, (k, v) in enumerate(magnitude_data.items()):
    ax1.text(i, v + 30, f'{v}\n({v/1972*100:.1f}%)', ha='center', fontsize=10, fontweight='bold')

# Azimuth distribution
azimuth_data = {
    'Normal': 888,
    'N': 480,
    'S': 168,
    'NW': 104,
    'W': 104,
    'SW': 96,
    'SE': 84,
    'E': 44,
    'NE': 4
}
colors_az = plt.cm.tab10(np.linspace(0, 1, len(azimuth_data)))
ax2.bar(azimuth_data.keys(), azimuth_data.values(), color=colors_az, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax2.set_title('Azimuth Class Distribution', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', rotation=45)
for i, (k, v) in enumerate(azimuth_data.items()):
    if v > 50:
        ax2.text(i, v + 20, f'{v}', ha='center', fontsize=9, fontweight='bold')
    else:
        ax2.text(i, v + 5, f'{v}', ha='center', fontsize=9, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig(output_dir / 'fig4_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 4: Class distribution saved")


# Figure 5: Training Progress Summary
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Loss comparison
ax1.plot(history['epoch'] + 1, history['train_loss'], 'o-', label='Train', linewidth=2)
ax1.plot(history['epoch'] + 1, history['val_loss'], 's-', label='Validation', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Magnitude F1
ax2.plot(history['epoch'] + 1, history['train_magnitude_f1'], 'o-', label='Train', linewidth=2)
ax2.plot(history['epoch'] + 1, history['val_magnitude_f1'], 's-', label='Validation', linewidth=2)
ax2.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='Target')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('F1 Score')
ax2.set_title('Magnitude F1 Score')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Azimuth F1
ax3.plot(history['epoch'] + 1, history['train_azimuth_f1'], 'o-', label='Train', linewidth=2)
ax3.plot(history['epoch'] + 1, history['val_azimuth_f1'], 's-', label='Validation', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('F1 Score')
ax3.set_title('Azimuth F1 Score')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Performance gap
train_val_gap = history['train_magnitude_f1'] - history['val_magnitude_f1']
ax4.plot(history['epoch'] + 1, train_val_gap, 'o-', linewidth=2, color='red')
ax4.axhline(y=0, color='green', linestyle='--', alpha=0.5)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('F1 Gap (Train - Val)')
ax4.set_title('Magnitude F1 Gap (Overfitting Indicator)')
ax4.grid(True, alpha=0.3)
ax4.fill_between(history['epoch'] + 1, 0, train_val_gap, alpha=0.3, color='red')

plt.tight_layout()
plt.savefig(output_dir / 'fig5_training_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 5: Training summary saved")


# Figure 6: Class Imbalance Visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Combine all classes
all_classes = list(magnitude_data.keys()) + [f"Az-{k}" for k in azimuth_data.keys() if k != 'Normal']
all_counts = list(magnitude_data.values()) + [v for k, v in azimuth_data.items() if k != 'Normal']

# Sort by count
sorted_data = sorted(zip(all_classes, all_counts), key=lambda x: x[1], reverse=True)
classes, counts = zip(*sorted_data)

# Color code by severity
colors = []
for count in counts:
    if count > 500:
        colors.append('#2ecc71')  # Green - sufficient
    elif count > 100:
        colors.append('#f39c12')  # Orange - moderate
    elif count > 20:
        colors.append('#e74c3c')  # Red - insufficient
    else:
        colors.append('#8e44ad')  # Purple - critical

bars = ax.barh(classes, counts, color=colors, edgecolor='black', linewidth=1)
ax.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_title('Class Imbalance Analysis (All Classes)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (cls, count) in enumerate(zip(classes, counts)):
    ax.text(count + 20, i, f'{count}', va='center', fontsize=9, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='Sufficient (>500)'),
    Patch(facecolor='#f39c12', label='Moderate (100-500)'),
    Patch(facecolor='#e74c3c', label='Insufficient (20-100)'),
    Patch(facecolor='#8e44ad', label='Critical (<20)')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig(output_dir / 'fig6_class_imbalance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Figure 6: Class imbalance analysis saved")

print("\n" + "="*60)
print("🎉 All visualizations generated successfully!")
print("="*60)
print(f"\n📁 Output directory: {output_dir}")
print("\nGenerated figures:")
print("  1. fig1_loss_curves.png - Training and validation loss")
print("  2. fig2_magnitude_f1.png - Magnitude F1 score progression")
print("  3. fig3_azimuth_f1.png - Azimuth F1 score progression")
print("  4. fig4_class_distribution.png - Dataset class distribution")
print("  5. fig5_training_summary.png - Comprehensive training summary")
print("  6. fig6_class_imbalance.png - Class imbalance analysis")
print("\n✅ Ready for inclusion in scientific paper!")
