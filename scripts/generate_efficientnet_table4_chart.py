import matplotlib.pyplot as plt
import numpy as np

# Data dari EfficientNet Paper (Tan & Le, 2019) - Table 4
# ImageNet Top-1 Accuracy Comparison
models = [
    'ResNet-50',
    'ResNet-152', 
    'DenseNet-169',
    'Inception-v3',
    'Xception',
    'ResNeXt-101',
    'SENet',
    'NASNet-A',
    'AmoebaNet-A',
    'EfficientNet-B0',
    'EfficientNet-B1',
    'EfficientNet-B2',
    'EfficientNet-B3',
    'EfficientNet-B4',
    'EfficientNet-B5',
    'EfficientNet-B6',
    'EfficientNet-B7'
]

# Top-1 Accuracy (%)
accuracy = [
    76.0,  # ResNet-50
    77.8,  # ResNet-152
    76.2,  # DenseNet-169
    78.8,  # Inception-v3
    79.0,  # Xception
    80.9,  # ResNeXt-101
    82.7,  # SENet
    82.7,  # NASNet-A
    83.5,  # AmoebaNet-A
    77.1,  # EfficientNet-B0
    79.1,  # EfficientNet-B1
    80.1,  # EfficientNet-B2
    81.6,  # EfficientNet-B3
    82.9,  # EfficientNet-B4
    83.6,  # EfficientNet-B5
    84.0,  # EfficientNet-B6
    84.4   # EfficientNet-B7
]

# FLOPs (Billions)
flops = [
    4.1,   # ResNet-50
    11.3,  # ResNet-152
    3.5,   # DenseNet-169
    5.7,   # Inception-v3
    8.4,   # Xception
    32.3,  # ResNeXt-101
    42.3,  # SENet
    23.8,  # NASNet-A
    23.1,  # AmoebaNet-A
    0.39,  # EfficientNet-B0
    0.70,  # EfficientNet-B1
    1.0,   # EfficientNet-B2
    1.8,   # EfficientNet-B3
    4.2,   # EfficientNet-B4
    9.9,   # EfficientNet-B5
    19.0,  # EfficientNet-B6
    37.0   # EfficientNet-B7
]

# Parameters (Millions)
params = [
    25.6,  # ResNet-50
    60.2,  # ResNet-152
    14.1,  # DenseNet-169
    23.8,  # Inception-v3
    22.9,  # Xception
    88.8,  # ResNeXt-101
    145.8, # SENet
    88.9,  # NASNet-A
    86.7,  # AmoebaNet-A
    5.3,   # EfficientNet-B0
    7.8,   # EfficientNet-B1
    9.2,   # EfficientNet-B2
    12.0,  # EfficientNet-B3
    19.0,  # EfficientNet-B4
    30.0,  # EfficientNet-B5
    43.0,  # EfficientNet-B6
    66.0   # EfficientNet-B7
]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=150)

# Color coding
colors = ['#1f77b4'] * 9 + ['#ff7f0e'] * 8  # Blue for others, Orange for EfficientNet

# Plot 1: Accuracy Comparison
ax1 = axes[0]
bars1 = ax1.bar(range(len(models)), accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('ImageNet Top-1 Accuracy', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='80% threshold')
ax1.legend()

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, accuracy)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}',
             ha='center', va='bottom', fontsize=7, rotation=0)

# Plot 2: FLOPs Comparison (Log Scale)
ax2 = axes[1]
bars2 = ax2.bar(range(len(models)), flops, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('FLOPs (Billions)', fontsize=12, fontweight='bold')
ax2.set_title('Computational Cost (FLOPs)', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Plot 3: Parameters Comparison (Log Scale)
ax3 = axes[2]
bars3 = ax3.bar(range(len(models)), params, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
ax3.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
ax3.set_title('Model Size (Parameters)', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(models)))
ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax3.set_yscale('log')
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Add overall title
fig.suptitle('EfficientNet Performance Comparison (Table 4 - Tan & Le, 2019)', 
             fontsize=16, fontweight='bold', y=1.02)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', edgecolor='black', label='Baseline Models'),
    Patch(facecolor='#ff7f0e', edgecolor='black', label='EfficientNet Family')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=11)

plt.tight_layout()
plt.savefig('efficientnet_table4_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Grafik tersimpan: efficientnet_table4_comparison.png")

# Create second figure: Efficiency Plot (Accuracy vs FLOPs)
fig2, ax = plt.subplots(figsize=(12, 8), dpi=150)

# Plot efficiency
for i, model in enumerate(models):
    if i < 9:
        marker = 'o'
        color = '#1f77b4'
        size = 100
    else:
        marker = 's'
        color = '#ff7f0e'
        size = 150
    
    ax.scatter(flops[i], accuracy[i], s=size, marker=marker, color=color, 
               alpha=0.7, edgecolors='black', linewidth=1.5)
    ax.annotate(model, (flops[i], accuracy[i]), 
                xytext=(5, 5), textcoords='offset points', 
                fontsize=8, alpha=0.8)

ax.set_xlabel('FLOPs (Billions, Log Scale)', fontsize=13, fontweight='bold')
ax.set_ylabel('ImageNet Top-1 Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Model Efficiency: Accuracy vs Computational Cost\n(EfficientNet achieves better accuracy with fewer FLOPs)', 
             fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
           markersize=10, label='Baseline Models', markeredgecolor='black', markeredgewidth=1),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e', 
           markersize=10, label='EfficientNet Family', markeredgecolor='black', markeredgewidth=1)
]
ax.legend(handles=legend_elements, fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig('efficientnet_efficiency_plot.png', dpi=150, bbox_inches='tight')
print("✅ Grafik efisiensi tersimpan: efficientnet_efficiency_plot.png")

plt.show()
print("\n📊 Data Summary:")
print(f"Total models compared: {len(models)}")
print(f"Best accuracy: {max(accuracy):.1f}% (EfficientNet-B7)")
print(f"Most efficient: EfficientNet-B0 ({accuracy[9]:.1f}% with only {flops[9]:.2f}B FLOPs)")
