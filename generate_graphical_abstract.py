#!/usr/bin/env python3
"""
Generate Graphical Abstract for journal submission.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Set up figure
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors
BLUE = '#3498db'
GREEN = '#2ecc71'
RED = '#e74c3c'
ORANGE = '#f39c12'
PURPLE = '#9b59b6'
GRAY = '#95a5a6'

# Title
ax.text(8, 7.5, 'Earthquake Precursor Detection using Deep Learning', 
        fontsize=18, fontweight='bold', ha='center', va='center')
ax.text(8, 7.0, 'A Comparative Study of VGG16 and EfficientNet-B0', 
        fontsize=14, ha='center', va='center', style='italic')

# ============================================================================
# LEFT SECTION: Input
# ============================================================================
# Magnetometer icon (simplified)
rect1 = FancyBboxPatch((0.5, 3), 2.5, 3, boxstyle="round,pad=0.1", 
                        facecolor='#ecf0f1', edgecolor='black', linewidth=2)
ax.add_patch(rect1)
ax.text(1.75, 5.5, '📡', fontsize=30, ha='center', va='center')
ax.text(1.75, 4.5, 'Geomagnetic\nData', fontsize=11, ha='center', va='center', fontweight='bold')
ax.text(1.75, 3.5, '(H, D, Z components)', fontsize=9, ha='center', va='center', color=GRAY)

# Arrow
ax.annotate('', xy=(3.5, 4.5), xytext=(3.0, 4.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Spectrogram box
rect2 = FancyBboxPatch((3.5, 3), 2.5, 3, boxstyle="round,pad=0.1", 
                        facecolor='#f8f9fa', edgecolor=ORANGE, linewidth=2)
ax.add_patch(rect2)

# Fake spectrogram
np.random.seed(42)
spec_data = np.random.rand(20, 20)
spec_data = np.cumsum(np.cumsum(spec_data, axis=0), axis=1)
ax_inset = fig.add_axes([0.26, 0.45, 0.1, 0.15])
ax_inset.imshow(spec_data, cmap='viridis', aspect='auto')
ax_inset.axis('off')
ax_inset.set_title('Spectrogram', fontsize=8)

ax.text(4.75, 3.3, 'ULF Band\n(0.001-0.01 Hz)', fontsize=8, ha='center', va='center')

# Arrow to models
ax.annotate('', xy=(6.5, 4.5), xytext=(6.0, 4.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# ============================================================================
# CENTER SECTION: Models
# ============================================================================
# VGG16 Box
rect_vgg = FancyBboxPatch((6.5, 4.5), 3, 2, boxstyle="round,pad=0.1", 
                           facecolor=BLUE, edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(rect_vgg)
ax.text(8, 6.0, 'VGG16', fontsize=14, fontweight='bold', ha='center', va='center', color='white')
ax.text(8, 5.4, '98.68% Magnitude Acc', fontsize=10, ha='center', va='center', color='white')
ax.text(8, 4.9, '528 MB | 125 ms', fontsize=9, ha='center', va='center', color='white')

# EfficientNet Box
rect_eff = FancyBboxPatch((6.5, 2), 3, 2, boxstyle="round,pad=0.1", 
                           facecolor=GREEN, edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(rect_eff)
ax.text(8, 3.5, 'EfficientNet-B0', fontsize=14, fontweight='bold', ha='center', va='center', color='white')
ax.text(8, 2.9, '94.37% Magnitude Acc', fontsize=10, ha='center', va='center', color='white')
ax.text(8, 2.4, '20 MB | 50 ms', fontsize=9, ha='center', va='center', color='white')

# Comparison annotation
ax.annotate('26× smaller\n2.5× faster', xy=(9.7, 3.5), fontsize=9, 
            ha='left', va='center', color=GREEN, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=GREEN))

# Arrows from models
ax.annotate('', xy=(10, 5.5), xytext=(9.5, 5.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.annotate('', xy=(10, 3), xytext=(9.5, 3),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# ============================================================================
# RIGHT SECTION: Output
# ============================================================================
# Prediction Results Box
rect_out = FancyBboxPatch((10, 2.5), 2.5, 4, boxstyle="round,pad=0.1", 
                           facecolor='#f8f9fa', edgecolor=PURPLE, linewidth=2)
ax.add_patch(rect_out)
ax.text(11.25, 6.0, 'Predictions', fontsize=12, fontweight='bold', ha='center', va='center')

# Magnitude output
ax.text(11.25, 5.3, 'Magnitude:', fontsize=10, ha='center', va='center', fontweight='bold')
ax.text(11.25, 4.8, 'Small | Medium | Large | Major', fontsize=8, ha='center', va='center')

# Azimuth output
ax.text(11.25, 4.1, 'Azimuth:', fontsize=10, ha='center', va='center', fontweight='bold')
ax.text(11.25, 3.6, 'N|NE|E|SE|S|SW|W|NW', fontsize=8, ha='center', va='center')

# Normal detection
ax.text(11.25, 2.9, '✓ 100% Normal Detection', fontsize=9, ha='center', va='center', 
        color=GREEN, fontweight='bold')

# Arrow to application
ax.annotate('', xy=(13, 4.5), xytext=(12.5, 4.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Application Box
rect_app = FancyBboxPatch((13, 3), 2.5, 3, boxstyle="round,pad=0.1", 
                           facecolor=RED, edgecolor='black', linewidth=2, alpha=0.8)
ax.add_patch(rect_app)
ax.text(14.25, 5.3, '⚠️', fontsize=30, ha='center', va='center')
ax.text(14.25, 4.3, 'Earthquake\nEarly Warning', fontsize=11, ha='center', va='center', 
        color='white', fontweight='bold')
ax.text(14.25, 3.4, 'Real-time\nMonitoring', fontsize=9, ha='center', va='center', color='white')

# ============================================================================
# BOTTOM: Key Findings
# ============================================================================
# Key findings box
rect_key = FancyBboxPatch((0.5, 0.3), 15, 1.5, boxstyle="round,pad=0.1", 
                           facecolor='#2c3e50', edgecolor='black', linewidth=2)
ax.add_patch(rect_key)

ax.text(8, 1.3, 'Key Findings', fontsize=12, fontweight='bold', ha='center', va='center', color='white')
ax.text(8, 0.7, 'VGG16: Best for Research (98.68% accuracy)  |  EfficientNet-B0: Best for Deployment (26× smaller, 2.5× faster)  |  Grad-CAM: Physically Interpretable', 
        fontsize=10, ha='center', va='center', color='white')

# Save
output_dir = Path('publication_package')
output_dir.mkdir(exist_ok=True)

plt.tight_layout()
fig.savefig(output_dir / 'graphical_abstract.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
fig.savefig(output_dir / 'graphical_abstract.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Graphical Abstract generated successfully!")
print(f"Output: {output_dir / 'graphical_abstract.png'}")
print(f"Output: {output_dir / 'graphical_abstract.pdf'}")
