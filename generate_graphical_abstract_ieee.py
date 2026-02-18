#!/usr/bin/env python3
"""
Generate Graphical Abstract for IEEE TGRS submission.
Alur: BMKG Magnetometer → STFT Preprocessing → VGG16 vs EfficientNet → Grad-CAM → Results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from pathlib import Path

# Set up figure - IEEE recommends landscape format
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')
fig.patch.set_facecolor('white')

# Colors - Professional palette
BMKG_BLUE = '#1a5276'
STFT_ORANGE = '#e67e22'
VGG_BLUE = '#2980b9'
EFF_GREEN = '#27ae60'
GRADCAM_RED = '#c0392b'
RESULT_PURPLE = '#8e44ad'
GRAY = '#7f8c8d'
LIGHT_GRAY = '#ecf0f1'

# ============================================================================
# TITLE
# ============================================================================
ax.text(7, 6.6, 'Deep Learning-Based Earthquake Precursor Detection from Geomagnetic Data', 
        fontsize=13, fontweight='bold', ha='center', va='center', color='#2c3e50')
ax.text(7, 6.2, 'Comparative Study: VGG16 vs EfficientNet-B0 with Grad-CAM Interpretability', 
        fontsize=10, ha='center', va='center', style='italic', color=GRAY)

# ============================================================================
# STAGE 1: BMKG Magnetometer Data
# ============================================================================
# Box
rect1 = FancyBboxPatch((0.3, 3.2), 2.2, 2.5, boxstyle="round,pad=0.08", 
                        facecolor=LIGHT_GRAY, edgecolor=BMKG_BLUE, linewidth=2)
ax.add_patch(rect1)

# BMKG Logo placeholder (text)
ax.text(1.4, 5.2, 'BMKG', fontsize=14, fontweight='bold', ha='center', va='center', color=BMKG_BLUE)
ax.text(1.4, 4.7, 'Indonesia', fontsize=9, ha='center', va='center', color=BMKG_BLUE)
ax.text(1.4, 4.2, 'Magnetometer', fontsize=9, ha='center', va='center', color=BMKG_BLUE)

# Data details
ax.text(1.4, 3.7, '25 Stations', fontsize=8, ha='center', va='center')
ax.text(1.4, 3.4, 'H, D, Z (1 Hz)', fontsize=8, ha='center', va='center', color=GRAY)

# Label
ax.text(1.4, 2.9, '① Data Source', fontsize=8, fontweight='bold', ha='center', va='center', color=BMKG_BLUE)

# Arrow 1→2
ax.annotate('', xy=(2.8, 4.45), xytext=(2.5, 4.45),
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

# ============================================================================
# STAGE 2: STFT Preprocessing
# ============================================================================
rect2 = FancyBboxPatch((2.8, 3.2), 2.2, 2.5, boxstyle="round,pad=0.08", 
                        facecolor='#fef9e7', edgecolor=STFT_ORANGE, linewidth=2)
ax.add_patch(rect2)

ax.text(3.9, 5.2, 'STFT', fontsize=14, fontweight='bold', ha='center', va='center', color=STFT_ORANGE)
ax.text(3.9, 4.7, 'Transform', fontsize=9, ha='center', va='center', color=STFT_ORANGE)
ax.text(3.9, 4.2, 'Spectrogram', fontsize=9, ha='center', va='center', color=STFT_ORANGE)

# Spectrogram mini visualization
ax_spec = fig.add_axes([0.24, 0.47, 0.08, 0.1])
np.random.seed(42)
spec = np.random.rand(15, 30) * np.linspace(1, 0.2, 15).reshape(-1, 1)
ax_spec.imshow(spec, cmap='viridis', aspect='auto')
ax_spec.axis('off')

ax.text(3.9, 3.4, 'ULF: 0.001-0.5 Hz', fontsize=7, ha='center', va='center', color=GRAY)
ax.text(3.9, 2.9, '② Preprocessing', fontsize=8, fontweight='bold', ha='center', va='center', color=STFT_ORANGE)

# Arrow 2→3
ax.annotate('', xy=(5.3, 4.45), xytext=(5.0, 4.45),
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

# ============================================================================
# STAGE 3: Model Comparison (VGG16 vs EfficientNet)
# ============================================================================
# Main comparison box
rect3 = FancyBboxPatch((5.3, 2.7), 3.4, 3.5, boxstyle="round,pad=0.08", 
                        facecolor='white', edgecolor='#2c3e50', linewidth=2)
ax.add_patch(rect3)

ax.text(7, 5.9, '③ Architecture Comparison', fontsize=8, fontweight='bold', 
        ha='center', va='center', color='#2c3e50')

# VGG16 sub-box
rect_vgg = FancyBboxPatch((5.5, 4.3), 1.5, 1.4, boxstyle="round,pad=0.05", 
                           facecolor=VGG_BLUE, edgecolor='white', linewidth=1, alpha=0.9)
ax.add_patch(rect_vgg)
ax.text(6.25, 5.35, 'VGG16', fontsize=10, fontweight='bold', ha='center', va='center', color='white')
ax.text(6.25, 5.0, '98.68%', fontsize=12, fontweight='bold', ha='center', va='center', color='white')
ax.text(6.25, 4.65, 'Mag Acc', fontsize=7, ha='center', va='center', color='white')
ax.text(6.25, 4.4, '528 MB', fontsize=7, ha='center', va='center', color='#d5dbdb')

# EfficientNet sub-box
rect_eff = FancyBboxPatch((7.0, 4.3), 1.5, 1.4, boxstyle="round,pad=0.05", 
                           facecolor=EFF_GREEN, edgecolor='white', linewidth=1, alpha=0.9)
ax.add_patch(rect_eff)
ax.text(7.75, 5.35, 'EfficientNet', fontsize=9, fontweight='bold', ha='center', va='center', color='white')
ax.text(7.75, 5.0, '94.37%', fontsize=12, fontweight='bold', ha='center', va='center', color='white')
ax.text(7.75, 4.65, 'Mag Acc', fontsize=7, ha='center', va='center', color='white')
ax.text(7.75, 4.4, '20 MB', fontsize=7, ha='center', va='center', color='#d5dbdb')

# Comparison metrics
ax.text(7, 3.9, '━━━━━━━━━━━━━━━━━━━━', fontsize=6, ha='center', va='center', color=GRAY)
ax.text(7, 3.55, '26× Smaller Model', fontsize=9, ha='center', va='center', color=EFF_GREEN, fontweight='bold')
ax.text(7, 3.2, '2.5× Faster Inference', fontsize=9, ha='center', va='center', color=EFF_GREEN, fontweight='bold')
ax.text(7, 2.85, 'LOEO Validation: <5% Drop', fontsize=8, ha='center', va='center', color=GRAY)

# Arrow 3→4
ax.annotate('', xy=(9.0, 4.45), xytext=(8.7, 4.45),
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

# ============================================================================
# STAGE 4: Grad-CAM Interpretability
# ============================================================================
rect4 = FancyBboxPatch((9.0, 3.2), 2.2, 2.5, boxstyle="round,pad=0.08", 
                        facecolor='#fdedec', edgecolor=GRADCAM_RED, linewidth=2)
ax.add_patch(rect4)

ax.text(10.1, 5.2, 'Grad-CAM', fontsize=12, fontweight='bold', ha='center', va='center', color=GRADCAM_RED)
ax.text(10.1, 4.7, 'Analysis', fontsize=9, ha='center', va='center', color=GRADCAM_RED)
ax.text(10.1, 4.2, 'Explainability', fontsize=9, ha='center', va='center', color=GRADCAM_RED)

# Grad-CAM mini heatmap
ax_gcam = fig.add_axes([0.68, 0.47, 0.08, 0.1])
np.random.seed(123)
gcam = np.zeros((15, 30))
gcam[3:10, 5:20] = np.random.rand(7, 15) * 0.8 + 0.2
ax_gcam.imshow(gcam, cmap='jet', aspect='auto', alpha=0.8)
ax_gcam.axis('off')

ax.text(10.1, 3.4, 'ULF Band Focus', fontsize=7, ha='center', va='center', color=GRAY)
ax.text(10.1, 2.9, '④ Interpretation', fontsize=8, fontweight='bold', ha='center', va='center', color=GRADCAM_RED)

# Arrow 4→5
ax.annotate('', xy=(11.5, 4.45), xytext=(11.2, 4.45),
            arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

# ============================================================================
# STAGE 5: Results & Application
# ============================================================================
rect5 = FancyBboxPatch((11.5, 3.2), 2.2, 2.5, boxstyle="round,pad=0.08", 
                        facecolor='#f4ecf7', edgecolor=RESULT_PURPLE, linewidth=2)
ax.add_patch(rect5)

ax.text(12.6, 5.2, 'Early', fontsize=12, fontweight='bold', ha='center', va='center', color=RESULT_PURPLE)
ax.text(12.6, 4.7, 'Warning', fontsize=12, fontweight='bold', ha='center', va='center', color=RESULT_PURPLE)
ax.text(12.6, 4.2, 'System', fontsize=9, ha='center', va='center', color=RESULT_PURPLE)

ax.text(12.6, 3.7, 'Magnitude: 4 classes', fontsize=7, ha='center', va='center')
ax.text(12.6, 3.45, 'Azimuth: 9 directions', fontsize=7, ha='center', va='center')
ax.text(12.6, 2.9, '⑤ Application', fontsize=8, fontweight='bold', ha='center', va='center', color=RESULT_PURPLE)

# ============================================================================
# BOTTOM: Key Contributions
# ============================================================================
rect_bottom = FancyBboxPatch((0.3, 0.4), 13.4, 1.8, boxstyle="round,pad=0.08", 
                              facecolor='#2c3e50', edgecolor='#1a252f', linewidth=2)
ax.add_patch(rect_bottom)

ax.text(7, 1.9, 'Key Contributions', fontsize=11, fontweight='bold', ha='center', va='center', color='white')

# Three columns of contributions
ax.text(2.5, 1.35, '* First VGG16 vs EfficientNet\n   comparison for geomagnetic\n   precursor detection', 
        fontsize=8, ha='center', va='center', color='white')
ax.text(7, 1.35, '* Multi-task learning:\n   Magnitude + Azimuth\n   prediction simultaneously', 
        fontsize=8, ha='center', va='center', color='white')
ax.text(11.5, 1.35, '* Physically interpretable:\n   Grad-CAM confirms ULF\n   band focus (0.001-0.01 Hz)', 
        fontsize=8, ha='center', va='center', color='white')

# Dividers
ax.plot([4.5, 4.5], [0.7, 1.9], color='white', alpha=0.3, lw=1)
ax.plot([9.5, 9.5], [0.7, 1.9], color='white', alpha=0.3, lw=1)

# ============================================================================
# Save outputs
# ============================================================================
output_dir = Path('publication/paper')
output_dir.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
fig.savefig(output_dir / 'graphical_abstract.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
fig.savefig(output_dir / 'graphical_abstract.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

# Also save to publication_package
pkg_dir = Path('publication_package')
pkg_dir.mkdir(exist_ok=True)
fig.savefig(pkg_dir / 'graphical_abstract_ieee.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
fig.savefig(pkg_dir / 'graphical_abstract_ieee.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

plt.close()

print("=" * 60)
print("Graphical Abstract for IEEE TGRS generated successfully!")
print("=" * 60)
print(f"\nOutput files:")
print(f"  - {output_dir / 'graphical_abstract.png'}")
print(f"  - {output_dir / 'graphical_abstract.pdf'}")
print(f"  - {pkg_dir / 'graphical_abstract_ieee.png'}")
print(f"  - {pkg_dir / 'graphical_abstract_ieee.pdf'}")
print("\nAlur yang ditampilkan:")
print("  ① BMKG Magnetometer → ② STFT Spectrogram → ③ VGG16 vs EfficientNet")
print("  → ④ Grad-CAM Interpretability → ⑤ Early Warning System")
