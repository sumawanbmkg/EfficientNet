#!/usr/bin/env python3
"""
Generate Comprehensive Project Graphical Abstract
Menampilkan alur lengkap project dari pengumpulan data hingga model produksi

Alur:
1. Data Collection (BMKG Magnetometer 25 Stations)
2. Earthquake Catalog (2018-2025, 105+ events M>=6.0)
3. Precursor Detection (PC3 band, Z/H ratio)
4. Spectrogram Generation (STFT, 224x224)
5. CNN Training (EfficientNet-B0)
6. Multi-task Classification (Azimuth 8 classes, Magnitude 5 classes)
7. Production Deployment (Dashboard + Scanner)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np
from pathlib import Path

def create_graphical_abstract():
    # Set up figure - landscape format
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Color palette
    BMKG_BLUE = '#1a5276'
    CATALOG_TEAL = '#16a085'
    PRECURSOR_ORANGE = '#e67e22'
    SPEC_PURPLE = '#8e44ad'
    CNN_GREEN = '#27ae60'
    CLASS_RED = '#c0392b'
    PROD_NAVY = '#2c3e50'
    GRAY = '#7f8c8d'
    LIGHT_GRAY = '#ecf0f1'
    
    # ========================================================================
    # TITLE
    # ========================================================================
    ax.text(8, 9.5, 'Earthquake Precursor Detection System', 
            fontsize=18, fontweight='bold', ha='center', va='center', color=PROD_NAVY)
    ax.text(8, 9.0, 'Deep Learning Analysis of Geomagnetic Spectrograms for Early Warning', 
            fontsize=11, ha='center', va='center', style='italic', color=GRAY)
    
    # ========================================================================
    # ROW 1: Data Collection → Earthquake Catalog → Precursor Detection
    # ========================================================================
    
    # --- Stage 1: BMKG Magnetometer ---
    rect1 = FancyBboxPatch((0.3, 6.0), 2.8, 2.3, boxstyle="round,pad=0.1", 
                            facecolor=LIGHT_GRAY, edgecolor=BMKG_BLUE, linewidth=2.5)
    ax.add_patch(rect1)
    
    ax.text(1.7, 7.9, '① Data Collection', fontsize=9, fontweight='bold', 
            ha='center', va='center', color=BMKG_BLUE)
    ax.text(1.7, 7.4, 'BMKG', fontsize=14, fontweight='bold', ha='center', va='center', color=BMKG_BLUE)
    ax.text(1.7, 7.0, 'Magnetometer', fontsize=10, ha='center', va='center', color=BMKG_BLUE)
    
    # Station info
    ax.text(1.7, 6.5, '25 Stations', fontsize=9, ha='center', va='center', fontweight='bold')
    ax.text(1.7, 6.2, 'H, D, Z components', fontsize=8, ha='center', va='center', color=GRAY)
    
    # Arrow 1→2
    ax.annotate('', xy=(3.4, 7.15), xytext=(3.1, 7.15),
                arrowprops=dict(arrowstyle='->', color=PROD_NAVY, lw=2.5))
    
    # --- Stage 2: Earthquake Catalog ---
    rect2 = FancyBboxPatch((3.5, 6.0), 2.8, 2.3, boxstyle="round,pad=0.1", 
                            facecolor='#e8f6f3', edgecolor=CATALOG_TEAL, linewidth=2.5)
    ax.add_patch(rect2)
    
    ax.text(4.9, 7.9, '② Earthquake Catalog', fontsize=9, fontweight='bold', 
            ha='center', va='center', color=CATALOG_TEAL)
    ax.text(4.9, 7.4, '2018-2025', fontsize=12, fontweight='bold', ha='center', va='center', color=CATALOG_TEAL)
    ax.text(4.9, 7.0, 'Indonesia Region', fontsize=9, ha='center', va='center', color=CATALOG_TEAL)
    
    ax.text(4.9, 6.5, '105+ Events', fontsize=9, ha='center', va='center', fontweight='bold')
    ax.text(4.9, 6.2, 'M ≥ 6.0', fontsize=8, ha='center', va='center', color=GRAY)
    
    # Arrow 2→3
    ax.annotate('', xy=(6.6, 7.15), xytext=(6.3, 7.15),
                arrowprops=dict(arrowstyle='->', color=PROD_NAVY, lw=2.5))
    
    # --- Stage 3: Precursor Detection ---
    rect3 = FancyBboxPatch((6.7, 6.0), 2.8, 2.3, boxstyle="round,pad=0.1", 
                            facecolor='#fef5e7', edgecolor=PRECURSOR_ORANGE, linewidth=2.5)
    ax.add_patch(rect3)
    
    ax.text(8.1, 7.9, '③ Precursor Detection', fontsize=9, fontweight='bold', 
            ha='center', va='center', color=PRECURSOR_ORANGE)
    ax.text(8.1, 7.4, 'PC3 Band', fontsize=12, fontweight='bold', ha='center', va='center', color=PRECURSOR_ORANGE)
    ax.text(8.1, 7.0, '10-45 mHz', fontsize=9, ha='center', va='center', color=PRECURSOR_ORANGE)
    
    ax.text(8.1, 6.5, 'Z/H Ratio Analysis', fontsize=9, ha='center', va='center', fontweight='bold')
    ax.text(8.1, 6.2, '4-20 days window', fontsize=8, ha='center', va='center', color=GRAY)
    
    # Arrow 3→4
    ax.annotate('', xy=(9.8, 7.15), xytext=(9.5, 7.15),
                arrowprops=dict(arrowstyle='->', color=PROD_NAVY, lw=2.5))
    
    # --- Stage 4: Spectrogram Generation ---
    rect4 = FancyBboxPatch((9.9, 6.0), 2.8, 2.3, boxstyle="round,pad=0.1", 
                            facecolor='#f5eef8', edgecolor=SPEC_PURPLE, linewidth=2.5)
    ax.add_patch(rect4)
    
    ax.text(11.3, 7.9, '④ Spectrogram', fontsize=9, fontweight='bold', 
            ha='center', va='center', color=SPEC_PURPLE)
    ax.text(11.3, 7.4, 'STFT', fontsize=12, fontweight='bold', ha='center', va='center', color=SPEC_PURPLE)
    ax.text(11.3, 7.0, 'Transform', fontsize=9, ha='center', va='center', color=SPEC_PURPLE)
    
    ax.text(11.3, 6.5, '224×224 pixels', fontsize=9, ha='center', va='center', fontweight='bold')
    ax.text(11.3, 6.2, '2000+ samples', fontsize=8, ha='center', va='center', color=GRAY)
    
    # Mini spectrogram visualization
    ax_spec = fig.add_axes([0.67, 0.64, 0.06, 0.08])
    np.random.seed(42)
    spec = np.random.rand(20, 30) * np.linspace(1, 0.3, 20).reshape(-1, 1)
    ax_spec.imshow(spec, cmap='viridis', aspect='auto')
    ax_spec.axis('off')
    
    # ========================================================================
    # ARROW DOWN to Row 2
    # ========================================================================
    ax.annotate('', xy=(13.1, 5.7), xytext=(13.1, 6.0),
                arrowprops=dict(arrowstyle='->', color=PROD_NAVY, lw=2.5))
    ax.annotate('', xy=(13.1, 5.0), xytext=(13.1, 5.7),
                arrowprops=dict(arrowstyle='->', color=PROD_NAVY, lw=2.5,
                               connectionstyle="arc3,rad=0"))
    
    # ========================================================================
    # ROW 2: CNN Training → Classification → Production
    # ========================================================================
    
    # --- Stage 5: CNN Training ---
    rect5 = FancyBboxPatch((9.9, 2.5), 2.8, 2.3, boxstyle="round,pad=0.1", 
                            facecolor='#e8f8f5', edgecolor=CNN_GREEN, linewidth=2.5)
    ax.add_patch(rect5)
    
    ax.text(11.3, 4.4, '⑤ CNN Training', fontsize=9, fontweight='bold', 
            ha='center', va='center', color=CNN_GREEN)
    ax.text(11.3, 3.9, 'EfficientNet-B0', fontsize=11, fontweight='bold', ha='center', va='center', color=CNN_GREEN)
    
    # Model metrics
    ax.text(11.3, 3.4, '97.47%', fontsize=14, fontweight='bold', ha='center', va='center', color=CNN_GREEN)
    ax.text(11.3, 3.0, 'Accuracy', fontsize=8, ha='center', va='center', color=GRAY)
    ax.text(11.3, 2.7, '20 MB model', fontsize=8, ha='center', va='center', color=GRAY)
    
    # Arrow 5→6
    ax.annotate('', xy=(9.6, 3.65), xytext=(9.9, 3.65),
                arrowprops=dict(arrowstyle='->', color=PROD_NAVY, lw=2.5))
    
    # --- Stage 6: Multi-task Classification ---
    rect6 = FancyBboxPatch((5.5, 2.0), 4.0, 3.3, boxstyle="round,pad=0.1", 
                            facecolor='#fdedec', edgecolor=CLASS_RED, linewidth=2.5)
    ax.add_patch(rect6)
    
    ax.text(7.5, 4.9, '⑥ Multi-task Classification', fontsize=9, fontweight='bold', 
            ha='center', va='center', color=CLASS_RED)
    
    # Azimuth box
    rect_az = FancyBboxPatch((5.7, 3.5), 1.7, 1.2, boxstyle="round,pad=0.05", 
                              facecolor='white', edgecolor=CLASS_RED, linewidth=1.5)
    ax.add_patch(rect_az)
    ax.text(6.55, 4.4, 'Azimuth', fontsize=9, fontweight='bold', ha='center', va='center', color=CLASS_RED)
    ax.text(6.55, 4.0, '8 Classes', fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(6.55, 3.65, 'N NE E SE S SW W NW', fontsize=6, ha='center', va='center', color=GRAY)
    
    # Magnitude box
    rect_mag = FancyBboxPatch((7.6, 3.5), 1.7, 1.2, boxstyle="round,pad=0.05", 
                               facecolor='white', edgecolor=CLASS_RED, linewidth=1.5)
    ax.add_patch(rect_mag)
    ax.text(8.45, 4.4, 'Magnitude', fontsize=9, fontweight='bold', ha='center', va='center', color=CLASS_RED)
    ax.text(8.45, 4.0, '5 Classes', fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(8.45, 3.65, 'Small Mod Med Large Major', fontsize=5.5, ha='center', va='center', color=GRAY)
    
    # Class details
    ax.text(7.5, 3.1, 'Small: M<4.0 | Moderate: M4.0-4.9 | Medium: M5.0-5.9', 
            fontsize=7, ha='center', va='center', color=GRAY)
    ax.text(7.5, 2.7, 'Large: M6.0-6.9 | Major: M≥7.0', 
            fontsize=7, ha='center', va='center', color=GRAY)
    ax.text(7.5, 2.3, 'LOEO Validation: <5% accuracy drop', 
            fontsize=7, ha='center', va='center', color=GRAY, style='italic')
    
    # Arrow 6→7
    ax.annotate('', xy=(5.2, 3.65), xytext=(5.5, 3.65),
                arrowprops=dict(arrowstyle='->', color=PROD_NAVY, lw=2.5))
    
    # --- Stage 7: Production Deployment ---
    rect7 = FancyBboxPatch((1.5, 2.0), 3.5, 3.3, boxstyle="round,pad=0.1", 
                            facecolor=PROD_NAVY, edgecolor='#1a252f', linewidth=2.5)
    ax.add_patch(rect7)
    
    ax.text(3.25, 4.9, '⑦ Production System', fontsize=9, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Dashboard icon
    rect_dash = FancyBboxPatch((1.7, 3.6), 1.4, 1.0, boxstyle="round,pad=0.05", 
                                facecolor='#3498db', edgecolor='white', linewidth=1)
    ax.add_patch(rect_dash)
    ax.text(2.4, 4.3, 'Dashboard', fontsize=8, fontweight='bold', ha='center', va='center', color='white')
    ax.text(2.4, 3.9, 'Streamlit', fontsize=7, ha='center', va='center', color='white')
    
    # Scanner icon
    rect_scan = FancyBboxPatch((3.3, 3.6), 1.4, 1.0, boxstyle="round,pad=0.05", 
                                facecolor='#e74c3c', edgecolor='white', linewidth=1)
    ax.add_patch(rect_scan)
    ax.text(4.0, 4.3, 'Scanner', fontsize=8, fontweight='bold', ha='center', va='center', color='white')
    ax.text(4.0, 3.9, 'Real-time', fontsize=7, ha='center', va='center', color='white')
    
    # Features
    ax.text(3.25, 3.2, '• SSH data fetching', fontsize=7, ha='center', va='center', color='white')
    ax.text(3.25, 2.85, '• Auto-update pipeline', fontsize=7, ha='center', va='center', color='white')
    ax.text(3.25, 2.5, '• Precursor alerts', fontsize=7, ha='center', va='center', color='white')
    ax.text(3.25, 2.15, '• Historical validation', fontsize=7, ha='center', va='center', color='white')
    
    # ========================================================================
    # BOTTOM: Key Statistics & Contributions
    # ========================================================================
    rect_bottom = FancyBboxPatch((0.3, 0.3), 15.4, 1.4, boxstyle="round,pad=0.08", 
                                  facecolor='#f8f9fa', edgecolor='#dee2e6', linewidth=2)
    ax.add_patch(rect_bottom)
    
    ax.text(8, 1.45, 'Project Statistics & Key Contributions', fontsize=10, fontweight='bold', 
            ha='center', va='center', color=PROD_NAVY)
    
    # Statistics columns
    stats = [
        ('25', 'Stations'),
        ('2000+', 'Samples'),
        ('105+', 'Events'),
        ('97.47%', 'Accuracy'),
        ('8+5', 'Classes'),
        ('20 MB', 'Model Size'),
    ]
    
    x_positions = [1.5, 4.0, 6.5, 9.0, 11.5, 14.0]
    for i, (value, label) in enumerate(stats):
        ax.text(x_positions[i], 0.9, value, fontsize=12, fontweight='bold', 
                ha='center', va='center', color=CNN_GREEN if 'Accuracy' in label else PROD_NAVY)
        ax.text(x_positions[i], 0.55, label, fontsize=8, ha='center', va='center', color=GRAY)
    
    # Dividers
    for x in [2.75, 5.25, 7.75, 10.25, 12.75]:
        ax.plot([x, x], [0.45, 1.1], color='#dee2e6', lw=1)
    
    # ========================================================================
    # Save outputs
    # ========================================================================
    output_dir = Path('publication')
    output_dir.mkdir(exist_ok=True)
    
    # Don't use tight_layout due to inset axes
    fig.savefig(output_dir / 'project_graphical_abstract.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    fig.savefig(output_dir / 'project_graphical_abstract.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    # Also save to root
    fig.savefig('project_graphical_abstract.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.close()
    
    print("=" * 70)
    print("Project Graphical Abstract generated successfully!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - publication/project_graphical_abstract.png")
    print(f"  - publication/project_graphical_abstract.pdf")
    print(f"  - project_graphical_abstract.png")
    print("\nAlur yang ditampilkan:")
    print("  ① Data Collection (BMKG 25 Stations)")
    print("  ② Earthquake Catalog (2018-2025, M≥6.0)")
    print("  ③ Precursor Detection (PC3 band, Z/H ratio)")
    print("  ④ Spectrogram Generation (STFT 224×224)")
    print("  ⑤ CNN Training (EfficientNet-B0, 97.47%)")
    print("  ⑥ Multi-task Classification (8 Azimuth + 5 Magnitude)")
    print("  ⑦ Production System (Dashboard + Scanner)")

if __name__ == '__main__':
    create_graphical_abstract()
