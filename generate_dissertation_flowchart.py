#!/usr/bin/env python3
"""
Generate Dissertation Proposal Flowchart
Flowchart penelitian 2 tahun untuk deteksi prekursor gempa bumi
dengan kualitas publikasi (300 DPI)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon, Rectangle
import numpy as np
from pathlib import Path

def draw_process_box(ax, x, y, width, height, text, fontsize=9):
    """Draw a process box (rectangle with rounded corners)"""
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    # Handle multi-line text
    lines = text.split('\n')
    if len(lines) == 1:
        ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', 
                fontweight='normal', wrap=True)
    else:
        line_height = 0.25
        start_y = y + (len(lines) - 1) * line_height / 2
        for i, line in enumerate(lines):
            ax.text(x, start_y - i * line_height, line, fontsize=fontsize, 
                    ha='center', va='center', fontweight='normal')

def draw_decision_diamond(ax, x, y, width, height, text, fontsize=8):
    """Draw a decision diamond"""
    diamond = Polygon([(x, y + height/2), (x + width/2, y), 
                       (x, y - height/2), (x - width/2, y)],
                      facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center')

def draw_terminal(ax, x, y, width, height, text, fontsize=10):
    """Draw a terminal (oval/ellipse)"""
    ellipse = Ellipse((x, y), width, height, facecolor='white', 
                      edgecolor='black', linewidth=1.5)
    ax.add_patch(ellipse)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', fontweight='bold')

def draw_connector(ax, x, y, radius, text, fontsize=10):
    """Draw a connector circle"""
    circle = plt.Circle((x, y), radius, facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', fontweight='bold')

def draw_arrow(ax, start, end, color='black'):
    """Draw an arrow from start to end"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

def draw_line(ax, start, end, color='black'):
    """Draw a line without arrow"""
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=1.5)

def create_flowchart():
    # Create figure with high DPI
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')
    fig.patch.set_facecolor('white')
    
    # Colors for year boxes
    YEAR1_COLOR = '#E67E22'  # Orange
    YEAR2_COLOR = '#27AE60'  # Green
    
    # Box dimensions
    BOX_W = 1.8
    BOX_H = 0.7
    DIAMOND_W = 1.2
    DIAMOND_H = 0.9
    
    # ========================================================================
    # TAHUN PERTAMA - Left side (Orange border)
    # ========================================================================
    
    # Year 1 border box
    year1_rect = FancyBboxPatch((0.3, 0.5), 6.4, 9.0, 
                                 boxstyle="round,pad=0.05,rounding_size=0.2",
                                 facecolor='none', edgecolor=YEAR1_COLOR, 
                                 linewidth=3, linestyle='--')
    ax.add_patch(year1_rect)
    ax.text(3.5, 0.25, 'Tahun Pertama', fontsize=12, fontweight='bold', 
            ha='center', va='center', color=YEAR1_COLOR)
    
    # --- Column 1: Main flow ---
    col1_x = 1.8
    
    # Start
    draw_terminal(ax, col1_x, 9.0, 1.2, 0.5, 'Start')
    
    # Process 1: Pengumpulan data geomagnetik
    draw_process_box(ax, col1_x, 8.0, BOX_W, BOX_H, 'Pengumpulan data\ngeomagnetik')
    draw_arrow(ax, (col1_x, 8.75), (col1_x, 8.35))
    
    # Process 2: Pre-processing
    draw_process_box(ax, col1_x, 7.0, BOX_W, BOX_H, 'Pre-processing\ndata geomagnetik')
    draw_arrow(ax, (col1_x, 7.65), (col1_x, 7.35))
    
    # Process 3: Ekstraksi fitur
    draw_process_box(ax, col1_x, 6.0, BOX_W, BOX_H, 'Ekstraksi fitur')
    draw_arrow(ax, (col1_x, 6.65), (col1_x, 6.35))
    
    # Process 4: Pembuatan model CNN
    draw_process_box(ax, col1_x, 5.0, BOX_W, BOX_H, 'Pembuatan model\ndeteksi CNN')
    draw_arrow(ax, (col1_x, 5.65), (col1_x, 5.35))
    
    # Decision 1: Akurasi > 80%?
    draw_decision_diamond(ax, col1_x, 3.8, DIAMOND_W, DIAMOND_H, 'Akurasi >\n80%?')
    draw_arrow(ax, (col1_x, 4.65), (col1_x, 4.25))
    
    # Connector a
    draw_connector(ax, col1_x, 2.6, 0.3, 'a')
    draw_arrow(ax, (col1_x, 3.35), (col1_x, 2.9))
    
    # Loop back arrow (No path)
    draw_line(ax, (col1_x - 0.6, 3.8), (col1_x - 1.2, 3.8))
    draw_line(ax, (col1_x - 1.2, 3.8), (col1_x - 1.2, 5.0))
    draw_arrow(ax, (col1_x - 1.2, 5.0), (col1_x - 0.9, 5.0))
    ax.text(col1_x - 0.85, 3.6, 'Tidak', fontsize=7, ha='center', va='center')
    ax.text(col1_x + 0.15, 3.1, 'Ya', fontsize=7, ha='center', va='center')
    
    # --- Column 2: After connector a ---
    col2_x = 5.0
    
    # Connector a (top)
    draw_connector(ax, col2_x, 9.0, 0.3, 'a')
    
    # Process 5: Generator Data Sintetis
    draw_process_box(ax, col2_x, 7.8, BOX_W, BOX_H, 'Pembuatan\nGenerator Data\nSintetis', fontsize=8)
    draw_arrow(ax, (col2_x, 8.7), (col2_x, 8.25))
    
    # Process 6: Model self-updating
    draw_process_box(ax, col2_x, 6.4, BOX_W, BOX_H+0.2, 'Pengembangan\nmodel deteksi CNN\ndengan fitur\nself-updating', fontsize=8)
    draw_arrow(ax, (col2_x, 7.35), (col2_x, 6.95))
    
    # Decision 2: Akurasi > 95%?
    draw_decision_diamond(ax, col2_x, 5.0, DIAMOND_W, DIAMOND_H, 'Akurasi >\n95%?')
    draw_arrow(ax, (col2_x, 5.85), (col2_x, 5.45))
    
    # Connector b
    draw_connector(ax, col2_x, 3.8, 0.3, 'b')
    draw_arrow(ax, (col2_x, 4.55), (col2_x, 4.1))
    
    # Loop back arrow (No path)
    draw_line(ax, (col2_x - 0.6, 5.0), (col2_x - 1.2, 5.0))
    draw_line(ax, (col2_x - 1.2, 5.0), (col2_x - 1.2, 6.4))
    draw_arrow(ax, (col2_x - 1.2, 6.4), (col2_x - 0.9, 6.4))
    ax.text(col2_x - 0.85, 4.8, 'Tidak', fontsize=7, ha='center', va='center')
    ax.text(col2_x + 0.15, 4.3, 'Ya', fontsize=7, ha='center', va='center')
    
    # ========================================================================
    # TAHUN KEDUA - Right side (Green border)
    # ========================================================================
    
    # Year 2 border box
    year2_rect = FancyBboxPatch((7.0, 0.5), 6.7, 9.0, 
                                 boxstyle="round,pad=0.05,rounding_size=0.2",
                                 facecolor='none', edgecolor=YEAR2_COLOR, 
                                 linewidth=3, linestyle='--')
    ax.add_patch(year2_rect)
    ax.text(10.35, 0.25, 'Tahun Kedua', fontsize=12, fontweight='bold', 
            ha='center', va='center', color=YEAR2_COLOR)
    
    # --- Column 3: From connector b ---
    col3_x = 8.5
    
    # Connector b (top)
    draw_connector(ax, col3_x, 9.0, 0.3, 'b')
    
    # Process 7: Pengumpulan data pendukung
    draw_process_box(ax, col3_x, 7.8, BOX_W, BOX_H, 'Pengumpulan data\npendukung')
    draw_arrow(ax, (col3_x, 8.7), (col3_x, 8.15))
    
    # Process 8: Pre-processing data pendukung
    draw_process_box(ax, col3_x, 6.8, BOX_W, BOX_H, 'Pre-processing\ndata pendukung')
    draw_arrow(ax, (col3_x, 7.45), (col3_x, 7.15))
    
    # Process 9: Integrasi multi-parameter
    draw_process_box(ax, col3_x, 5.8, BOX_W, BOX_H, 'Integrasi data\nmulti-parameter')
    draw_arrow(ax, (col3_x, 6.45), (col3_x, 6.15))
    
    # Process 10: Model prediksi parameter gempa
    draw_process_box(ax, col3_x, 4.7, BOX_W, BOX_H+0.1, 'Pengembangan\nmodel prediksi\nparameter gempa', fontsize=8)
    draw_arrow(ax, (col3_x, 5.45), (col3_x, 5.15))
    
    # Decision 3: Akurasi > 85%?
    draw_decision_diamond(ax, col3_x, 3.4, DIAMOND_W, DIAMOND_H, 'Akurasi >\n85%?')
    draw_arrow(ax, (col3_x, 4.15), (col3_x, 3.85))
    
    # Connector c
    draw_connector(ax, col3_x, 2.2, 0.3, 'c')
    draw_arrow(ax, (col3_x, 2.95), (col3_x, 2.5))
    
    # Loop back arrow (No path)
    draw_line(ax, (col3_x - 0.6, 3.4), (col3_x - 1.2, 3.4))
    draw_line(ax, (col3_x - 1.2, 3.4), (col3_x - 1.2, 4.7))
    draw_arrow(ax, (col3_x - 1.2, 4.7), (col3_x - 0.9, 4.7))
    ax.text(col3_x - 0.85, 3.2, 'Tidak', fontsize=7, ha='center', va='center')
    ax.text(col3_x + 0.15, 2.7, 'Ya', fontsize=7, ha='center', va='center')
    
    # --- Column 4: From connector c ---
    col4_x = 12.0
    
    # Connector c (top)
    draw_connector(ax, col4_x, 9.0, 0.3, 'c')
    
    # Process 11: Generator Data Sintetis (for supporting data)
    draw_process_box(ax, col4_x, 7.8, BOX_W, BOX_H, 'Pembuatan\nGenerator Data\nSintetis', fontsize=8)
    draw_arrow(ax, (col4_x, 8.7), (col4_x, 8.25))
    
    # Process 12: Model online learning
    draw_process_box(ax, col4_x, 6.4, BOX_W, BOX_H+0.2, 'Pengembangan\nmodel prediksi\ndengan fitur\nonline learning', fontsize=8)
    draw_arrow(ax, (col4_x, 7.35), (col4_x, 6.95))
    
    # Decision 4: Akurasi > 95%?
    draw_decision_diamond(ax, col4_x, 5.0, DIAMOND_W, DIAMOND_H, 'Akurasi >\n95%?')
    draw_arrow(ax, (col4_x, 5.85), (col4_x, 5.45))
    
    # Finish
    draw_terminal(ax, col4_x, 3.8, 1.2, 0.5, 'Finish')
    draw_arrow(ax, (col4_x, 4.55), (col4_x, 4.05))
    
    # Loop back arrow (No path)
    draw_line(ax, (col4_x - 0.6, 5.0), (col4_x - 1.2, 5.0))
    draw_line(ax, (col4_x - 1.2, 5.0), (col4_x - 1.2, 6.4))
    draw_arrow(ax, (col4_x - 1.2, 6.4), (col4_x - 0.9, 6.4))
    ax.text(col4_x - 0.85, 4.8, 'Tidak', fontsize=7, ha='center', va='center')
    ax.text(col4_x + 0.15, 4.3, 'Ya', fontsize=7, ha='center', va='center')
    
    # ========================================================================
    # Save outputs
    # ========================================================================
    output_dir = Path('publication')
    output_dir.mkdir(exist_ok=True)
    
    # Save with high DPI for publication
    fig.savefig(output_dir / 'dissertation_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    fig.savefig(output_dir / 'dissertation_flowchart.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    fig.savefig('dissertation_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    
    # Also save TIFF for some journals
    fig.savefig(output_dir / 'dissertation_flowchart.tiff', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    
    plt.close()
    
    print("=" * 70)
    print("Dissertation Flowchart generated successfully!")
    print("=" * 70)
    print(f"\nOutput files (300 DPI - Publication Quality):")
    print(f"  - publication/dissertation_flowchart.png")
    print(f"  - publication/dissertation_flowchart.pdf")
    print(f"  - publication/dissertation_flowchart.tiff")
    print(f"  - dissertation_flowchart.png")
    print("\nFlowchart Structure:")
    print("  TAHUN PERTAMA:")
    print("    - Pengumpulan data geomagnetik")
    print("    - Pre-processing data geomagnetik")
    print("    - Ekstraksi fitur")
    print("    - Pembuatan model deteksi CNN (target: >80%)")
    print("    - Generator Data Sintetis")
    print("    - Model CNN dengan self-updating (target: >95%)")
    print("  TAHUN KEDUA:")
    print("    - Pengumpulan data pendukung")
    print("    - Pre-processing data pendukung")
    print("    - Integrasi data multi-parameter")
    print("    - Model prediksi parameter gempa (target: >85%)")
    print("    - Generator Data Sintetis")
    print("    - Model prediksi dengan online learning (target: >95%)")

if __name__ == '__main__':
    create_flowchart()
