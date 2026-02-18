#!/usr/bin/env python3
"""
Generate Progress Flowchart for Dissertation
Menampilkan flowchart dengan status progress setiap poin
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon, Rectangle, Circle
import numpy as np
from pathlib import Path

def draw_process_box_with_progress(ax, x, y, width, height, text, progress, status='done', fontsize=8):
    """Draw a process box with progress indicator"""
    # Status colors
    if status == 'done':
        face_color = '#d5f5e3'  # Light green
        edge_color = '#27ae60'  # Green
        check_mark = '✓'
    elif status == 'partial':
        face_color = '#fef9e7'  # Light yellow
        edge_color = '#f39c12'  # Orange
        check_mark = '◐'
    else:  # pending
        face_color = '#fadbd8'  # Light red
        edge_color = '#e74c3c'  # Red
        check_mark = '○'
    
    # Main box
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=face_color, edgecolor=edge_color, linewidth=2)
    ax.add_patch(rect)
    
    # Text
    lines = text.split('\n')
    if len(lines) == 1:
        ax.text(x, y + 0.05, text, fontsize=fontsize, ha='center', va='center', wrap=True)
    else:
        line_height = 0.2
        start_y = y + (len(lines) - 1) * line_height / 2 + 0.1
        for i, line in enumerate(lines):
            ax.text(x, start_y - i * line_height, line, fontsize=fontsize, 
                    ha='center', va='center')
    
    # Progress bar
    bar_width = width - 0.2
    bar_height = 0.12
    bar_y = y - height/2 + 0.15
    
    # Background bar
    bar_bg = Rectangle((x - bar_width/2, bar_y), bar_width, bar_height,
                        facecolor='#ecf0f1', edgecolor='none')
    ax.add_patch(bar_bg)
    
    # Progress bar
    prog_width = bar_width * (progress / 100)
    bar_prog = Rectangle((x - bar_width/2, bar_y), prog_width, bar_height,
                          facecolor=edge_color, edgecolor='none')
    ax.add_patch(bar_prog)
    
    # Progress text
    ax.text(x, bar_y + bar_height/2, f'{progress}%', fontsize=6, 
            ha='center', va='center', color='white' if progress > 50 else 'black',
            fontweight='bold')
    
    # Status indicator
    ax.text(x + width/2 - 0.15, y + height/2 - 0.15, check_mark, 
            fontsize=10, ha='center', va='center', color=edge_color, fontweight='bold')

def draw_decision_diamond_with_status(ax, x, y, width, height, text, achieved=False, fontsize=7):
    """Draw a decision diamond with achievement status"""
    if achieved:
        face_color = '#d5f5e3'
        edge_color = '#27ae60'
    else:
        face_color = 'white'
        edge_color = 'black'
    
    diamond = Polygon([(x, y + height/2), (x + width/2, y), 
                       (x, y - height/2), (x - width/2, y)],
                      facecolor=face_color, edgecolor=edge_color, linewidth=1.5)
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
    circle = Circle((x, y), radius, facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', fontweight='bold')

def draw_arrow(ax, start, end, color='black'):
    """Draw an arrow from start to end"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

def draw_line(ax, start, end, color='black'):
    """Draw a line without arrow"""
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=1.5)

def create_progress_flowchart():
    # Create figure with high DPI
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_aspect('equal')
    fig.patch.set_facecolor('white')
    
    # Colors for year boxes
    YEAR1_COLOR = '#E67E22'  # Orange
    YEAR2_COLOR = '#27AE60'  # Green
    
    # Title
    ax.text(8, 11.5, 'Progress Flowchart Disertasi', fontsize=16, fontweight='bold', 
            ha='center', va='center', color='#2c3e50')
    ax.text(8, 11.1, 'Pengembangan Sistem Prediksi Gempa Bumi Berbasis Deep Learning', 
            fontsize=10, ha='center', va='center', color='#7f8c8d', style='italic')
    ax.text(8, 10.75, 'Update: 11 Februari 2026', fontsize=8, ha='center', va='center', color='#95a5a6')
    
    # Box dimensions
    BOX_W = 2.0
    BOX_H = 0.9
    DIAMOND_W = 1.3
    DIAMOND_H = 1.0
    
    # ========================================================================
    # TAHUN PERTAMA - Left side (Orange border)
    # ========================================================================
    
    # Year 1 border box
    year1_rect = FancyBboxPatch((0.3, 1.0), 7.2, 9.3, 
                                 boxstyle="round,pad=0.05,rounding_size=0.2",
                                 facecolor='#fef9e7', edgecolor=YEAR1_COLOR, 
                                 linewidth=3, linestyle='--', alpha=0.3)
    ax.add_patch(year1_rect)
    ax.text(3.9, 0.6, 'TAHUN PERTAMA (Progress: 95%)', fontsize=11, fontweight='bold', 
            ha='center', va='center', color=YEAR1_COLOR)
    
    # Progress bar for Year 1
    bar_y1 = 0.35
    bar_bg1 = Rectangle((1.5, bar_y1), 4.8, 0.15, facecolor='#ecf0f1', edgecolor='none')
    ax.add_patch(bar_bg1)
    bar_prog1 = Rectangle((1.5, bar_y1), 4.8 * 0.95, 0.15, facecolor=YEAR1_COLOR, edgecolor='none')
    ax.add_patch(bar_prog1)
    
    # --- Column 1: Main flow ---
    col1_x = 2.0
    
    # Start
    draw_terminal(ax, col1_x, 9.8, 1.2, 0.5, 'Start')
    
    # Process 1: Pengumpulan data geomagnetik - DONE 100%
    draw_process_box_with_progress(ax, col1_x, 8.6, BOX_W, BOX_H, 
                                   '1. Pengumpulan data\ngeomagnetik', 100, 'done')
    draw_arrow(ax, (col1_x, 9.55), (col1_x, 9.05))
    
    # Process 2: Pre-processing - DONE 100%
    draw_process_box_with_progress(ax, col1_x, 7.4, BOX_W, BOX_H, 
                                   '2. Pre-processing\ndata geomagnetik', 100, 'done')
    draw_arrow(ax, (col1_x, 8.15), (col1_x, 7.85))
    
    # Process 3: Ekstraksi fitur - DONE 100%
    draw_process_box_with_progress(ax, col1_x, 6.2, BOX_W, BOX_H, 
                                   '3. Ekstraksi fitur\n(STFT Spectrogram)', 100, 'done')
    draw_arrow(ax, (col1_x, 6.95), (col1_x, 6.65))
    
    # Process 4: Pembuatan model CNN - DONE 100%
    draw_process_box_with_progress(ax, col1_x, 5.0, BOX_W, BOX_H, 
                                   '4. Pembuatan model\ndeteksi CNN', 100, 'done')
    draw_arrow(ax, (col1_x, 5.75), (col1_x, 5.45))
    
    # Decision 1: Akurasi > 80%? - ACHIEVED
    draw_decision_diamond_with_status(ax, col1_x, 3.7, DIAMOND_W, DIAMOND_H, 
                                      'Akurasi >\n80%?', achieved=True)
    draw_arrow(ax, (col1_x, 4.55), (col1_x, 4.2))
    
    # Achievement label
    ax.text(col1_x + 0.9, 3.7, '97.47%', fontsize=8, ha='left', va='center', 
            color='#27ae60', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#d5f5e3', edgecolor='#27ae60'))
    
    # Connector a
    draw_connector(ax, col1_x, 2.5, 0.25, 'a')
    draw_arrow(ax, (col1_x, 3.2), (col1_x, 2.75))
    ax.text(col1_x + 0.15, 2.95, 'Ya', fontsize=7, ha='left', va='center')
    
    # --- Column 2: After connector a ---
    col2_x = 5.5
    
    # Connector a (top)
    draw_connector(ax, col2_x, 9.8, 0.25, 'a')
    
    # Process 5: Generator Data Sintetis - PARTIAL 70%
    draw_process_box_with_progress(ax, col2_x, 8.5, BOX_W, BOX_H+0.1, 
                                   '5. Pembuatan\nGenerator Data\nSintetis (SMOTE)', 70, 'partial', fontsize=7)
    draw_arrow(ax, (col2_x, 9.55), (col2_x, 9.05))
    
    # Process 6: Model self-updating - DONE 100%
    draw_process_box_with_progress(ax, col2_x, 7.0, BOX_W, BOX_H+0.2, 
                                   '6. Model CNN\ndengan fitur\nself-updating', 100, 'done', fontsize=7)
    draw_arrow(ax, (col2_x, 7.95), (col2_x, 7.55))
    
    # Decision 2: Akurasi > 95%? - ACHIEVED
    draw_decision_diamond_with_status(ax, col2_x, 5.5, DIAMOND_W, DIAMOND_H, 
                                      'Akurasi >\n95%?', achieved=True)
    draw_arrow(ax, (col2_x, 6.35), (col2_x, 6.0))
    
    # Achievement label
    ax.text(col2_x + 0.9, 5.5, '97.47%', fontsize=8, ha='left', va='center', 
            color='#27ae60', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#d5f5e3', edgecolor='#27ae60'))
    
    # Connector b
    draw_connector(ax, col2_x, 4.3, 0.25, 'b')
    draw_arrow(ax, (col2_x, 5.0), (col2_x, 4.55))
    ax.text(col2_x + 0.15, 4.75, 'Ya', fontsize=7, ha='left', va='center')
    
    # ========================================================================
    # TAHUN KEDUA - Right side (Green border)
    # ========================================================================
    
    # Year 2 border box
    year2_rect = FancyBboxPatch((8.0, 1.0), 7.7, 9.3, 
                                 boxstyle="round,pad=0.05,rounding_size=0.2",
                                 facecolor='#eafaf1', edgecolor=YEAR2_COLOR, 
                                 linewidth=3, linestyle='--', alpha=0.3)
    ax.add_patch(year2_rect)
    ax.text(11.85, 0.6, 'TAHUN KEDUA (Progress: 7%)', fontsize=11, fontweight='bold', 
            ha='center', va='center', color=YEAR2_COLOR)
    
    # Progress bar for Year 2
    bar_y2 = 0.35
    bar_bg2 = Rectangle((9.5, bar_y2), 4.8, 0.15, facecolor='#ecf0f1', edgecolor='none')
    ax.add_patch(bar_bg2)
    bar_prog2 = Rectangle((9.5, bar_y2), 4.8 * 0.07, 0.15, facecolor=YEAR2_COLOR, edgecolor='none')
    ax.add_patch(bar_prog2)
    
    # --- Column 3: From connector b ---
    col3_x = 9.5
    
    # Connector b (top)
    draw_connector(ax, col3_x, 9.8, 0.25, 'b')
    
    # Process 7: Pengumpulan data pendukung - PENDING 0%
    draw_process_box_with_progress(ax, col3_x, 8.6, BOX_W, BOX_H, 
                                   '7. Pengumpulan data\npendukung', 0, 'pending')
    draw_arrow(ax, (col3_x, 9.55), (col3_x, 9.05))
    
    # Process 8: Pre-processing data pendukung - PENDING 0%
    draw_process_box_with_progress(ax, col3_x, 7.4, BOX_W, BOX_H, 
                                   '8. Pre-processing\ndata pendukung', 0, 'pending')
    draw_arrow(ax, (col3_x, 8.15), (col3_x, 7.85))
    
    # Process 9: Integrasi multi-parameter - PENDING 0%
    draw_process_box_with_progress(ax, col3_x, 6.2, BOX_W, BOX_H, 
                                   '9. Integrasi data\nmulti-parameter', 0, 'pending')
    draw_arrow(ax, (col3_x, 6.95), (col3_x, 6.65))
    
    # Process 10: Model prediksi parameter gempa - PARTIAL 40%
    draw_process_box_with_progress(ax, col3_x, 5.0, BOX_W, BOX_H+0.1, 
                                   '10. Model prediksi\nparameter gempa', 40, 'partial', fontsize=7)
    draw_arrow(ax, (col3_x, 5.75), (col3_x, 5.45))
    
    # Decision 3: Akurasi > 85%? - NOT YET
    draw_decision_diamond_with_status(ax, col3_x, 3.7, DIAMOND_W, DIAMOND_H, 
                                      'Akurasi >\n85%?', achieved=False)
    draw_arrow(ax, (col3_x, 4.45), (col3_x, 4.2))
    
    # Connector c
    draw_connector(ax, col3_x, 2.5, 0.25, 'c')
    draw_arrow(ax, (col3_x, 3.2), (col3_x, 2.75))
    
    # --- Column 4: From connector c ---
    col4_x = 13.5
    
    # Connector c (top)
    draw_connector(ax, col4_x, 9.8, 0.25, 'c')
    
    # Process 11: Generator Data Sintetis (for supporting data) - PENDING 0%
    draw_process_box_with_progress(ax, col4_x, 8.5, BOX_W, BOX_H+0.1, 
                                   '11. Generator Data\nSintetis\n(data pendukung)', 0, 'pending', fontsize=7)
    draw_arrow(ax, (col4_x, 9.55), (col4_x, 9.05))
    
    # Process 12: Model online learning - PENDING 0%
    draw_process_box_with_progress(ax, col4_x, 7.0, BOX_W, BOX_H+0.2, 
                                   '12. Model prediksi\ndengan fitur\nonline learning', 0, 'pending', fontsize=7)
    draw_arrow(ax, (col4_x, 7.95), (col4_x, 7.55))
    
    # Decision 4: Akurasi > 95%? - NOT YET
    draw_decision_diamond_with_status(ax, col4_x, 5.5, DIAMOND_W, DIAMOND_H, 
                                      'Akurasi >\n95%?', achieved=False)
    draw_arrow(ax, (col4_x, 6.35), (col4_x, 6.0))
    
    # Finish
    draw_terminal(ax, col4_x, 4.3, 1.2, 0.5, 'Finish')
    draw_arrow(ax, (col4_x, 5.0), (col4_x, 4.55))
    
    # ========================================================================
    # LEGEND
    # ========================================================================
    legend_y = 11.3
    legend_x = 12.5
    
    # Legend box
    legend_rect = FancyBboxPatch((legend_x - 0.3, legend_y - 1.1), 3.5, 1.3,
                                  boxstyle="round,pad=0.05",
                                  facecolor='white', edgecolor='#bdc3c7', linewidth=1)
    ax.add_patch(legend_rect)
    ax.text(legend_x + 1.45, legend_y + 0.05, 'Legend', fontsize=9, fontweight='bold', 
            ha='center', va='center')
    
    # Done
    done_rect = Rectangle((legend_x, legend_y - 0.35), 0.3, 0.2, facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=1.5)
    ax.add_patch(done_rect)
    ax.text(legend_x + 0.45, legend_y - 0.25, '✓ Selesai (100%)', fontsize=7, ha='left', va='center')
    
    # Partial
    partial_rect = Rectangle((legend_x, legend_y - 0.65), 0.3, 0.2, facecolor='#fef9e7', edgecolor='#f39c12', linewidth=1.5)
    ax.add_patch(partial_rect)
    ax.text(legend_x + 0.45, legend_y - 0.55, '◐ Sebagian', fontsize=7, ha='left', va='center')
    
    # Pending
    pending_rect = Rectangle((legend_x, legend_y - 0.95), 0.3, 0.2, facecolor='#fadbd8', edgecolor='#e74c3c', linewidth=1.5)
    ax.add_patch(pending_rect)
    ax.text(legend_x + 0.45, legend_y - 0.85, '○ Belum dimulai', fontsize=7, ha='left', va='center')
    
    # ========================================================================
    # OVERALL PROGRESS
    # ========================================================================
    overall_y = 10.4
    ax.text(1.5, overall_y, 'Overall Progress:', fontsize=10, fontweight='bold', 
            ha='left', va='center', color='#2c3e50')
    
    # Overall progress bar
    bar_overall_bg = Rectangle((4.0, overall_y - 0.1), 6.0, 0.25, facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=1)
    ax.add_patch(bar_overall_bg)
    bar_overall_prog = Rectangle((4.0, overall_y - 0.1), 6.0 * 0.51, 0.25, facecolor='#3498db', edgecolor='none')
    ax.add_patch(bar_overall_prog)
    ax.text(7.0, overall_y, '51%', fontsize=9, ha='center', va='center', color='white', fontweight='bold')
    
    # ========================================================================
    # Save outputs
    # ========================================================================
    output_dir = Path('disertasi/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with high DPI for publication
    fig.savefig(output_dir / 'progress_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    fig.savefig(output_dir / 'progress_flowchart.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    fig.savefig('disertasi_progress_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    
    plt.close()
    
    print("=" * 70)
    print("Progress Flowchart generated successfully!")
    print("=" * 70)
    print(f"\nOutput files (300 DPI - Publication Quality):")
    print(f"  - disertasi/figures/progress_flowchart.png")
    print(f"  - disertasi/figures/progress_flowchart.pdf")
    print(f"  - disertasi_progress_flowchart.png")
    print("\nProgress Summary:")
    print("  TAHUN PERTAMA (95%):")
    print("    ✓ 1. Pengumpulan data geomagnetik - 100%")
    print("    ✓ 2. Pre-processing data - 100%")
    print("    ✓ 3. Ekstraksi fitur - 100%")
    print("    ✓ 4. Model deteksi CNN - 100% (97.47%)")
    print("    ◐ 5. Generator data sintetis - 70%")
    print("    ✓ 6. Model self-updating - 100% (97.47%)")
    print("  TAHUN KEDUA (7%):")
    print("    ○ 7. Pengumpulan data pendukung - 0%")
    print("    ○ 8. Pre-processing data pendukung - 0%")
    print("    ○ 9. Integrasi multi-parameter - 0%")
    print("    ◐ 10. Model prediksi parameter - 40%")
    print("    ○ 11. Generator data sintetis (pendukung) - 0%")
    print("    ○ 12. Model online learning - 0%")

if __name__ == '__main__':
    create_progress_flowchart()
