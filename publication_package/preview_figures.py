"""
Preview Generated Figures
Quick script to open all generated figures for review
"""

import os
import subprocess
import sys

def preview_figures():
    """Open all generated figures for preview"""
    
    figures = [
        'fig_confusion.png',
        'fig_gradcam.png',
        'fig_architecture_comparison.png',
        'fig_deployment_feasibility.png'
    ]
    
    print("\n" + "="*80)
    print("PREVIEW GENERATED FIGURES")
    print("="*80)
    
    for fig in figures:
        if os.path.exists(fig):
            print(f"\n✓ Opening: {fig}")
            try:
                # Windows
                os.startfile(fig)
            except AttributeError:
                # macOS
                try:
                    subprocess.call(['open', fig])
                except:
                    # Linux
                    subprocess.call(['xdg-open', fig])
        else:
            print(f"\n✗ Not found: {fig}")
    
    print("\n" + "="*80)
    print("All figures opened in default image viewer")
    print("="*80)
    print("\nPlease check:")
    print("  1. Text is readable and not overlapping")
    print("  2. Colors are clear and distinguishable")
    print("  3. Labels are properly positioned")
    print("  4. Resolution is high quality (300 DPI)")
    print("\nIf any issues, please report and I will fix them.")
    print("="*80)

if __name__ == "__main__":
    preview_figures()
