import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, w, h, text, color='#dddddd', subtext=None):
    # Box
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  linewidth=1, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    
    # Main Text
    ax.text(x + w/2, y + h/2 + (0.1 if subtext else 0), text, 
            ha='center', va='center', fontsize=9, fontweight='bold', multialignment='center')
    
    # Subtext
    if subtext:
        ax.text(x + w/2, y + h/2 - 0.2, subtext, 
                ha='center', va='center', fontsize=7, color='#333333', multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#555555'))

def create_architecture_diagram_horizontal():
    # Make wide figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(8, 7.5, "EfficientNet-B0 Hierarchical Champion Model (Left-to-Right Flow)", 
            ha='center', fontsize=16, fontweight='bold', color='#2c3e50')
    ax.text(8, 7.2, "File Reference: src/trainer_effi.py", 
            ha='center', fontsize=10, style='italic', color='blue')

    # 1. INPUT (Far Left)
    draw_box(ax, 0.5, 3.5, 2.0, 1.5, "Input\nSpectrogram", "#e8f6f3", "(Batch, 3, 224, 224)")
    draw_arrow(ax, 2.5, 4.25, 3.5, 4.25)

    # 2. BACKBONE
    draw_box(ax, 3.5, 3.0, 2.5, 2.5, "Backbone:\nEfficientNet-B0", "#d6eaf8", 
             "Pretrained ImageNet\nOutput: (Batch, 1280, 7, 7)\nFrozen in Phase 1")
    draw_arrow(ax, 6.0, 4.25, 7.0, 4.25)

    # 3. POOLING & NECK
    draw_box(ax, 7.0, 3.0, 2.5, 2.5, "Shared Neck", "#fcf3cf", 
             "GlobalAvgPool -> 1280\nLinear -> 512\nSiLU -> Dropout(0.3)")
    
    # Feature Vector Output
    # draw_arrow(ax, 9.5, 4.25, 10.5, 4.25)

    # Arrows to Heads (Branching)
    # Top Branch (Binary)
    draw_arrow(ax, 9.5, 4.25, 11.5, 6.5) 
    # Middle Branch (Magnitude)
    draw_arrow(ax, 9.5, 4.25, 11.5, 4.25)
    # Bottom Branch (Azimuth)
    draw_arrow(ax, 9.5, 4.25, 11.5, 2.0)

    # 4. HEADS (Stacked Vertically on the Right)
    
    # Stage 1: Binary (Top)
    draw_box(ax, 11.5, 5.5, 3.5, 2.0, "Stage 1: Binary Head\n(Gatekeeper)", "#d5f5e3", 
             "Linear(512 -> 2)\n[Normal, Precursor]\nLoss Weight: 2.0")
             
    # Stage 2: Magnitude (Middle)
    draw_box(ax, 11.5, 3.25, 3.5, 2.0, "Stage 2: Magnitude Head\n(Energy Estimator)", "#fdebd0", 
             "Linear(512 -> 4)\n[Norm, Mod, Med, Large]\nLoss Weight: 1.0")

    # Stage 3: Azimuth (Bottom)
    draw_box(ax, 11.5, 1.0, 3.5, 2.0, "Stage 3: Azimuth Head\n(Spatial Locator)", "#fadbd8", 
             "Linear(512 -> 9)\n[8 Directions + Normal]\nLoss Weight: 0.5")

    # Legend / Notes
    ax.text(8, 0.5, "Note: Hierarchical Loss = w1*L_bin + w2*L_mag + w3*L_azi", 
            ha='center', fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    output_path = "d:/multi/auto_effi/architecture_diagram.png" # Overwrite same file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    create_architecture_diagram_horizontal()
