import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box_flow(ax, x, y, w, h, title, text, color, edge_color='black', script_ref=None):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  linewidth=1.5, edgecolor=edge_color, facecolor=color)
    ax.add_patch(rect)
    
    # Title
    ax.text(x + w/2, y + h - 0.2, title, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Content
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(x + w/2, y + h/2 - 0.1, text, ha='center', va='center', fontsize=8, multialignment='center')
    
    # Script Refernce Pill
    if script_ref:
        ax.text(x + w/2, y - 0.2, f"[{script_ref}]", ha='center', fontsize=7, color='blue', style='italic')

def draw_arrow_flow(ax, x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#555555'))
    if text:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.1, text, ha='center', fontsize=7, backgroundcolor='white')

def create_workflow_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(8, 7.5, "Partial Auto-Effi Scaler Strategy Workflow", 
            ha='center', fontsize=16, fontweight='bold', color='#2c3e50')

    # 1. CONTINUOUS MONITORING
    draw_box_flow(ax, 0.5, 4.0, 3, 2.5, "1. MONITORING", 
                  "Raw Geomagnetic Data\n(H, D, Z)\nReal-time Scan", 
                  "#d6eaf8", script_ref="run_continuous_scan.py")
    
    draw_arrow_flow(ax, 3.5, 5.25, 4.5, 5.25, "Precursor Detected")

    # 2. EVIDENCE CAPTURE
    draw_box_flow(ax, 4.5, 4.0, 3, 2.5, "2. EVIDENCE CAPTURE", 
                  "Save Spectrogram -> .png\nSave Metadata -> .csv\nFolder: data/pending/", 
                  "#fcf3cf", script_ref="prekursor_scanner.py")

    draw_arrow_flow(ax, 7.5, 5.25, 8.5, 5.25, "Check Threshold >= 5")

    # 3. TRIGGER & MERGE
    draw_box_flow(ax, 8.5, 4.0, 3, 2.5, "3. TRIGGER LOGIC", 
                  "Validate CSVs\nMerge to staging_metadata.csv\n(Original + New)", 
                  "#d5f5e3", script_ref="run_effi_update.py")

    draw_arrow_flow(ax, 11.5, 5.25, 12.5, 5.25, "Start Training")

    # 4. PARTIAL TRAINING
    draw_box_flow(ax, 12.5, 3.5, 3, 3.5, "4. PARTIAL TRAINING", 
                  "Mode: Partial (<50 Data)\nFROZEN: Backbone + Bin Head\nACTIVE: Mag Head + Azi Head\nOptimized Loss Function", 
                  "#fadbd8", edge_color='red', script_ref="trainer_effi.py")
    
    # 5. DEPLOYMENT FLOW (Bottom)
    draw_arrow_flow(ax, 14.0, 3.5, 14.0, 2.5)
    
    draw_box_flow(ax, 12.5, 0.5, 3, 1.5, "5. DEPLOYMENT", 
                  "No-Harm Check (Recall > Champion)\nPromote Model -> Production", 
                  "#abebc6")

    # Loop back
    ax.annotate("", xy=(2.0, 4.0), xytext=(12.5, 1.25),
                arrowprops=dict(arrowstyle="->", lw=1.0, color='#aaaaaa', connectionstyle="arc3,rad=-0.2", linestyle="dashed"))
    ax.text(7, 2.5, "Model Updated (Cycle complete)", ha='center', fontsize=8, color='#888888', style='italic')

    plt.tight_layout()
    output_path = "d:/multi/auto_effi/workflow_diagram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Workflow saved to {output_path}")

if __name__ == "__main__":
    create_workflow_diagram()
