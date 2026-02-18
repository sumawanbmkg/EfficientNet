#!/usr/bin/env python3
"""
Generate All Publication Figures for ConvNeXt Paper
Including MCC Analysis Figures

Date: 6 February 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

OUTPUT_DIR = Path("publication_convnext/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
with open("loeo_convnext_results/loeo_convnext_final_results.json", 'r') as f:
    results = json.load(f)

with open("publication_convnext/MCC_ANALYSIS.json", 'r') as f:
    mcc_data = json.load(f)

print("=" * 60)
print("GENERATING ALL FIGURES WITH MCC")
print("=" * 60)

# Extract data
folds = [r['fold'] for r in results['per_fold_results']]
mag_accs = [r['magnitude_accuracy'] for r in results['per_fold_results']]
azi_accs = [r['azimuth_accuracy'] for r in results['per_fold_results']]
combined = [r['combined_accuracy'] for r in results['per_fold_results']]
samples = [r['n_test_samples'] for r in results['per_fold_results']]

# ============================================================
# FIGURE 1: LOEO Per-Fold Accuracy
# ============================================================
def fig1_loeo_per_fold():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(folds))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mag_accs, width, label='Magnitude', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, azi_accs, width, label='Azimuth', color='#3498db', edgecolor='black')
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('LOEO 10-Fold Cross-Validation: Per-Fold Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Add mean lines
    ax.axhline(y=results['magnitude_accuracy']['mean'], color='#27ae60', linestyle='--', 
               label=f"Mag Mean: {results['magnitude_accuracy']['mean']:.2f}%")
    ax.axhline(y=results['azimuth_accuracy']['mean'], color='#2980b9', linestyle='--',
               label=f"Azi Mean: {results['azimuth_accuracy']['mean']:.2f}%")
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_loeo_per_fold_accuracy.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: LOEO Per-Fold Accuracy")

# ============================================================
# FIGURE 2: Model Comparison
# ============================================================
def fig2_model_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = ['VGG16', 'EfficientNet-B0', 'ConvNeXt-Tiny']
    mag_vals = [98.68, 97.53, results['magnitude_accuracy']['mean']]
    azi_vals = [54.93, 69.51, results['azimuth_accuracy']['mean']]
    colors = ['#e74c3c', '#f39c12', '#9b59b6']
    
    # Magnitude
    bars1 = axes[0].bar(models, mag_vals, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Magnitude Classification Accuracy')
    axes[0].set_ylim(90, 100)
    for bar, val in zip(bars1, mag_vals):
        axes[0].annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    # Azimuth
    bars2 = axes[1].bar(models, azi_vals, color=colors, edgecolor='black')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Azimuth Classification Accuracy')
    axes[1].set_ylim(0, 100)
    for bar, val in zip(bars2, azi_vals):
        axes[1].annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    # Add random baseline for azimuth
    axes[1].axhline(y=11.11, color='gray', linestyle='--', label='Random (11.11%)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_model_comparison.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Model Comparison")

# ============================================================
# FIGURE 3: Architecture Diagram
# ============================================================
def fig3_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#3498db',
        'stem': '#2ecc71',
        'stage': '#9b59b6',
        'head': '#e74c3c',
        'output': '#f39c12'
    }
    
    # Input
    ax.add_patch(plt.Rectangle((0.5, 3), 1.5, 2, facecolor=colors['input'], edgecolor='black', lw=2))
    ax.text(1.25, 4, 'Input\n224×224×3', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Stem
    ax.add_patch(plt.Rectangle((2.5, 3), 1.5, 2, facecolor=colors['stem'], edgecolor='black', lw=2))
    ax.text(3.25, 4, 'Stem\n4×4 Conv\n56×56×96', ha='center', va='center', fontsize=8)
    
    # Stages
    stages = [
        ('Stage 1\n3 blocks\n56×56×96', 4.5),
        ('Stage 2\n3 blocks\n28×28×192', 6.5),
        ('Stage 3\n9 blocks\n14×14×384', 8.5),
        ('Stage 4\n3 blocks\n7×7×768', 10.5)
    ]
    for text, x in stages:
        ax.add_patch(plt.Rectangle((x, 3), 1.5, 2, facecolor=colors['stage'], edgecolor='black', lw=2))
        ax.text(x + 0.75, 4, text, ha='center', va='center', fontsize=7)
    
    # GAP + Heads
    ax.add_patch(plt.Rectangle((12.5, 4.5), 1, 1, facecolor=colors['head'], edgecolor='black', lw=2))
    ax.text(13, 5, 'GAP', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.add_patch(plt.Rectangle((12, 2), 1, 1.5, facecolor=colors['output'], edgecolor='black', lw=2))
    ax.text(12.5, 2.75, 'Mag\n(4)', ha='center', va='center', fontsize=8)
    
    ax.add_patch(plt.Rectangle((13.5, 2), 1, 1.5, facecolor=colors['output'], edgecolor='black', lw=2))
    ax.text(14, 2.75, 'Azi\n(9)', ha='center', va='center', fontsize=8)
    
    # Arrows
    arrow_y = 4
    arrows_x = [(2, 2.5), (4, 4.5), (6, 6.5), (8, 8.5), (10, 10.5), (12, 12.5)]
    for x1, x2 in arrows_x:
        ax.annotate('', xy=(x2, arrow_y), xytext=(x1, arrow_y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Arrows to heads
    ax.annotate('', xy=(12.5, 3.5), xytext=(13, 4.5), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(14, 3.5), xytext=(13, 4.5), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Title
    ax.text(7, 7.5, 'ConvNeXt-Tiny Architecture for Earthquake Precursor Detection', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(7, 6.8, 'Total Parameters: 28.6M | Pretrained: ImageNet-1K', 
            ha='center', va='center', fontsize=10)
    
    # Legend
    legend_items = [
        (colors['input'], 'Input'),
        (colors['stem'], 'Stem'),
        (colors['stage'], 'ConvNeXt Stages'),
        (colors['head'], 'Global Avg Pool'),
        (colors['output'], 'Output Heads')
    ]
    for i, (color, label) in enumerate(legend_items):
        ax.add_patch(plt.Rectangle((0.5 + i*2.5, 0.5), 0.3, 0.3, facecolor=color, edgecolor='black'))
        ax.text(0.9 + i*2.5, 0.65, label, fontsize=8, va='center')
    
    plt.savefig(OUTPUT_DIR / 'fig3_architecture_diagram.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Architecture Diagram")

# ============================================================
# FIGURE 4: LOEO Box Plot
# ============================================================
def fig4_boxplot():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = [mag_accs, azi_accs, combined]
    labels = ['Magnitude', 'Azimuth', 'Combined']
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('LOEO 10-Fold Cross-Validation Results Distribution')
    
    # Add mean markers
    means = [np.mean(d) for d in data]
    ax.scatter([1, 2, 3], means, color='red', marker='D', s=50, zorder=5, label='Mean')
    
    # Add annotations
    for i, (mean, std) in enumerate(zip(means, [np.std(d) for d in data])):
        ax.annotate(f'{mean:.2f}±{std:.2f}%', xy=(i+1, mean), xytext=(10, 10),
                   textcoords='offset points', fontsize=9)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_loeo_boxplot.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 4: LOEO Box Plot")

# ============================================================
# FIGURE 5: ConvNeXt vs EfficientNet
# ============================================================
def fig5_convnext_vs_efficientnet():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    metrics = ['Magnitude\nAccuracy', 'Azimuth\nAccuracy', 'Parameters\n(M)', 'Model Size\n(MB)']
    efficientnet = [97.53, 69.51, 5.3, 20]
    convnext = [results['magnitude_accuracy']['mean'], results['azimuth_accuracy']['mean'], 28.6, 112]
    
    # Accuracy comparison
    x = np.arange(2)
    width = 0.35
    
    axes[0].bar(x - width/2, [efficientnet[0], efficientnet[1]], width, label='EfficientNet-B0', color='#f39c12')
    axes[0].bar(x + width/2, [convnext[0], convnext[1]], width, label='ConvNeXt-Tiny', color='#9b59b6')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Magnitude', 'Azimuth'])
    axes[0].legend()
    axes[0].set_ylim(0, 105)
    
    # Model specs
    axes[1].bar(x - width/2, [efficientnet[2], efficientnet[3]], width, label='EfficientNet-B0', color='#f39c12')
    axes[1].bar(x + width/2, [convnext[2], convnext[3]], width, label='ConvNeXt-Tiny', color='#9b59b6')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Model Specifications')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Parameters (M)', 'Size (MB)'])
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_convnext_vs_efficientnet.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: ConvNeXt vs EfficientNet")

# ============================================================
# FIGURE 6: Per-Fold Heatmap
# ============================================================
def fig6_heatmap():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = np.array([mag_accs, azi_accs, combined])
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=50, vmax=100)
    
    ax.set_xticks(np.arange(len(folds)))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.set_yticklabels(['Magnitude', 'Azimuth', 'Combined'])
    
    # Add text annotations
    for i in range(3):
        for j in range(len(folds)):
            text = ax.text(j, i, f'{data[i, j]:.1f}%', ha='center', va='center', 
                          color='black', fontsize=9)
    
    ax.set_title('LOEO Per-Fold Performance Heatmap')
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_fold_heatmap.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 6: Per-Fold Heatmap")

# ============================================================
# FIGURE 7: Sample Distribution
# ============================================================
def fig7_sample_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(folds, samples, color='#3498db', edgecolor='black')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Number of Test Samples')
    ax.set_title('Sample Distribution per LOEO Fold')
    
    for bar, val in zip(bars, samples):
        ax.annotate(str(val), xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    ax.axhline(y=np.mean(samples), color='red', linestyle='--', label=f'Mean: {np.mean(samples):.0f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_sample_distribution.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 7: Sample Distribution")

# ============================================================
# FIGURE 8: Summary Table
# ============================================================
def fig8_summary_table():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Mean', 'Std Dev', 'Min', 'Max'],
        ['Magnitude Accuracy', f"{results['magnitude_accuracy']['mean']:.2f}%", 
         f"±{results['magnitude_accuracy']['std']:.2f}%",
         f"{results['magnitude_accuracy']['min']:.2f}%",
         f"{results['magnitude_accuracy']['max']:.2f}%"],
        ['Azimuth Accuracy', f"{results['azimuth_accuracy']['mean']:.2f}%",
         f"±{results['azimuth_accuracy']['std']:.2f}%",
         f"{results['azimuth_accuracy']['min']:.2f}%",
         f"{results['azimuth_accuracy']['max']:.2f}%"],
        ['Combined Accuracy', f"{np.mean(combined):.2f}%",
         f"±{np.std(combined):.2f}%",
         f"{np.min(combined):.2f}%",
         f"{np.max(combined):.2f}%"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Header styling
    for j in range(5):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title('LOEO 10-Fold Cross-Validation Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(OUTPUT_DIR / 'fig8_summary_table.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 8: Summary Table")

# ============================================================
# FIGURE 9: MCC Analysis (NEW!)
# ============================================================
def fig9_mcc_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Accuracy vs Random Baseline
    tasks = ['Magnitude\n(4 classes)', 'Azimuth\n(9 classes)']
    model_acc = [results['magnitude_accuracy']['mean'], results['azimuth_accuracy']['mean']]
    random_baseline = [25.0, 11.11]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, random_baseline, width, label='Random Baseline', color='#95a5a6', edgecolor='black')
    bars2 = axes[0].bar(x + width/2, model_acc, width, label='ConvNeXt Model', color='#27ae60', edgecolor='black')
    
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Model Accuracy vs Random Baseline', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks)
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(0, 110)
    
    # Add improvement annotations
    for i, (rand, model) in enumerate(zip(random_baseline, model_acc)):
        improvement = model / rand
        axes[0].annotate(f'{improvement:.1f}x\nimprovement', 
                        xy=(i + width/2, model + 3), ha='center', fontsize=10, fontweight='bold', color='#27ae60')
        axes[0].annotate(f'{rand:.1f}%', xy=(i - width/2, rand + 2), ha='center', fontsize=9)
        axes[0].annotate(f'{model:.1f}%', xy=(i + width/2, model + 2), ha='center', fontsize=9)
    
    # Right: MCC Values
    mcc_values = [mcc_data['magnitude']['estimated_mcc'], mcc_data['azimuth']['estimated_mcc']]
    colors = ['#2ecc71', '#3498db']
    
    bars3 = axes[1].bar(tasks, mcc_values, color=colors, edgecolor='black', width=0.5)
    axes[1].set_ylabel('Matthews Correlation Coefficient (MCC)')
    axes[1].set_title('MCC Performance Metric', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    
    # Add reference lines
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Random Guessing (MCC=0)')
    axes[1].axhline(y=1, color='green', linestyle='--', linewidth=2, label='Perfect Prediction (MCC=1)')
    
    for bar, val in zip(bars3, mcc_values):
        axes[1].annotate(f'MCC = {val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val + 0.03),
                        ha='center', fontsize=11, fontweight='bold')
    
    axes[1].legend(loc='lower right')
    
    plt.suptitle('Matthews Correlation Coefficient (MCC) Analysis\nConvNeXt-Tiny for Earthquake Precursor Detection', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig9_mcc_analysis.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 9: MCC Analysis")

# ============================================================
# FIGURE 10: MCC Comparison with Random (NEW!)
# ============================================================
def fig10_mcc_improvement():
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Data
    categories = ['Magnitude\n(4 classes)', 'Azimuth\n(9 classes)']
    random_acc = [25.0, 11.11]
    model_acc = [results['magnitude_accuracy']['mean'], results['azimuth_accuracy']['mean']]
    mcc_vals = [mcc_data['magnitude']['estimated_mcc'], mcc_data['azimuth']['estimated_mcc']]
    improvement = [mcc_data['magnitude']['improvement_factor'], mcc_data['azimuth']['improvement_factor']]
    
    # Create grouped bar chart
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax.bar(x - width, random_acc, width, label='Random Baseline', color='#e74c3c', alpha=0.7)
    bars2 = ax.bar(x, model_acc, width, label='Model Accuracy', color='#27ae60', alpha=0.9)
    bars3 = ax.bar(x + width, [m*100 for m in mcc_vals], width, label='MCC × 100', color='#3498db', alpha=0.9)
    
    ax.set_ylabel('Value (%)')
    ax.set_title('Performance Metrics: Accuracy and MCC Analysis\nConvNeXt-Tiny for Earthquake Precursor Detection', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 120)
    
    # Add annotations
    for i, (rand, model, mcc, imp) in enumerate(zip(random_acc, model_acc, mcc_vals, improvement)):
        # Improvement factor
        ax.annotate(f'{imp:.1f}x better\nthan random', 
                   xy=(i, model + 5), ha='center', fontsize=9, fontweight='bold', color='#27ae60')
        ax.annotate(f'MCC={mcc:.2f}', 
                   xy=(i + width, mcc*100 + 3), ha='center', fontsize=9, fontweight='bold', color='#3498db')
    
    # Add text box with key findings
    textstr = f"""Key Findings:
• Azimuth: {model_acc[1]:.1f}% accuracy = {improvement[1]:.1f}× random baseline
• Azimuth MCC: {mcc_vals[1]:.2f} (substantial predictive capability)
• MCC=0 means random, MCC=1 means perfect"""
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig10_mcc_improvement.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 10: MCC Improvement")

# ============================================================
# FIGURE 11: MCC Interpretation Guide (NEW!)
# ============================================================
def fig11_mcc_interpretation():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # MCC scale visualization
    mcc_scale = np.linspace(-1, 1, 100)
    colors_scale = plt.cm.RdYlGn((mcc_scale + 1) / 2)
    
    for i, (mcc, color) in enumerate(zip(mcc_scale, colors_scale)):
        ax.axvline(x=mcc, color=color, linewidth=3)
    
    # Add markers for our results
    ax.axvline(x=0, color='black', linewidth=3, linestyle='--', label='Random Guessing')
    ax.axvline(x=mcc_data['azimuth']['estimated_mcc'], color='blue', linewidth=4, 
               label=f"Azimuth MCC: {mcc_data['azimuth']['estimated_mcc']:.2f}")
    ax.axvline(x=mcc_data['magnitude']['estimated_mcc'], color='green', linewidth=4,
               label=f"Magnitude MCC: {mcc_data['magnitude']['estimated_mcc']:.2f}")
    
    # Annotations
    ax.annotate('Random\nGuessing', xy=(0, 0.8), xytext=(0, 0.9), ha='center', fontsize=10, fontweight='bold')
    ax.annotate('Perfect\nPrediction', xy=(1, 0.8), xytext=(1, 0.9), ha='center', fontsize=10, fontweight='bold')
    ax.annotate('Total\nDisagreement', xy=(-1, 0.8), xytext=(-1, 0.9), ha='center', fontsize=10, fontweight='bold')
    
    # Add interpretation zones
    ax.axvspan(-1, -0.3, alpha=0.2, color='red', label='Poor')
    ax.axvspan(-0.3, 0.3, alpha=0.2, color='yellow', label='Weak')
    ax.axvspan(0.3, 0.7, alpha=0.2, color='lightgreen', label='Moderate')
    ax.axvspan(0.7, 1, alpha=0.2, color='green', label='Strong')
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Matthews Correlation Coefficient (MCC)', fontsize=12)
    ax.set_title('MCC Interpretation Scale with ConvNeXt Results', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig11_mcc_interpretation.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 11: MCC Interpretation")

# ============================================================
# FIGURE 12: Complete Results Summary with MCC (NEW!)
# ============================================================
def fig12_complete_summary():
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top Left: Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    tasks = ['Magnitude', 'Azimuth']
    model_acc = [results['magnitude_accuracy']['mean'], results['azimuth_accuracy']['mean']]
    random_acc = [25.0, 11.11]
    
    x = np.arange(len(tasks))
    width = 0.35
    ax1.bar(x - width/2, random_acc, width, label='Random', color='#e74c3c', alpha=0.7)
    ax1.bar(x + width/2, model_acc, width, label='ConvNeXt', color='#27ae60')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy vs Random Baseline')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks)
    ax1.legend()
    ax1.set_ylim(0, 110)
    
    # Top Right: MCC values
    ax2 = fig.add_subplot(gs[0, 1])
    mcc_vals = [mcc_data['magnitude']['estimated_mcc'], mcc_data['azimuth']['estimated_mcc']]
    colors = ['#2ecc71', '#3498db']
    bars = ax2.bar(tasks, mcc_vals, color=colors, edgecolor='black')
    ax2.axhline(y=0, color='red', linestyle='--', label='Random (MCC=0)')
    ax2.set_ylabel('MCC')
    ax2.set_title('Matthews Correlation Coefficient')
    ax2.set_ylim(0, 1.1)
    for bar, val in zip(bars, mcc_vals):
        ax2.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val + 0.03), ha='center', fontweight='bold')
    ax2.legend()
    
    # Bottom Left: Per-fold accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(folds))
    width = 0.35
    ax3.bar(x - width/2, mag_accs, width, label='Magnitude', color='#2ecc71')
    ax3.bar(x + width/2, azi_accs, width, label='Azimuth', color='#3498db')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('LOEO Per-Fold Results')
    ax3.set_xticks(x)
    ax3.set_xticklabels(folds)
    ax3.legend()
    ax3.set_ylim(0, 105)
    
    # Bottom Right: Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    table_data = [
        ['Metric', 'Value', 'MCC'],
        ['Magnitude Acc', f"{results['magnitude_accuracy']['mean']:.2f}% ± {results['magnitude_accuracy']['std']:.2f}%", 
         f"{mcc_data['magnitude']['estimated_mcc']:.2f}"],
        ['Azimuth Acc', f"{results['azimuth_accuracy']['mean']:.2f}% ± {results['azimuth_accuracy']['std']:.2f}%",
         f"{mcc_data['azimuth']['estimated_mcc']:.2f}"],
        ['Azi Improvement', f"{mcc_data['azimuth']['improvement_factor']:.1f}x over random", '-'],
        ['Model', 'ConvNeXt-Tiny (28.6M params)', '-']
    ]
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.45, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for j in range(3):
        table[(0, j)].set_facecolor('#3498db')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('Results Summary', fontsize=12, fontweight='bold')
    
    plt.suptitle('ConvNeXt-Tiny: Complete LOEO Validation Results with MCC Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(OUTPUT_DIR / 'fig12_complete_summary.png', bbox_inches='tight')
    plt.close()
    print("✓ Figure 12: Complete Summary")

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\nGenerating all figures...\n")
    
    # Original figures
    fig1_loeo_per_fold()
    fig2_model_comparison()
    fig3_architecture()
    fig4_boxplot()
    fig5_convnext_vs_efficientnet()
    fig6_heatmap()
    fig7_sample_distribution()
    fig8_summary_table()
    
    # NEW MCC figures
    fig9_mcc_analysis()
    fig10_mcc_improvement()
    fig11_mcc_interpretation()
    fig12_complete_summary()
    
    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nTotal: 12 figures saved to {OUTPUT_DIR}/")
    print("\nFigures 1-8: Original publication figures")
    print("Figures 9-12: NEW MCC analysis figures")
    
    # List all files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")
