#!/usr/bin/env python3
"""
Generate All Publication Figures for ConvNeXt Paper
Complete figure generation with LOEO validation results

Author: Earthquake Prediction Research Team
Date: 6 February 2026
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("publication_convnext/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOEO_RESULTS_DIR = Path("loeo_convnext_results")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 300

print("=" * 70)
print("GENERATING CONVNEXT PUBLICATION FIGURES")
print("=" * 70)


def load_loeo_results():
    """Load LOEO validation results"""
    results_file = LOEO_RESULTS_DIR / "loeo_convnext_final_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def plot_loeo_per_fold_accuracy(results, output_path):
    """Figure 1: LOEO Per-Fold Accuracy Bar Chart"""
    if results is None:
        print("No LOEO results available")
        return
    
    folds = [r['fold'] for r in results['per_fold_results']]
    mag_acc = [r['magnitude_accuracy'] for r in results['per_fold_results']]
    azi_acc = [r['azimuth_accuracy'] for r in results['per_fold_results']]
    combined = [r['combined_accuracy'] for r in results['per_fold_results']]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    x = np.arange(len(folds))
    width = 0.6
    
    # Magnitude Accuracy
    colors_mag = ['#e74c3c' if v < 96 else '#2ecc71' for v in mag_acc]
    bars1 = axes[0].bar(x, mag_acc, width, color=colors_mag, edgecolor='black', linewidth=1)
    axes[0].axhline(y=results['magnitude_accuracy']['mean'], color='blue', linestyle='--', 
                    linewidth=2, label=f"Mean: {results['magnitude_accuracy']['mean']:.2f}%")
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('(a) Magnitude Classification Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'F{f}' for f in folds])
    axes[0].set_ylim([90, 100])
    axes[0].legend(loc='lower right')
    for bar, val in zip(bars1, mag_acc):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')

    # Azimuth Accuracy
    colors_azi = ['#e74c3c' if v < 65 else '#2ecc71' for v in azi_acc]
    bars2 = axes[1].bar(x, azi_acc, width, color=colors_azi, edgecolor='black', linewidth=1)
    axes[1].axhline(y=results['azimuth_accuracy']['mean'], color='blue', linestyle='--',
                    linewidth=2, label=f"Mean: {results['azimuth_accuracy']['mean']:.2f}%")
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('(b) Azimuth Classification Accuracy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'F{f}' for f in folds])
    axes[1].set_ylim([50, 90])
    axes[1].legend(loc='lower right')
    for bar, val in zip(bars2, azi_acc):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')
    
    # Combined Accuracy
    colors_comb = ['#e74c3c' if v < 80 else '#2ecc71' for v in combined]
    bars3 = axes[2].bar(x, combined, width, color=colors_comb, edgecolor='black', linewidth=1)
    mean_combined = np.mean(combined)
    axes[2].axhline(y=mean_combined, color='blue', linestyle='--',
                    linewidth=2, label=f"Mean: {mean_combined:.2f}%")
    axes[2].set_xlabel('Fold')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('(c) Combined Accuracy')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'F{f}' for f in folds])
    axes[2].set_ylim([70, 95])
    axes[2].legend(loc='lower right')
    for bar, val in zip(bars3, combined):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')
    
    plt.suptitle('ConvNeXt-Tiny LOEO 10-Fold Cross-Validation Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_model_comparison(results, output_path):
    """Figure 2: Model Comparison Chart"""
    models = ['VGG16', 'EfficientNet-B0', 'ConvNeXt-Tiny']
    
    # Get ConvNeXt results
    if results:
        convnext_mag = results['magnitude_accuracy']['mean']
        convnext_azi = results['azimuth_accuracy']['mean']
    else:
        convnext_mag = 97.53
        convnext_azi = 69.30
    
    mag_acc = [98.68, 97.53, convnext_mag]
    azi_acc = [54.93, 69.51, convnext_azi]
    params = [138, 5.3, 28.6]
    model_size = [528, 20, 112]  # MB
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # Magnitude Accuracy
    bars1 = axes[0, 0].bar(models, mag_acc, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('(a) Magnitude Classification (LOEO)')
    axes[0, 0].set_ylim([50, 105])
    for bar, val in zip(bars1, mag_acc):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Azimuth Accuracy
    bars2 = axes[0, 1].bar(models, azi_acc, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('(b) Azimuth Classification (LOEO)')
    axes[0, 1].set_ylim([0, 105])
    for bar, val in zip(bars2, azi_acc):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')

    # Parameters
    bars3 = axes[1, 0].bar(models, params, color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('Parameters (Millions)')
    axes[1, 0].set_title('(c) Model Parameters')
    for bar, val in zip(bars3, params):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                       f'{val}M', ha='center', fontsize=11, fontweight='bold')
    
    # Model Size
    bars4 = axes[1, 1].bar(models, model_size, color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('Model Size (MB)')
    axes[1, 1].set_title('(d) Storage Requirements')
    for bar, val in zip(bars4, model_size):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                       f'{val} MB', ha='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('Model Comparison: VGG16 vs EfficientNet-B0 vs ConvNeXt-Tiny', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_architecture_diagram(output_path):
    """Figure 3: ConvNeXt Architecture Diagram"""
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Colors
    input_color = '#95a5a6'
    stem_color = '#3498db'
    stage_colors = ['#27ae60', '#f39c12', '#e74c3c', '#9b59b6']
    head_color = '#1abc9c'
    output_color = '#34495e'

    # Input
    ax.add_patch(plt.Rectangle((0.3, 3), 1.2, 2, facecolor=input_color, edgecolor='black', linewidth=2))
    ax.text(0.9, 4, 'Input\n224×224×3', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Stem
    ax.add_patch(plt.Rectangle((2, 3), 1.5, 2, facecolor=stem_color, edgecolor='black', linewidth=2))
    ax.text(2.75, 4, 'Stem\n4×4 conv\n96 ch', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.annotate('', xy=(1.9, 4), xytext=(1.6, 4), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Stages
    stage_info = [
        ('Stage 1', '3 blocks\n96 ch', 4),
        ('Stage 2', '3 blocks\n192 ch', 6.5),
        ('Stage 3', '9 blocks\n384 ch', 9),
        ('Stage 4', '3 blocks\n768 ch', 11.5),
    ]
    
    for i, (name, info, x) in enumerate(stage_info):
        ax.add_patch(plt.Rectangle((x, 2.5), 2, 3, facecolor=stage_colors[i], edgecolor='black', linewidth=2))
        ax.text(x+1, 4, f'{name}\n{info}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        ax.annotate('', xy=(x-0.1, 4), xytext=(x-0.4, 4), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # GAP
    ax.add_patch(plt.Circle((14.2, 4), 0.4, facecolor='#bdc3c7', edgecolor='black', linewidth=2))
    ax.text(14.2, 4, 'GAP', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.annotate('', xy=(13.7, 4), xytext=(13.6, 4), arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Heads
    ax.add_patch(plt.Rectangle((15, 5), 2.5, 1.8, facecolor=head_color, edgecolor='black', linewidth=2))
    ax.text(16.25, 5.9, 'Magnitude Head\n512→4 classes', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    ax.add_patch(plt.Rectangle((15, 1.2), 2.5, 1.8, facecolor=head_color, edgecolor='black', linewidth=2))
    ax.text(16.25, 2.1, 'Azimuth Head\n512→9 classes', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    ax.annotate('', xy=(14.9, 5.9), xytext=(14.7, 4.3), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(14.9, 2.1), xytext=(14.7, 3.7), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Title and info
    ax.text(9, 7.5, 'ConvNeXt-Tiny Multi-Task Architecture for Earthquake Precursor Detection',
           ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(9, 0.5, 'Total Parameters: 28.6M | Pretrained: ImageNet-1K | Optimizer: AdamW | LR: 1e-4',
           ha='center', va='center', fontsize=10, style='italic')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_loeo_boxplot(results, output_path):
    """Figure 4: LOEO Results Box Plot"""
    if results is None:
        return
    
    mag_acc = [r['magnitude_accuracy'] for r in results['per_fold_results']]
    azi_acc = [r['azimuth_accuracy'] for r in results['per_fold_results']]
    combined = [r['combined_accuracy'] for r in results['per_fold_results']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [mag_acc, azi_acc, combined]
    labels = ['Magnitude\nAccuracy', 'Azimuth\nAccuracy', 'Combined\nAccuracy']
    colors = ['#2ecc71', '#3498db', '#9b59b6']

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(i+1, 0.04, len(d))
        ax.scatter(x, d, alpha=0.6, color='black', s=50, zorder=3)
    
    # Add mean markers
    means = [np.mean(d) for d in data]
    ax.scatter([1, 2, 3], means, marker='D', color='red', s=100, zorder=4, label='Mean')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('ConvNeXt-Tiny LOEO 10-Fold Cross-Validation Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Magnitude: {means[0]:.2f}% ± {np.std(mag_acc):.2f}%\n"
    stats_text += f"Azimuth: {means[1]:.2f}% ± {np.std(azi_acc):.2f}%\n"
    stats_text += f"Combined: {means[2]:.2f}% ± {np.std(combined):.2f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_loeo_comparison_efficientnet(results, output_path):
    """Figure 5: ConvNeXt vs EfficientNet LOEO Comparison"""
    if results is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Data
    models = ['EfficientNet-B0', 'ConvNeXt-Tiny']
    mag_means = [97.53, results['magnitude_accuracy']['mean']]
    mag_stds = [0.96, results['magnitude_accuracy']['std']]
    azi_means = [69.51, results['azimuth_accuracy']['mean']]
    azi_stds = [5.65, results['azimuth_accuracy']['std']]
    
    x = np.arange(len(models))
    width = 0.5
    colors = ['#2ecc71', '#e74c3c']
    
    # Magnitude
    bars1 = axes[0].bar(x, mag_means, width, yerr=mag_stds, capsize=10, 
                        color=colors, edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('(a) Magnitude Classification (LOEO)', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=11)
    axes[0].set_ylim([90, 102])
    for bar, mean, std in zip(bars1, mag_means, mag_stds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                    f'{mean:.2f}%\n±{std:.2f}%', ha='center', fontsize=10, fontweight='bold')

    # Azimuth
    bars2 = axes[1].bar(x, azi_means, width, yerr=azi_stds, capsize=10,
                        color=colors, edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('(b) Azimuth Classification (LOEO)', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=11)
    axes[1].set_ylim([50, 85])
    for bar, mean, std in zip(bars2, azi_means, azi_stds):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                    f'{mean:.2f}%\n±{std:.2f}%', ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('LOEO Cross-Validation: EfficientNet-B0 vs ConvNeXt-Tiny', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_fold_heatmap(results, output_path):
    """Figure 6: Per-Fold Results Heatmap"""
    if results is None:
        return
    
    folds = [f"Fold {r['fold']}" for r in results['per_fold_results']]
    metrics = ['Magnitude', 'Azimuth', 'Combined']
    
    data = np.array([
        [r['magnitude_accuracy'] for r in results['per_fold_results']],
        [r['azimuth_accuracy'] for r in results['per_fold_results']],
        [r['combined_accuracy'] for r in results['per_fold_results']]
    ])
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=55, vmax=100)

    ax.set_xticks(np.arange(len(folds)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(folds, fontsize=10)
    ax.set_yticklabels(metrics, fontsize=11)
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(folds)):
            text = ax.text(j, i, f'{data[i, j]:.1f}%',
                          ha='center', va='center', color='black', fontsize=9, fontweight='bold')
    
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va='bottom', fontsize=11)
    
    ax.set_title('ConvNeXt-Tiny LOEO Per-Fold Performance Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_sample_distribution(results, output_path):
    """Figure 7: Sample Distribution per Fold"""
    if results is None:
        return
    
    folds = [r['fold'] for r in results['per_fold_results']]
    train_samples = [r['n_train_samples'] for r in results['per_fold_results']]
    test_samples = [r['n_test_samples'] for r in results['per_fold_results']]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(len(folds))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_samples, width, label='Training', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, test_samples, width, label='Test', color='#e74c3c', edgecolor='black')

    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('LOEO Cross-Validation Sample Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
               f'{int(bar.get_height())}', ha='center', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               f'{int(bar.get_height())}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_summary_table(results, output_path):
    """Create summary table as image"""
    if results is None:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Table data
    headers = ['Metric', 'Mean', 'Std Dev', 'Min', 'Max', 'Best Fold']
    
    mag_best = max(results['per_fold_results'], key=lambda x: x['magnitude_accuracy'])
    azi_best = max(results['per_fold_results'], key=lambda x: x['azimuth_accuracy'])
    combined = [r['combined_accuracy'] for r in results['per_fold_results']]
    comb_best = results['per_fold_results'][np.argmax(combined)]

    data = [
        ['Magnitude Accuracy', f"{results['magnitude_accuracy']['mean']:.2f}%", 
         f"±{results['magnitude_accuracy']['std']:.2f}%",
         f"{results['magnitude_accuracy']['min']:.2f}%", 
         f"{results['magnitude_accuracy']['max']:.2f}%",
         f"Fold {mag_best['fold']}"],
        ['Azimuth Accuracy', f"{results['azimuth_accuracy']['mean']:.2f}%",
         f"±{results['azimuth_accuracy']['std']:.2f}%",
         f"{results['azimuth_accuracy']['min']:.2f}%",
         f"{results['azimuth_accuracy']['max']:.2f}%",
         f"Fold {azi_best['fold']}"],
        ['Combined Accuracy', f"{np.mean(combined):.2f}%",
         f"±{np.std(combined):.2f}%",
         f"{np.min(combined):.2f}%",
         f"{np.max(combined):.2f}%",
         f"Fold {comb_best['fold']}"],
    ]
    
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Style rows
    for i in range(1, len(data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax.set_title('ConvNeXt-Tiny LOEO 10-Fold Cross-Validation Summary', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    """Main function to generate all figures"""
    print("\nLoading LOEO results...")
    results = load_loeo_results()
    
    if results:
        print(f"✓ Loaded results for {results['model']}")
        print(f"  Magnitude: {results['magnitude_accuracy']['mean']:.2f}% ± {results['magnitude_accuracy']['std']:.2f}%")
        print(f"  Azimuth: {results['azimuth_accuracy']['mean']:.2f}% ± {results['azimuth_accuracy']['std']:.2f}%")
    else:
        print("✗ No LOEO results found!")
        return
    
    print("\n" + "-" * 50)
    print("Generating figures...")
    print("-" * 50)
    
    # Generate all figures
    print("\n1. LOEO Per-Fold Accuracy...")
    plot_loeo_per_fold_accuracy(results, OUTPUT_DIR / "fig1_loeo_per_fold_accuracy.png")
    
    print("\n2. Model Comparison...")
    plot_model_comparison(results, OUTPUT_DIR / "fig2_model_comparison.png")
    
    print("\n3. Architecture Diagram...")
    plot_architecture_diagram(OUTPUT_DIR / "fig3_architecture_diagram.png")
    
    print("\n4. LOEO Box Plot...")
    plot_loeo_boxplot(results, OUTPUT_DIR / "fig4_loeo_boxplot.png")
    
    print("\n5. ConvNeXt vs EfficientNet Comparison...")
    plot_loeo_comparison_efficientnet(results, OUTPUT_DIR / "fig5_convnext_vs_efficientnet.png")
    
    print("\n6. Per-Fold Heatmap...")
    plot_fold_heatmap(results, OUTPUT_DIR / "fig6_fold_heatmap.png")
    
    print("\n7. Sample Distribution...")
    plot_sample_distribution(results, OUTPUT_DIR / "fig7_sample_distribution.png")
    
    print("\n8. Summary Table...")
    create_summary_table(results, OUTPUT_DIR / "fig8_summary_table.png")
    
    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Total figures generated: 8")
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
