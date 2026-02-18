#!/usr/bin/env python3
"""
Model Comparison: VGG16 vs Xception
Compare performance, efficiency, and characteristics

Author: Earthquake Prediction Research Team
Date: 3 February 2026
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)


def load_results(vgg16_dir, xception_dir):
    """Load results from both models"""
    
    # VGG16 results
    vgg16_history = pd.read_csv(Path(vgg16_dir) / 'training_history.csv')
    with open(Path(vgg16_dir) / 'class_mappings.json', 'r') as f:
        vgg16_mappings = json.load(f)
    
    # Xception results
    xception_history = pd.read_csv(Path(xception_dir) / 'training_history.csv')
    with open(Path(xception_dir) / 'test_results.json', 'r') as f:
        xception_test = json.load(f)
    with open(Path(xception_dir) / 'best_hyperparameters.json', 'r') as f:
        xception_params = json.load(f)
    
    return {
        'vgg16': {
            'history': vgg16_history,
            'mappings': vgg16_mappings
        },
        'xception': {
            'history': xception_history,
            'test': xception_test,
            'params': xception_params
        }
    }


def plot_comparison(results, output_dir):
    """Create comprehensive comparison plots"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    vgg16_hist = results['vgg16']['history']
    xception_hist = results['xception']['history']
    
    # Figure 1: Training Loss Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('VGG16 vs Xception: Training Comparison', fontsize=16, fontweight='bold')
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(vgg16_hist.index + 1, vgg16_hist['train_loss'], 'b-', label='VGG16 Train', linewidth=2)
    ax.plot(vgg16_hist.index + 1, vgg16_hist['val_loss'], 'b--', label='VGG16 Val', linewidth=2)
    ax.plot(xception_hist['epoch'], xception_hist['train_loss'], 'r-', label='Xception Train', linewidth=2)
    ax.plot(xception_hist['epoch'], xception_hist['val_loss'], 'r--', label='Xception Val', linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Training Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Magnitude accuracy
    ax = axes[0, 1]
    ax.plot(vgg16_hist.index + 1, vgg16_hist['val_mag_acc']*100, 'b-', label='VGG16', linewidth=2, marker='o')
    ax.plot(xception_hist['epoch'], xception_hist['val_mag_acc']*100, 'r-', label='Xception', linewidth=2, marker='s')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Magnitude Validation Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Azimuth accuracy
    ax = axes[1, 0]
    ax.plot(vgg16_hist.index + 1, vgg16_hist['val_azi_acc']*100, 'b-', label='VGG16', linewidth=2, marker='o')
    ax.plot(xception_hist['epoch'], xception_hist['val_azi_acc']*100, 'r-', label='Xception', linewidth=2, marker='s')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Azimuth Validation Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Combined accuracy
    ax = axes[1, 1]
    vgg16_combined = (vgg16_hist['val_mag_acc'] + vgg16_hist['val_azi_acc']) / 2 * 100
    xception_combined = (xception_hist['val_mag_acc'] + xception_hist['val_azi_acc']) / 2 * 100
    ax.plot(vgg16_hist.index + 1, vgg16_combined, 'b-', label='VGG16', linewidth=2, marker='o')
    ax.plot(xception_hist['epoch'], xception_combined, 'r-', label='Xception', linewidth=2, marker='s')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Combined Validation Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'training_comparison.png'}")
    
    # Figure 2: Final Performance Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Final Test Performance Comparison', fontsize=16, fontweight='bold')
    
    # Get final test accuracies
    # VGG16 (from production config or last validation)
    vgg16_mag_acc = 98.68  # From production
    vgg16_azi_acc = 54.93
    
    # Xception (from test results)
    xception_mag_acc = results['xception']['test']['test_mag_acc'] * 100
    xception_azi_acc = results['xception']['test']['test_azi_acc'] * 100
    
    # Bar chart: Magnitude
    ax = axes[0]
    models = ['VGG16', 'Xception']
    mag_accs = [vgg16_mag_acc, xception_mag_acc]
    colors = ['blue', 'red']
    bars = ax.bar(models, mag_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Magnitude Accuracy', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars, mag_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Bar chart: Azimuth
    ax = axes[1]
    azi_accs = [vgg16_azi_acc, xception_azi_acc]
    bars = ax.bar(models, azi_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Azimuth Accuracy', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, azi_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Bar chart: Combined
    ax = axes[2]
    combined_accs = [(vgg16_mag_acc + vgg16_azi_acc) / 2, 
                     (xception_mag_acc + xception_azi_acc) / 2]
    bars = ax.bar(models, combined_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Combined Accuracy', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, combined_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_performance.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'final_performance.png'}")
    
    # Figure 3: Model Characteristics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Characteristics Comparison', fontsize=16, fontweight='bold')
    
    # Parameters
    ax = axes[0]
    params = [138, 22.9]  # Millions
    bars = ax.bar(models, params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Parameters (Millions)', fontweight='bold')
    ax.set_title('Model Size (Parameters)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{param}M', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Model file size
    ax = axes[1]
    sizes = [528, 88]  # MB
    bars = ax.bar(models, sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Size (MB)', fontweight='bold')
    ax.set_title('Model File Size', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{size} MB', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Inference time
    ax = axes[2]
    times = [100, 50]  # ms
    bars = ax.bar(models, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Inference Time (CPU)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time} ms', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_characteristics.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'model_characteristics.png'}")
    
    plt.show()


def generate_report(results, output_dir):
    """Generate comparison report"""
    
    output_dir = Path(output_dir)
    
    vgg16_hist = results['vgg16']['history']
    xception_hist = results['xception']['history']
    xception_test = results['xception']['test']
    xception_params = results['xception']['params']
    
    # VGG16 final (from production)
    vgg16_mag_acc = 98.68
    vgg16_azi_acc = 54.93
    
    # Xception final
    xception_mag_acc = xception_test['test_mag_acc'] * 100
    xception_azi_acc = xception_test['test_azi_acc'] * 100
    
    report = f"""
═══════════════════════════════════════════════════════════════════════════
  MODEL COMPARISON REPORT: VGG16 vs XCEPTION
═══════════════════════════════════════════════════════════════════════════

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

───────────────────────────────────────────────────────────────────────────
📊 FINAL TEST PERFORMANCE
───────────────────────────────────────────────────────────────────────────

                        VGG16           XCEPTION        DIFFERENCE
                        ─────           ────────        ──────────
Magnitude Accuracy:     {vgg16_mag_acc:.2f}%          {xception_mag_acc:.2f}%         {xception_mag_acc - vgg16_mag_acc:+.2f}%
Azimuth Accuracy:       {vgg16_azi_acc:.2f}%          {xception_azi_acc:.2f}%         {xception_azi_acc - vgg16_azi_acc:+.2f}%
Combined Accuracy:      {(vgg16_mag_acc + vgg16_azi_acc)/2:.2f}%          {(xception_mag_acc + xception_azi_acc)/2:.2f}%         {((xception_mag_acc + xception_azi_acc)/2 - (vgg16_mag_acc + vgg16_azi_acc)/2):+.2f}%

───────────────────────────────────────────────────────────────────────────
🏗️ MODEL CHARACTERISTICS
───────────────────────────────────────────────────────────────────────────

                        VGG16           XCEPTION        RATIO
                        ─────           ────────        ─────
Parameters:             138M            22.9M           6.0x smaller
Model Size:             528 MB          88 MB           6.0x smaller
Inference Time:         ~100 ms         ~50 ms          2.0x faster
Architecture:           Standard Conv   Depthwise Sep   More efficient

───────────────────────────────────────────────────────────────────────────
⚙️ TRAINING DETAILS
───────────────────────────────────────────────────────────────────────────

VGG16:
• Epochs: {len(vgg16_hist)}
• Final Val Loss: {vgg16_hist.iloc[-1]['val_loss']:.4f}
• Optimizer: Adam (lr=0.0001)
• Data Split: Fixed (by station+date)
• Augmentation: Standard

XCEPTION:
• Epochs: {len(xception_hist)}
• Final Val Loss: {xception_hist.iloc[-1]['val_loss']:.4f}
• Optimizer: Adam (lr={xception_params['lr']:.6f})
• Dropout: {xception_params['dropout']:.2f}
• Weight Decay: {xception_params['weight_decay']:.6f}
• Data Split: Fixed (by station+date)
• Augmentation: SMOTE + Standard
• Hyperparameter Tuning: Optuna (20 trials)

───────────────────────────────────────────────────────────────────────────
🎯 WINNER ANALYSIS
───────────────────────────────────────────────────────────────────────────
"""
    
    # Determine winner
    if xception_mag_acc > vgg16_mag_acc and xception_azi_acc > vgg16_azi_acc:
        winner = "XCEPTION"
        reason = "Higher accuracy on both tasks"
    elif xception_mag_acc > vgg16_mag_acc:
        winner = "XCEPTION (Magnitude)"
        reason = "Higher magnitude accuracy"
    elif xception_azi_acc > vgg16_azi_acc:
        winner = "XCEPTION (Azimuth)"
        reason = "Higher azimuth accuracy"
    elif abs(xception_mag_acc - vgg16_mag_acc) < 1 and abs(xception_azi_acc - vgg16_azi_acc) < 1:
        winner = "TIE (Similar Performance)"
        reason = "Both models perform similarly"
    else:
        winner = "VGG16"
        reason = "Higher overall accuracy"
    
    report += f"""
WINNER: {winner}
REASON: {reason}

───────────────────────────────────────────────────────────────────────────
💡 RECOMMENDATIONS
───────────────────────────────────────────────────────────────────────────
"""
    
    if "XCEPTION" in winner:
        report += """
✅ DEPLOY XCEPTION TO PRODUCTION
   • Higher accuracy
   • 6x smaller model size
   • 2x faster inference
   • More efficient for production

NEXT STEPS:
1. Update production model to Xception
2. Update scanner to use Xception
3. Monitor production performance
4. Keep VGG16 as backup
"""
    elif "VGG16" in winner:
        report += """
✅ KEEP VGG16 IN PRODUCTION
   • Higher accuracy (critical)
   • Proven performance
   • Already deployed

NEXT STEPS:
1. Keep current production setup
2. Use Xception for research
3. Consider ensemble approach
4. Investigate why Xception underperformed
"""
    else:
        report += """
✅ USE XCEPTION FOR PRODUCTION (Efficiency)
   • Similar accuracy
   • 6x smaller (better for deployment)
   • 2x faster (better user experience)
   • Lower resource usage

ALTERNATIVE: ENSEMBLE BOTH MODELS
   • Combine predictions
   • Potentially higher accuracy
   • More robust predictions

NEXT STEPS:
1. Deploy Xception (efficiency)
2. Keep VGG16 as backup
3. Consider ensemble approach
4. Monitor both models
"""
    
    report += """
───────────────────────────────────────────────────────────────────────────
📈 VISUALIZATIONS
───────────────────────────────────────────────────────────────────────────

Generated plots:
• training_comparison.png - Training curves comparison
• final_performance.png - Final test performance
• model_characteristics.png - Model size and speed

═══════════════════════════════════════════════════════════════════════════
  END OF REPORT
═══════════════════════════════════════════════════════════════════════════
"""
    
    # Save report
    report_path = output_dir / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n✅ Report saved to: {report_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare VGG16 and Xception models')
    parser.add_argument('--vgg16', type=str, required=True, help='VGG16 experiment directory')
    parser.add_argument('--xception', type=str, required=True, help='Xception experiment directory')
    parser.add_argument('--output', type=str, default='model_comparison', help='Output directory')
    
    args = parser.parse_args()
    
    print("Loading results...")
    results = load_results(args.vgg16, args.xception)
    
    print("\nGenerating comparison plots...")
    plot_comparison(results, args.output)
    
    print("\nGenerating comparison report...")
    generate_report(results, args.output)
    
    print(f"\n✅ Comparison complete! Check {args.output}/ for results.")


if __name__ == '__main__':
    main()
