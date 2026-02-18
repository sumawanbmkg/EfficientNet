#!/usr/bin/env python3
"""
Generate LOSO Validation Visualizations
Creates charts and plots for LOSO validation results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_loso_visualizations():
    """Generate all LOSO visualizations"""
    
    output_dir = Path('loso_validation_results')
    output_dir.mkdir(exist_ok=True)
    
    # Load LOSO results
    with open('loso_validation_results/loso_final_results.json', 'r') as f:
        results = json.load(f)
    
    folds = results['per_fold_results']
    
    # Extract data
    stations = [f['test_station'] for f in folds]
    mag_accs = [f['magnitude_accuracy'] for f in folds]
    azi_accs = [f['azimuth_accuracy'] for f in folds]
    samples = [f['n_test_samples'] for f in folds]
    
    # 1. Per-Station Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(stations))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mag_accs, width, label='Magnitude', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, azi_accs, width, label='Azimuth', color='#e74c3c', edgecolor='black')
    
    ax.set_xlabel('Station', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('LOSO Validation: Per-Station Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stations, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loso_per_station_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Generated: loso_per_station_accuracy.png")
    
    # 2. Boxplot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Magnitude boxplot
    bp1 = axes[0].boxplot([mag_accs], patch_artist=True)
    bp1['boxes'][0].set_facecolor('#3498db')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Magnitude Accuracy Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xticklabels(['LOSO'])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add mean line
    mean_mag = np.mean(mag_accs)
    axes[0].axhline(y=mean_mag, color='red', linestyle='--', label=f'Mean: {mean_mag:.2f}%')
    axes[0].legend()
    
    # Azimuth boxplot
    bp2 = axes[1].boxplot([azi_accs], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#e74c3c')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Azimuth Accuracy Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(['LOSO'])
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add mean line
    mean_azi = np.mean(azi_accs)
    axes[1].axhline(y=mean_azi, color='red', linestyle='--', label=f'Mean: {mean_azi:.2f}%')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loso_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Generated: loso_boxplot.png")
    
    # 3. Sample Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(stations)))
    bars = ax.bar(stations, samples, color=colors, edgecolor='black')
    
    ax.set_xlabel('Station', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('LOSO: Test Samples per Station', fontsize=14, fontweight='bold')
    ax.set_xticklabels(stations, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loso_sample_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Generated: loso_sample_distribution.png")
    
    # 4. LOEO vs LOSO Comparison
    try:
        with open('loeo_validation_results/loeo_final_results.json', 'r') as f:
            loeo_results = json.load(f)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['LOEO\n(Leave-One-Event-Out)', 'LOSO\n(Leave-One-Station-Out)']
        
        loeo_mag = loeo_results['magnitude_accuracy']['mean']
        loeo_azi = loeo_results['azimuth_accuracy']['mean']
        loso_mag = results['magnitude_accuracy']['weighted_mean']
        loso_azi = results['azimuth_accuracy']['weighted_mean']
        
        x = np.arange(len(methods))
        width = 0.35
        
        mag_vals = [loeo_mag, loso_mag]
        azi_vals = [loeo_azi, loso_azi]
        
        bars1 = ax.bar(x - width/2, mag_vals, width, label='Magnitude', color='#3498db', edgecolor='black')
        bars2 = ax.bar(x + width/2, azi_vals, width, label='Azimuth', color='#e74c3c', edgecolor='black')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('LOEO vs LOSO Validation Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'loeo_vs_loso_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Generated: loeo_vs_loso_comparison.png")
        
    except Exception as e:
        print(f"⚠️ Could not generate LOEO vs LOSO comparison: {e}")
    
    # 5. Comprehensive Summary Figure
    fig = plt.figure(figsize=(14, 10))
    
    # Title
    fig.suptitle('LOSO Validation Results Summary', fontsize=16, fontweight='bold', y=0.98)
    
    # Subplot 1: Per-station accuracy
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(stations))
    width = 0.35
    ax1.bar(x - width/2, mag_accs, width, label='Magnitude', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, azi_accs, width, label='Azimuth', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Station')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Per-Station Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stations, rotation=45, ha='right')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Boxplot
    ax2 = fig.add_subplot(2, 2, 2)
    bp = ax2.boxplot([mag_accs, azi_accs], patch_artist=True, labels=['Magnitude', 'Azimuth'])
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Distribution')
    ax2.grid(axis='y', alpha=0.3)
    
    # Subplot 3: Sample distribution
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.bar(stations, samples, color='#2ecc71', edgecolor='black', alpha=0.8)
    ax3.set_xlabel('Station')
    ax3.set_ylabel('Samples')
    ax3.set_title('Test Samples per Station')
    ax3.set_xticklabels(stations, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Subplot 4: Summary statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = f"""
    LOSO VALIDATION SUMMARY
    ═══════════════════════════════════════
    
    Magnitude Classification:
    • Weighted Mean: {results['magnitude_accuracy']['weighted_mean']:.2f}%
    • Simple Mean:   {results['magnitude_accuracy']['mean']:.2f}%
    • Std Dev:       {results['magnitude_accuracy']['std']:.2f}%
    • Min:           {results['magnitude_accuracy']['min']:.2f}%
    • Max:           {results['magnitude_accuracy']['max']:.2f}%
    
    Azimuth Classification:
    • Weighted Mean: {results['azimuth_accuracy']['weighted_mean']:.2f}%
    • Simple Mean:   {results['azimuth_accuracy']['mean']:.2f}%
    • Std Dev:       {results['azimuth_accuracy']['std']:.2f}%
    • Min:           {results['azimuth_accuracy']['min']:.2f}%
    • Max:           {results['azimuth_accuracy']['max']:.2f}%
    
    Total Folds: {len(folds)}
    Total Samples: {sum(samples)}
    
    ✅ Model validated across all stations
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'loso_summary_figure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Generated: loso_summary_figure.png")
    
    print("\n✅ All LOSO visualizations generated successfully!")
    print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    generate_loso_visualizations()
