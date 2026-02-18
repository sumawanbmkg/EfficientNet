"""
Visualisasi dan Analisis Dataset Geomagnetic
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')


def plot_class_distribution(metadata_df, save_dir='visualizations'):
    """Plot distribusi kelas untuk azimuth dan magnitude"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Azimuth distribution
    if 'azimuth_class' in metadata_df.columns:
        azm_counts = metadata_df['azimuth_class'].value_counts()
        azm_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        azm_counts = azm_counts.reindex(azm_order, fill_value=0)
        
        colors_azm = plt.cm.Set3(np.linspace(0, 1, len(azm_counts)))
        bars1 = axes[0].bar(range(len(azm_counts)), azm_counts.values, color=colors_azm)
        axes[0].set_xticks(range(len(azm_counts)))
        axes[0].set_xticklabels(azm_counts.index, fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Jumlah Samples', fontsize=12, fontweight='bold')
        axes[0].set_title('Distribusi Azimuth Classes', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Magnitude distribution
    if 'magnitude_class' in metadata_df.columns:
        mag_counts = metadata_df['magnitude_class'].value_counts()
        mag_order = ['Small', 'Moderate', 'Medium', 'Large', 'Major']
        mag_counts = mag_counts.reindex(mag_order, fill_value=0)
        
        colors_mag = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(mag_counts)))
        bars2 = axes[1].bar(range(len(mag_counts)), mag_counts.values, color=colors_mag)
        axes[1].set_xticks(range(len(mag_counts)))
        axes[1].set_xticklabels(mag_counts.index, fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Jumlah Samples', fontsize=12, fontweight='bold')
        axes[1].set_title('Distribusi Magnitude Classes', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'class_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Class distribution plot saved to {save_path}")


def plot_signal_statistics(metadata_df, save_dir='visualizations'):
    """Plot statistik sinyal PC3"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # PC3 standard deviation untuk setiap komponen
    components = ['h_pc3_std', 'd_pc3_std', 'z_pc3_std']
    comp_names = ['H Component', 'D Component', 'Z Component']
    colors = ['red', 'green', 'blue']
    
    for idx, (comp, name, color) in enumerate(zip(components, comp_names, colors)):
        if comp in metadata_df.columns:
            row = idx // 2
            col = idx % 2
            
            axes[row, col].hist(metadata_df[comp], bins=30, color=color, 
                               alpha=0.7, edgecolor='black')
            axes[row, col].set_xlabel(f'{name} Std (nT)', fontsize=11, fontweight='bold')
            axes[row, col].set_ylabel('Frequency', fontsize=11, fontweight='bold')
            axes[row, col].set_title(f'{name} PC3 Std Distribution', 
                                    fontsize=12, fontweight='bold')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = metadata_df[comp].mean()
            std_val = metadata_df[comp].std()
            axes[row, col].axvline(mean_val, color='black', linestyle='--', 
                                  linewidth=2, label=f'Mean: {mean_val:.3f}')
            axes[row, col].legend(fontsize=10)
    
    # Magnetic storm detection
    if 'is_magnetic_storm' in metadata_df.columns:
        storm_counts = metadata_df['is_magnetic_storm'].value_counts()
        
        # Create labels based on what's actually in the data
        labels = []
        colors = []
        if False in storm_counts.index:
            labels.append('No Storm')
            colors.append('lightgreen')
        if True in storm_counts.index:
            labels.append('Storm')
            colors.append('red')
        
        axes[1, 1].pie(storm_counts.values, 
                      labels=labels,
                      autopct='%1.1f%%',
                      colors=colors,
                      startangle=90)
        axes[1, 1].set_title('Magnetic Storm Detection', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'signal_statistics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Signal statistics plot saved to {save_path}")


def plot_azimuth_magnitude_heatmap(metadata_df, save_dir='visualizations'):
    """Plot heatmap azimuth vs magnitude"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if 'azimuth_class' not in metadata_df.columns or 'magnitude_class' not in metadata_df.columns:
        logger.warning("Azimuth or magnitude class not found, skipping heatmap")
        return
    
    # Create contingency table
    contingency = pd.crosstab(
        metadata_df['azimuth_class'], 
        metadata_df['magnitude_class']
    )
    
    # Reorder
    azm_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    mag_order = ['Small', 'Moderate', 'Medium', 'Large', 'Major']
    
    contingency = contingency.reindex(index=azm_order, columns=mag_order, fill_value=0)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd', 
               cbar_kws={'label': 'Count'}, linewidths=0.5, ax=ax)
    
    ax.set_xlabel('Magnitude Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Azimuth Class', fontsize=12, fontweight='bold')
    ax.set_title('Azimuth vs Magnitude Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'azimuth_magnitude_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Azimuth-Magnitude heatmap saved to {save_path}")


def plot_station_distribution(metadata_df, save_dir='visualizations'):
    """Plot distribusi per stasiun"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if 'station' not in metadata_df.columns:
        logger.warning("Station column not found")
        return
    
    station_counts = metadata_df['station'].value_counts()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(station_counts)))
    bars = ax.bar(range(len(station_counts)), station_counts.values, color=colors)
    ax.set_xticks(range(len(station_counts)))
    ax.set_xticklabels(station_counts.index, fontsize=11, fontweight='bold', rotation=45)
    ax.set_ylabel('Jumlah Samples', fontsize=12, fontweight='bold')
    ax.set_title('Distribusi Data per Stasiun', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'station_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Station distribution plot saved to {save_path}")


def generate_summary_report(metadata_df, save_dir='visualizations'):
    """Generate text summary report"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    report_path = Path(save_dir) / 'dataset_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("GEOMAGNETIC DATASET ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Basic info
        f.write(f"Total Samples: {len(metadata_df)}\n")
        f.write(f"Date Range: {metadata_df['date'].min()} to {metadata_df['date'].max()}\n\n")
        
        # Station info
        if 'station' in metadata_df.columns:
            f.write(f"Number of Stations: {metadata_df['station'].nunique()}\n")
            f.write(f"Stations: {', '.join(metadata_df['station'].unique())}\n\n")
        
        # Azimuth distribution
        if 'azimuth_class' in metadata_df.columns:
            f.write("AZIMUTH DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            azm_counts = metadata_df['azimuth_class'].value_counts()
            for azm, count in azm_counts.items():
                pct = count / len(metadata_df) * 100
                f.write(f"  {azm:8s}: {count:4d} samples ({pct:5.1f}%)\n")
            f.write("\n")
        
        # Magnitude distribution
        if 'magnitude_class' in metadata_df.columns:
            f.write("MAGNITUDE DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            mag_counts = metadata_df['magnitude_class'].value_counts()
            for mag, count in mag_counts.items():
                pct = count / len(metadata_df) * 100
                f.write(f"  {mag:10s}: {count:4d} samples ({pct:5.1f}%)\n")
            f.write("\n")
        
        # Signal statistics
        f.write("SIGNAL STATISTICS (PC3 Filtered):\n")
        f.write("-" * 40 + "\n")
        
        for comp, name in [('h_pc3_std', 'H'), ('d_pc3_std', 'D'), ('z_pc3_std', 'Z')]:
            if comp in metadata_df.columns:
                mean_val = metadata_df[comp].mean()
                std_val = metadata_df[comp].std()
                min_val = metadata_df[comp].min()
                max_val = metadata_df[comp].max()
                f.write(f"  {name} Component:\n")
                f.write(f"    Mean: {mean_val:.4f} nT\n")
                f.write(f"    Std:  {std_val:.4f} nT\n")
                f.write(f"    Min:  {min_val:.4f} nT\n")
                f.write(f"    Max:  {max_val:.4f} nT\n")
        
        # Magnetic storm
        if 'is_magnetic_storm' in metadata_df.columns:
            n_storms = metadata_df['is_magnetic_storm'].sum()
            pct_storms = n_storms / len(metadata_df) * 100
            f.write(f"\nMagnetic Storms Detected: {n_storms} ({pct_storms:.1f}%)\n")
    
    logger.info(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Geomagnetic Dataset')
    parser.add_argument('--metadata', default='dataset_spectrogram/metadata/dataset_metadata.csv',
                       help='Path ke metadata CSV')
    parser.add_argument('--output-dir', default='visualizations',
                       help='Output directory untuk visualisasi')
    
    args = parser.parse_args()
    
    # Load metadata
    logger.info(f"Loading metadata from {args.metadata}")
    metadata_df = pd.read_csv(args.metadata)
    logger.info(f"Loaded {len(metadata_df)} samples")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    plot_class_distribution(metadata_df, args.output_dir)
    plot_signal_statistics(metadata_df, args.output_dir)
    plot_azimuth_magnitude_heatmap(metadata_df, args.output_dir)
    plot_station_distribution(metadata_df, args.output_dir)
    generate_summary_report(metadata_df, args.output_dir)
    
    logger.info(f"\nAll visualizations saved to {args.output_dir}/")
    logger.info("Done!")


if __name__ == '__main__':
    main()
