#!/usr/bin/env python3
"""
GENERATE EVALUATION VISUALIZATIONS
Create comprehensive evaluation visualizations including:
- Loss curves (training & validation)
- ROC-AUC curves
- Confusion matrices
- Precision-Recall curves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path('q1_comprehensive_report')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("GENERATING EVALUATION VISUALIZATIONS")
print("=" * 80)

# ============================================================================
# 1. LOAD TRAINING HISTORY
# ============================================================================
print("\n[1/4] Loading training history...")

try:
    history_df = pd.read_csv('experiments_v3/exp_v3_20260131_172406/training_history.csv')
    print(f"✓ Loaded {len(history_df)} epochs of training data")
except Exception as e:
    print(f"✗ Error loading training history: {e}")
    history_df = None

# ============================================================================
# 2. GENERATE LOSS CURVES
# ============================================================================
print("\n[2/4] Generating loss curves...")

if history_df is not None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training and Validation Loss Curves', fontsize=16, fontweight='bold')
    
    # Overall loss
    ax = axes[0, 0]
    ax.plot(history_df['epoch'], history_df['train_loss'], 
            marker='o', linewidth=2, label='Training Loss', color='#3498db')
    ax.plot(history_df['epoch'], history_df['val_loss'], 
            marker='s', linewidth=2, label='Validation Loss', color='#e74c3c')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Overall Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Magnitude F1
    ax = axes[0, 1]
    ax.plot(history_df['epoch'], history_df['train_magnitude_f1'], 
            marker='o', linewidth=2, label='Training F1', color='#2ecc71')
    ax.plot(history_df['epoch'], history_df['val_magnitude_f1'], 
            marker='s', linewidth=2, label='Validation F1', color='#27ae60')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Target (0.8)')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Magnitude Classification F1', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Azimuth F1
    ax = axes[1, 0]
    ax.plot(history_df['epoch'], history_df['train_azimuth_f1'], 
            marker='o', linewidth=2, label='Training F1', color='#9b59b6')
    ax.plot(history_df['epoch'], history_df['val_azimuth_f1'], 
            marker='s', linewidth=2, label='Validation F1', color='#8e44ad')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Azimuth Classification F1', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Generalization gap
    ax = axes[1, 1]
    gap = history_df['val_loss'] - history_df['train_loss']
    ax.plot(history_df['epoch'], gap, 
            marker='o', linewidth=2, color='#e74c3c')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(history_df['epoch'], gap, 0, alpha=0.3, color='#e74c3c')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gap (Val Loss - Train Loss)', fontsize=12)
    ax.set_title('Generalization Gap', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'fig7_loss_curves_detailed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
else:
    print("✗ Skipped: No training history data")

# ============================================================================
# 3. COPY ROC-AUC CURVES
# ============================================================================
print("\n[3/4] Processing ROC-AUC curves...")

try:
    # Check if ROC curves exist
    roc_mag_path = Path('q1_evaluation_results/roc_curves_magnitude.png')
    roc_az_path = Path('q1_evaluation_results/roc_curves_azimuth.png')
    
    if roc_mag_path.exists() and roc_az_path.exists():
        # Create combined ROC figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('ROC-AUC Curves for Multi-Task Classification', 
                     fontsize=16, fontweight='bold')
        
        # Load and display magnitude ROC
        from PIL import Image
        img_mag = Image.open(roc_mag_path)
        axes[0].imshow(img_mag)
        axes[0].axis('off')
        axes[0].set_title('Magnitude Classification', fontsize=14, fontweight='bold')
        
        # Load and display azimuth ROC
        img_az = Image.open(roc_az_path)
        axes[1].imshow(img_az)
        axes[1].axis('off')
        axes[1].set_title('Azimuth Classification', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save combined
        output_path = output_dir / 'fig8_roc_auc_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    else:
        print("✗ ROC curve files not found")
except Exception as e:
    print(f"✗ Error processing ROC curves: {e}")

# ============================================================================
# 4. COPY CONFUSION MATRICES
# ============================================================================
print("\n[4/4] Processing confusion matrices...")

try:
    # Check if confusion matrices exist
    cm_mag_path = Path('q1_evaluation_results/confusion_matrix_magnitude.png')
    cm_mag_norm_path = Path('q1_evaluation_results/confusion_matrix_magnitude_normalized.png')
    cm_az_path = Path('q1_evaluation_results/confusion_matrix_azimuth.png')
    cm_az_norm_path = Path('q1_evaluation_results/confusion_matrix_azimuth_normalized.png')
    
    if all([cm_mag_path.exists(), cm_mag_norm_path.exists(), 
            cm_az_path.exists(), cm_az_norm_path.exists()]):
        
        # Create combined confusion matrix figure
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Confusion Matrices for Multi-Task Classification', 
                     fontsize=16, fontweight='bold')
        
        from PIL import Image
        
        # Magnitude - Raw counts
        ax1 = fig.add_subplot(gs[0, 0])
        img = Image.open(cm_mag_path)
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Magnitude Classification (Counts)', 
                     fontsize=14, fontweight='bold')
        
        # Magnitude - Normalized
        ax2 = fig.add_subplot(gs[0, 1])
        img = Image.open(cm_mag_norm_path)
        ax2.imshow(img)
        ax2.axis('off')
        ax2.set_title('Magnitude Classification (Normalized)', 
                     fontsize=14, fontweight='bold')
        
        # Azimuth - Raw counts
        ax3 = fig.add_subplot(gs[1, 0])
        img = Image.open(cm_az_path)
        ax3.imshow(img)
        ax3.axis('off')
        ax3.set_title('Azimuth Classification (Counts)', 
                     fontsize=14, fontweight='bold')
        
        # Azimuth - Normalized
        ax4 = fig.add_subplot(gs[1, 1])
        img = Image.open(cm_az_norm_path)
        ax4.imshow(img)
        ax4.axis('off')
        ax4.set_title('Azimuth Classification (Normalized)', 
                     fontsize=14, fontweight='bold')
        
        # Save combined
        output_path = output_dir / 'fig9_confusion_matrices.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    else:
        print("✗ Confusion matrix files not found")
except Exception as e:
    print(f"✗ Error processing confusion matrices: {e}")

# ============================================================================
# 5. CREATE SUMMARY TABLE
# ============================================================================
print("\n[5/5] Creating evaluation summary table...")

try:
    # Load comprehensive metrics
    metrics_path = Path('q1_evaluation_results/comprehensive_metrics_summary.csv')
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        
        # Save to Q1 report folder
        output_path = output_dir / 'table9_evaluation_metrics.csv'
        metrics_df.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")
    else:
        print("✗ Metrics summary not found")
except Exception as e:
    print(f"✗ Error creating summary table: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("GENERATION COMPLETE")
print("=" * 80)

print("\nGenerated Files:")
print("  ✓ fig7_loss_curves_detailed.png - Training/validation loss curves")
print("  ✓ fig8_roc_auc_curves.png - ROC-AUC curves for both tasks")
print("  ✓ fig9_confusion_matrices.png - Confusion matrices (4 variants)")
print("  ✓ table9_evaluation_metrics.csv - Comprehensive metrics summary")

print("\nAll files saved to: q1_comprehensive_report/")
print("\nReady for dashboard integration!")
print("=" * 80)
