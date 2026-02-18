#!/usr/bin/env python3
"""
Check Phase 1 Training Results
Simple script to check training status and results

Author: Earthquake Prediction Research Team
Date: February 1, 2026
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def get_latest_experiment():
    """Get the latest experiment directory"""
    exp_dir = Path('experiments_v4')
    if not exp_dir.exists():
        return None
    
    experiments = sorted(exp_dir.glob('exp_v4_phase1_*_smote'))
    if not experiments:
        return None
    
    return experiments[-1]

def check_training_status():
    """Check the status of training"""
    print("🔍 CHECKING PHASE 1 TRAINING STATUS")
    print("=" * 70)
    
    # Check for latest experiment
    latest_exp = get_latest_experiment()
    if not latest_exp:
        print("❌ No experiments found")
        print("🚀 Ready to start new training")
        return None
    
    print(f"📁 Latest experiment: {latest_exp.name}")
    print(f"📂 Location: {latest_exp}")
    
    # Check for files
    config_file = latest_exp / 'config.json'
    model_file = latest_exp / 'best_model.pth'
    history_file = latest_exp / 'training_history.csv'
    
    print(f"\n📋 Files found:")
    print(f"   Config: {'✅' if config_file.exists() else '❌'}")
    print(f"   Model: {'✅' if model_file.exists() else '❌'}")
    print(f"   History: {'✅' if history_file.exists() else '❌'}")
    
    # Check training history
    if history_file.exists():
        df = pd.read_csv(history_file)
        
        print(f"\n📊 Training Progress:")
        print(f"   Epochs completed: {len(df)}/30")
        
        if len(df) > 0:
            last_row = df.iloc[-1]
            print(f"\n📈 Latest Metrics (Epoch {int(last_row['epoch']) + 1}):")
            print(f"   Train Loss: {last_row['train_loss']:.4f}")
            print(f"   Val Loss: {last_row['val_loss']:.4f}")
            print(f"   Val Magnitude F1: {last_row['val_magnitude_f1']:.4f}")
            print(f"   Val Azimuth F1: {last_row['val_azimuth_f1']:.4f}")
            
            # Best metrics
            best_mag = df['val_magnitude_f1'].max()
            best_az = df['val_azimuth_f1'].max()
            best_mag_epoch = df['val_magnitude_f1'].idxmax()
            best_az_epoch = df['val_azimuth_f1'].idxmax()
            
            print(f"\n🏆 Best Metrics:")
            print(f"   Best Magnitude F1: {best_mag:.4f} (epoch {int(df.loc[best_mag_epoch, 'epoch']) + 1})")
            print(f"   Best Azimuth F1: {best_az:.4f} (epoch {int(df.loc[best_az_epoch, 'epoch']) + 1})")
        
        if len(df) >= 30:
            print(f"\n✅ Training COMPLETED!")
            return 'completed', latest_exp, df
        else:
            print(f"\n⚠️  Training INCOMPLETE ({len(df)}/30 epochs)")
            return 'incomplete', latest_exp, df
    
    elif model_file.exists():
        print(f"\n⚠️  Model checkpoint found but no training history")
        print(f"   Training may have been interrupted")
        return 'interrupted', latest_exp, None
    
    else:
        print(f"\n❌ No training results found")
        return 'failed', latest_exp, None

def show_detailed_results(exp_dir, df):
    """Show detailed training results"""
    print("\n" + "=" * 70)
    print("  DETAILED TRAINING RESULTS")
    print("=" * 70)
    
    # Final metrics
    last_row = df.iloc[-1]
    final_mag = last_row['val_magnitude_f1']
    final_az = last_row['val_azimuth_f1']
    
    print(f"\n📊 Final Metrics (Epoch {len(df)}):")
    print(f"   Magnitude F1: {final_mag:.4f}")
    print(f"   Azimuth F1: {final_az:.4f}")
    print(f"   Combined F1: {(final_mag + final_az) / 2:.4f}")
    
    # Baseline comparison
    baseline_mag = 0.87
    baseline_az = 0.60
    
    mag_improvement = ((final_mag - baseline_mag) / baseline_mag) * 100
    az_improvement = ((final_az - baseline_az) / baseline_az) * 100
    
    print(f"\n📈 Improvement vs Baseline:")
    print(f"   Magnitude F1: {baseline_mag:.4f} → {final_mag:.4f} ({mag_improvement:+.1f}%)")
    print(f"   Azimuth F1: {baseline_az:.4f} → {final_az:.4f} ({az_improvement:+.1f}%)")
    
    # Target achievement
    target_mag = 0.92
    target_az = 0.68
    
    mag_achieved = final_mag >= target_mag
    az_achieved = final_az >= target_az
    
    print(f"\n🎯 Target Achievement:")
    print(f"   Magnitude F1 ≥ {target_mag}: {'✅ ACHIEVED' if mag_achieved else f'❌ NOT YET ({final_mag:.4f})'}")
    print(f"   Azimuth F1 ≥ {target_az}: {'✅ ACHIEVED' if az_achieved else f'❌ NOT YET ({final_az:.4f})'}")
    
    if mag_achieved and az_achieved:
        print(f"\n🎉 CONGRATULATIONS! Both targets achieved!")
        print(f"✅ Ready to proceed to Phase 2")
    elif mag_achieved or az_achieved:
        print(f"\n⚠️  Partial success - one target achieved")
        print(f"💡 Consider continuing training or adjusting hyperparameters")
    else:
        print(f"\n⚠️  Targets not achieved yet")
        print(f"💡 Continue training or adjust approach")
    
    # Create visualization
    print(f"\n📊 Creating training visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Magnitude F1
    axes[0, 1].plot(df['epoch'], df['train_magnitude_f1'], label='Train', linewidth=2, alpha=0.7)
    axes[0, 1].plot(df['epoch'], df['val_magnitude_f1'], label='Validation', linewidth=2)
    axes[0, 1].axhline(y=baseline_mag, color='red', linestyle='--', label='Baseline (0.87)', alpha=0.6)
    axes[0, 1].axhline(y=target_mag, color='green', linestyle='--', label='Target (0.92)', alpha=0.6)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('F1 Score', fontsize=12)
    axes[0, 1].set_title('Magnitude F1 Score', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.5, 1.0])
    
    # Azimuth F1
    axes[1, 0].plot(df['epoch'], df['train_azimuth_f1'], label='Train', linewidth=2, alpha=0.7)
    axes[1, 0].plot(df['epoch'], df['val_azimuth_f1'], label='Validation', linewidth=2)
    axes[1, 0].axhline(y=baseline_az, color='red', linestyle='--', label='Baseline (0.60)', alpha=0.6)
    axes[1, 0].axhline(y=target_az, color='green', linestyle='--', label='Target (0.68)', alpha=0.6)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('F1 Score', fontsize=12)
    axes[1, 0].set_title('Azimuth F1 Score', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0.3, 0.9])
    
    # Combined F1
    combined_train = (df['train_magnitude_f1'] + df['train_azimuth_f1']) / 2
    combined_val = (df['val_magnitude_f1'] + df['val_azimuth_f1']) / 2
    
    axes[1, 1].plot(df['epoch'], combined_train, label='Train', linewidth=2, alpha=0.7)
    axes[1, 1].plot(df['epoch'], combined_val, label='Validation', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Combined F1 Score', fontsize=12)
    axes[1, 1].set_title('Combined F1 Score (Average)', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = exp_dir / 'training_curves_phase1.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✅ Training curves saved to: {plot_file}")
    
    plt.close()
    
    # Save summary
    summary_file = exp_dir / 'training_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("PHASE 1 TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Experiment: {exp_dir.name}\n")
        f.write(f"Epochs: {len(df)}/30\n\n")
        f.write(f"Final Metrics:\n")
        f.write(f"  Magnitude F1: {final_mag:.4f}\n")
        f.write(f"  Azimuth F1: {final_az:.4f}\n")
        f.write(f"  Combined F1: {(final_mag + final_az) / 2:.4f}\n\n")
        f.write(f"Improvement vs Baseline:\n")
        f.write(f"  Magnitude: {baseline_mag:.4f} → {final_mag:.4f} ({mag_improvement:+.1f}%)\n")
        f.write(f"  Azimuth: {baseline_az:.4f} → {final_az:.4f} ({az_improvement:+.1f}%)\n\n")
        f.write(f"Target Achievement:\n")
        f.write(f"  Magnitude ≥ {target_mag}: {'✅ YES' if mag_achieved else '❌ NO'}\n")
        f.write(f"  Azimuth ≥ {target_az}: {'✅ YES' if az_achieved else '❌ NO'}\n")
    
    print(f"✅ Summary saved to: {summary_file}")

def main():
    """Main function"""
    result = check_training_status()
    
    if result is None:
        print(f"\n💡 To start training, run:")
        print(f"   python train_with_improvements_v4.py")
        return
    
    status, exp_dir, df = result
    
    if status == 'completed' and df is not None:
        show_detailed_results(exp_dir, df)
    elif status == 'incomplete' and df is not None:
        print(f"\n💡 Training is incomplete. Options:")
        print(f"   1. Wait if training is still running")
        print(f"   2. Start new training: python train_with_improvements_v4.py")
        
        # Show partial results
        if len(df) > 5:
            print(f"\n📊 Partial results available:")
            show_detailed_results(exp_dir, df)
    else:
        print(f"\n💡 To start training, run:")
        print(f"   python train_with_improvements_v4.py")

if __name__ == '__main__':
    main()
