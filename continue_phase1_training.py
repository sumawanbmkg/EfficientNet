#!/usr/bin/env python3
"""
Continue or Start Phase 1 Training with Better Monitoring
Provides real-time progress updates and saves checkpoints

Author: Earthquake Prediction Research Team
Date: February 1, 2026
"""

import os
import sys
import time
from pathlib import Path

# Check if training is already running
def check_running_training():
    """Check if training process is already running"""
    import psutil
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'train_with_improvements_v4.py' in ' '.join(cmdline):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

# Check for existing experiments
def get_latest_experiment():
    """Get the latest experiment directory"""
    exp_dir = Path('experiments_v4')
    if not exp_dir.exists():
        return None
    
    experiments = sorted(exp_dir.glob('exp_v4_phase1_*_smote'))
    if not experiments:
        return None
    
    return experiments[-1]

# Check training status
def check_training_status():
    """Check the status of training"""
    print("🔍 Checking training status...")
    print("=" * 70)
    
    # Check if process is running
    running_pid = check_running_training()
    if running_pid:
        print(f"✅ Training is currently RUNNING (PID: {running_pid})")
        print(f"⏳ Please wait for training to complete")
        return 'running', running_pid
    
    # Check for latest experiment
    latest_exp = get_latest_experiment()
    if not latest_exp:
        print("❌ No experiments found")
        print("🚀 Ready to start new training")
        return 'not_started', None
    
    print(f"📁 Latest experiment: {latest_exp.name}")
    
    # Check for training history
    history_file = latest_exp / 'training_history.csv'
    if history_file.exists():
        import pandas as pd
        df = pd.read_csv(history_file)
        
        print(f"✅ Training history found: {len(df)} epochs completed")
        print(f"\n📊 Latest metrics:")
        if len(df) > 0:
            last_row = df.iloc[-1]
            print(f"   Epoch: {int(last_row['epoch']) + 1}")
            print(f"   Val Magnitude F1: {last_row['val_magnitude_f1']:.4f}")
            print(f"   Val Azimuth F1: {last_row['val_azimuth_f1']:.4f}")
            print(f"   Val Loss: {last_row['val_loss']:.4f}")
        
        if len(df) >= 30:
            print(f"\n✅ Training COMPLETED (30 epochs)")
            return 'completed', latest_exp
        else:
            print(f"\n⚠️  Training INCOMPLETE ({len(df)}/30 epochs)")
            return 'incomplete', latest_exp
    
    # Check for best model
    model_file = latest_exp / 'best_model.pth'
    if model_file.exists():
        print(f"✅ Model checkpoint found")
        print(f"⚠️  No training history - training may have been interrupted")
        return 'interrupted', latest_exp
    
    print(f"❌ No training results found")
    return 'failed', latest_exp

# Start new training
def start_training():
    """Start new training with monitoring"""
    print("\n🚀 Starting Phase 1 Training...")
    print("=" * 70)
    
    # Run training
    import subprocess
    
    try:
        # Start training process
        process = subprocess.Popen(
            [sys.executable, 'train_with_improvements_v4.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print(f"✅ Training started (PID: {process.pid})")
        print(f"📊 Monitoring output...\n")
        
        # Monitor output
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            print("\n✅ Training completed successfully!")
            return True
        else:
            print(f"\n❌ Training failed with return code: {return_code}")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False

# Show results
def show_results(exp_dir):
    """Show training results"""
    print("\n📊 TRAINING RESULTS")
    print("=" * 70)
    
    history_file = exp_dir / 'training_history.csv'
    if not history_file.exists():
        print("❌ No training history found")
        return
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(history_file)
    
    # Print summary
    print(f"\n✅ Training completed: {len(df)} epochs")
    print(f"\n📈 Final Metrics:")
    last_row = df.iloc[-1]
    print(f"   Magnitude F1: {last_row['val_magnitude_f1']:.4f}")
    print(f"   Azimuth F1: {last_row['val_azimuth_f1']:.4f}")
    print(f"   Combined F1: {(last_row['val_magnitude_f1'] + last_row['val_azimuth_f1']) / 2:.4f}")
    
    # Best metrics
    best_mag_idx = df['val_magnitude_f1'].idxmax()
    best_az_idx = df['val_azimuth_f1'].idxmax()
    
    print(f"\n🏆 Best Metrics:")
    print(f"   Best Magnitude F1: {df.loc[best_mag_idx, 'val_magnitude_f1']:.4f} (epoch {int(df.loc[best_mag_idx, 'epoch']) + 1})")
    print(f"   Best Azimuth F1: {df.loc[best_az_idx, 'val_azimuth_f1']:.4f} (epoch {int(df.loc[best_az_idx, 'epoch']) + 1})")
    
    # Compare with baseline
    print(f"\n📊 Comparison with Baseline:")
    baseline_mag = 0.87
    baseline_az = 0.60
    
    final_mag = last_row['val_magnitude_f1']
    final_az = last_row['val_azimuth_f1']
    
    mag_improvement = ((final_mag - baseline_mag) / baseline_mag) * 100
    az_improvement = ((final_az - baseline_az) / baseline_az) * 100
    
    print(f"   Magnitude F1: {baseline_mag:.4f} → {final_mag:.4f} ({mag_improvement:+.1f}%)")
    print(f"   Azimuth F1: {baseline_az:.4f} → {final_az:.4f} ({az_improvement:+.1f}%)")
    
    # Check if target achieved
    target_mag = 0.92
    target_az = 0.68
    
    print(f"\n🎯 Target Achievement:")
    mag_achieved = "✅" if final_mag >= target_mag else "❌"
    az_achieved = "✅" if final_az >= target_az else "❌"
    
    print(f"   Magnitude F1 ≥ {target_mag}: {mag_achieved} ({final_mag:.4f})")
    print(f"   Azimuth F1 ≥ {target_az}: {az_achieved} ({final_az:.4f})")
    
    # Create plots
    print(f"\n📈 Creating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', alpha=0.7)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Magnitude F1
    axes[0, 1].plot(df['epoch'], df['train_magnitude_f1'], label='Train', alpha=0.7)
    axes[0, 1].plot(df['epoch'], df['val_magnitude_f1'], label='Val', alpha=0.7)
    axes[0, 1].axhline(y=baseline_mag, color='r', linestyle='--', label='Baseline', alpha=0.5)
    axes[0, 1].axhline(y=target_mag, color='g', linestyle='--', label='Target', alpha=0.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Magnitude F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Azimuth F1
    axes[1, 0].plot(df['epoch'], df['train_azimuth_f1'], label='Train', alpha=0.7)
    axes[1, 0].plot(df['epoch'], df['val_azimuth_f1'], label='Val', alpha=0.7)
    axes[1, 0].axhline(y=baseline_az, color='r', linestyle='--', label='Baseline', alpha=0.5)
    axes[1, 0].axhline(y=target_az, color='g', linestyle='--', label='Target', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Azimuth F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined metrics
    combined_train = (df['train_magnitude_f1'] + df['train_azimuth_f1']) / 2
    combined_val = (df['val_magnitude_f1'] + df['val_azimuth_f1']) / 2
    
    axes[1, 1].plot(df['epoch'], combined_train, label='Train', alpha=0.7)
    axes[1, 1].plot(df['epoch'], combined_val, label='Val', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Combined F1 Score')
    axes[1, 1].set_title('Combined F1 Score (Average)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = exp_dir / 'training_curves.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✅ Training curves saved to: {plot_file}")
    
    plt.close()

def main():
    """Main function"""
    print("=" * 70)
    print("  PHASE 1 TRAINING - CONTINUATION SCRIPT")
    print("=" * 70)
    
    # Check status
    status, info = check_training_status()
    
    if status == 'running':
        print(f"\n⏳ Training is already running (PID: {info})")
        print(f"💡 Wait for it to complete or stop it first")
        return
    
    elif status == 'completed':
        print(f"\n✅ Training already completed!")
        show_results(info)
        return
    
    elif status == 'incomplete' or status == 'interrupted':
        print(f"\n⚠️  Previous training was incomplete")
        print(f"🔄 Starting new training...")
        
    elif status == 'not_started' or status == 'failed':
        print(f"\n🚀 Starting new training...")
    
    # Start training
    success = start_training()
    
    if success:
        # Show results
        latest_exp = get_latest_experiment()
        if latest_exp:
            show_results(latest_exp)
    else:
        print(f"\n❌ Training failed or was interrupted")
        print(f"💡 Check the logs for errors")

if __name__ == '__main__':
    main()
