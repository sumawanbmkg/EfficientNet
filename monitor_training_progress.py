#!/usr/bin/env python3
"""
Monitor Training Progress
Quick script to check training status
"""

from pathlib import Path
import json
import pandas as pd

print("="*70)
print("TRAINING PROGRESS MONITOR")
print("="*70)

# Check for experiment directories
exp_dir = Path('experiments_fixed')

if not exp_dir.exists():
    print("\n⏳ Training not started yet (no experiments_fixed directory)")
    exit(0)

# Get latest experiment
experiments = sorted(exp_dir.glob('exp_*'))

if not experiments:
    print("\n⏳ Training not started yet (no experiments found)")
    exit(0)

latest_exp = experiments[-1]
print(f"\n📁 Latest Experiment: {latest_exp.name}")

# Check for training history
history_file = latest_exp / 'training_history.csv'

if not history_file.exists():
    print("\n⏳ Training in progress (no history file yet)")
    print("   Model is likely still initializing or in early epochs")
    exit(0)

# Load and display history
df = pd.read_csv(history_file)

print(f"\n📊 Training Progress:")
print(f"   Epochs completed: {len(df)}")

if len(df) > 0:
    latest = df.iloc[-1]
    
    print(f"\n   Latest Epoch Metrics:")
    print(f"      Magnitude Accuracy:")
    print(f"         Train: {latest['train_mag_acc']:.2f}%")
    print(f"         Val: {latest['val_mag_acc']:.2f}%")
    
    print(f"\n      Azimuth Accuracy:")
    print(f"         Train: {latest['train_azi_acc']:.2f}%")
    print(f"         Val: {latest['val_azi_acc']:.2f}%")
    
    print(f"\n      Loss:")
    print(f"         Train: {latest['train_loss']:.4f}")
    print(f"         Val: {latest['val_loss']:.4f}")

# Check for best model
best_model = latest_exp / 'best_model.pth'

if best_model.exists():
    print(f"\n✅ Best model saved: {best_model}")
else:
    print(f"\n⏳ No best model yet")

print(f"\n{'='*70}")
