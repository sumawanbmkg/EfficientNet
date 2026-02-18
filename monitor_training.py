"""
Quick script to monitor training progress
"""
import os
import pandas as pd
from glob import glob

# Find latest experiment
exp_dirs = sorted(glob('experiments/exp_*'))
if not exp_dirs:
    print("No experiments found!")
    exit()

latest_exp = exp_dirs[-1]
history_file = os.path.join(latest_exp, 'training_history.csv')

print("="*80)
print(f"MONITORING: {latest_exp}")
print("="*80)

if os.path.exists(history_file):
    df = pd.read_csv(history_file)
    
    print(f"\nCompleted Epochs: {len(df)}/50")
    print(f"\nLatest Metrics (Epoch {len(df)}):")
    print("-"*80)
    
    latest = df.iloc[-1]
    print(f"Train Loss:     {latest['train_loss']:.4f}")
    print(f"Val Loss:       {latest['val_loss']:.4f}")
    print(f"Mag Acc (T/V):  {latest['train_mag_acc']:.4f} / {latest['val_mag_acc']:.4f}")
    print(f"Az Acc (T/V):   {latest['train_az_acc']:.4f} / {latest['val_az_acc']:.4f}")
    print(f"Learning Rate:  {latest['lr']:.6f}")
    
    print(f"\nBest Validation Loss: {df['val_loss'].min():.4f} (Epoch {df['val_loss'].idxmin()+1})")
    
    print("\n" + "="*80)
    print("TRAINING HISTORY (Last 10 Epochs)")
    print("="*80)
    print(df[['train_loss', 'val_loss', 'train_mag_acc', 'val_mag_acc', 
              'train_az_acc', 'val_az_acc']].tail(10).to_string(index=False))
    
else:
    print("\nTraining history not yet available. Training may have just started.")
    print("Check again in a few minutes...")

print("\n" + "="*80)
