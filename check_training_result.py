import torch
from pathlib import Path

# Find latest experiment
exp_dir = Path('experiments_improved')
latest = sorted(exp_dir.glob('improved_*'))[-1]
print(f'Latest experiment: {latest}')

# Load checkpoint
checkpoint = torch.load(latest / 'best_model.pth', weights_only=False)
print(f'\nCheckpoint keys: {checkpoint.keys()}')
print(f'Best epoch: {checkpoint["epoch"]}')
print(f'\nValidation metrics:')
vm = checkpoint['val_metrics']
print(f'  Magnitude Accuracy: {vm["mag_acc"]:.2f}%')
print(f'  Magnitude F1: {vm["mag_f1"]:.2f}%')
print(f'  Azimuth Accuracy: {vm["azi_acc"]:.2f}%')
print(f'  Azimuth F1: {vm["azi_f1"]:.2f}%')
print(f'  Combined F1: {(vm["mag_f1"] + vm["azi_f1"])/2:.2f}%')
