#!/usr/bin/env python3
"""
Find Best Model - Compare all Phase 1 models
"""

import os
import json
from pathlib import Path

print("="*70)
print("FINDING BEST MODEL - PHASE 1")
print("="*70)

exp_dir = Path('experiments_v4')
experiments = sorted(exp_dir.glob('exp_v4_phase1_*'))

print(f"\nFound {len(experiments)} experiments\n")

best_model = None
best_f1 = 0
best_val_loss = float('inf')

results = []

for exp in experiments:
    metrics_file = exp / 'metrics.json'
    
    if not metrics_file.exists():
        continue
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Get best epoch metrics
    val_loss = metrics.get('best_val_loss', float('inf'))
    val_f1_mag = metrics.get('best_val_f1_magnitude', 0)
    val_f1_az = metrics.get('best_val_f1_azimuth', 0)
    avg_f1 = (val_f1_mag + val_f1_az) / 2
    
    results.append({
        'experiment': exp.name,
        'val_loss': val_loss,
        'val_f1_magnitude': val_f1_mag,
        'val_f1_azimuth': val_f1_az,
        'avg_f1': avg_f1
    })
    
    print(f"📊 {exp.name}")
    print(f"   Val Loss: {val_loss:.4f}")
    print(f"   Val F1 Magnitude: {val_f1_mag:.4f}")
    print(f"   Val F1 Azimuth: {val_f1_az:.4f}")
    print(f"   Avg F1: {avg_f1:.4f}")
    print()
    
    # Track best by F1 score
    if avg_f1 > best_f1:
        best_f1 = avg_f1
        best_model = exp

# Sort by avg F1
results.sort(key=lambda x: x['avg_f1'], reverse=True)

print("="*70)
print("RANKING BY AVG F1 SCORE")
print("="*70)

for i, result in enumerate(results, 1):
    print(f"\n{i}. {result['experiment']}")
    print(f"   Avg F1: {result['avg_f1']:.4f}")
    print(f"   Magnitude F1: {result['val_f1_magnitude']:.4f}")
    print(f"   Azimuth F1: {result['val_f1_azimuth']:.4f}")
    print(f"   Val Loss: {result['val_loss']:.4f}")

print("\n" + "="*70)
print("BEST MODEL")
print("="*70)

if best_model:
    print(f"\n🏆 {best_model.name}")
    print(f"   Avg F1: {best_f1:.4f}")
    print(f"   Model Path: {best_model / 'best_model.pth'}")
else:
    print("\n❌ No valid models found")
