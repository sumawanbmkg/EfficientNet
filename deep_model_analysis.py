#!/usr/bin/env python3
"""
Deep Model Analysis - Investigate why model predicts Normal for earthquake data
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json

print("="*70)
print("DEEP MODEL ANALYSIS")
print("Why does model predict 'Normal' for M6.4 earthquake data?")
print("="*70)

# ============================================================================
# 1. Analyze Training Data Distribution
# ============================================================================

print("\n📊 1. TRAINING DATA DISTRIBUTION")
print("-"*50)

# Load metadata
metadata_file = Path('dataset_unified/metadata/unified_metadata.csv')
if metadata_file.exists():
    df = pd.read_csv(metadata_file)
    
    print(f"Total samples: {len(df)}")
    
    # Magnitude distribution
    print(f"\nMagnitude class distribution:")
    mag_counts = df['magnitude_class'].value_counts()
    for cls, count in mag_counts.items():
        pct = count / len(df) * 100
        print(f"   {cls}: {count} ({pct:.1f}%)")
    
    # Azimuth distribution
    print(f"\nAzimuth class distribution:")
    azi_counts = df['azimuth_class'].value_counts()
    for cls, count in azi_counts.items():
        pct = count / len(df) * 100
        print(f"   {cls}: {count} ({pct:.1f}%)")
    
    # Check for TRT station
    print(f"\nTRT station data:")
    trt_df = df[df['station'] == 'TRT']
    print(f"   Total TRT samples: {len(trt_df)}")
    if len(trt_df) > 0:
        print(f"   TRT magnitude distribution:")
        trt_mag = trt_df['magnitude_class'].value_counts()
        for cls, count in trt_mag.items():
            print(f"      {cls}: {count}")
        print(f"   TRT azimuth distribution:")
        trt_azi = trt_df['azimuth_class'].value_counts()
        for cls, count in trt_azi.items():
            print(f"      {cls}: {count}")
else:
    print("   ❌ Metadata file not found")

# ============================================================================
# 2. Check Training Split
# ============================================================================

print("\n📊 2. TRAINING SPLIT ANALYSIS")
print("-"*50)

split_file = Path('dataset_unified/metadata/fixed_split_indices.json')
if split_file.exists():
    with open(split_file, 'r') as f:
        split_indices = json.load(f)
    
    train_df = df.iloc[split_indices['train_indices']]
    val_df = df.iloc[split_indices['val_indices']]
    test_df = df.iloc[split_indices['test_indices']]
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Val set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Training set distribution
    print(f"\nTraining set magnitude distribution:")
    train_mag = train_df['magnitude_class'].value_counts()
    for cls, count in train_mag.items():
        pct = count / len(train_df) * 100
        print(f"   {cls}: {count} ({pct:.1f}%)")
    
    # Check Normal class ratio
    normal_count = train_mag.get('Normal', 0)
    large_count = train_mag.get('Large', 0)
    
    print(f"\n⚠️  CRITICAL IMBALANCE:")
    print(f"   Normal samples: {normal_count}")
    print(f"   Large samples: {large_count}")
    if large_count > 0:
        ratio = normal_count / large_count
        print(f"   Ratio Normal:Large = {ratio:.1f}:1")
else:
    print("   ❌ Split file not found")

# ============================================================================
# 3. Analyze Model Weights
# ============================================================================

print("\n📊 3. MODEL WEIGHTS ANALYSIS")
print("-"*50)

model_path = Path('experiments_fixed/exp_fixed_20260202_163643/best_model.pth')
if model_path.exists():
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Analyze magnitude head weights
    mag_weight = state_dict.get('magnitude_head.weight')
    mag_bias = state_dict.get('magnitude_head.bias')
    
    if mag_weight is not None:
        print(f"Magnitude head weight shape: {mag_weight.shape}")
        print(f"Magnitude head bias: {mag_bias}")
        
        # Check bias values - higher bias = more likely to predict that class
        print(f"\nMagnitude head bias analysis:")
        bias_values = mag_bias.numpy()
        classes = ['Normal', 'Moderate', 'Medium', 'Large']
        for i, (cls, bias) in enumerate(zip(classes, bias_values)):
            print(f"   {cls}: bias = {bias:.4f}")
        
        # The class with highest bias is most likely to be predicted
        max_bias_idx = np.argmax(bias_values)
        print(f"\n⚠️  Highest bias: {classes[max_bias_idx]} ({bias_values[max_bias_idx]:.4f})")
        print(f"   This class is most likely to be predicted by default!")
    
    # Analyze azimuth head weights
    azi_weight = state_dict.get('azimuth_head.weight')
    azi_bias = state_dict.get('azimuth_head.bias')
    
    if azi_weight is not None:
        print(f"\nAzimuth head weight shape: {azi_weight.shape}")
        print(f"Azimuth head bias: {azi_bias}")
        
        bias_values = azi_bias.numpy()
        classes = ['Normal', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        print(f"\nAzimuth head bias analysis:")
        for i, (cls, bias) in enumerate(zip(classes, bias_values)):
            print(f"   {cls}: bias = {bias:.4f}")
        
        max_bias_idx = np.argmax(bias_values)
        print(f"\n⚠️  Highest bias: {classes[max_bias_idx]} ({bias_values[max_bias_idx]:.4f})")
else:
    print("   ❌ Model file not found")

# ============================================================================
# 4. Check Training History
# ============================================================================

print("\n📊 4. TRAINING HISTORY ANALYSIS")
print("-"*50)

history_file = Path('experiments_fixed/exp_fixed_20260202_163643/training_history.csv')
if history_file.exists():
    history = pd.read_csv(history_file)
    
    print(f"Training epochs: {len(history)}")
    
    # Final metrics
    final = history.iloc[-1]
    print(f"\nFinal training metrics:")
    print(f"   Train Mag Acc: {final['train_mag_acc']:.2f}%")
    print(f"   Train Azi Acc: {final['train_azi_acc']:.2f}%")
    print(f"   Val Mag Acc: {final['val_mag_acc']:.2f}%")
    print(f"   Val Azi Acc: {final['val_azi_acc']:.2f}%")
    
    # Best metrics
    best_mag_acc = history['val_mag_acc'].max()
    best_azi_acc = history['val_azi_acc'].max()
    print(f"\nBest validation metrics:")
    print(f"   Best Val Mag Acc: {best_mag_acc:.2f}%")
    print(f"   Best Val Azi Acc: {best_azi_acc:.2f}%")
    
    # Check for overfitting
    train_mag_final = final['train_mag_acc']
    val_mag_final = final['val_mag_acc']
    gap = train_mag_final - val_mag_final
    
    print(f"\nOverfitting analysis:")
    print(f"   Train-Val gap (Magnitude): {gap:.2f}%")
    if gap > 10:
        print(f"   ⚠️  Significant overfitting detected!")
else:
    print("   ❌ Training history not found")

# ============================================================================
# 5. ROOT CAUSE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("ROOT CAUSE ANALYSIS")
print("="*70)

print("""
Based on the analysis, the likely root causes are:

1. SEVERE CLASS IMBALANCE
   - Normal class dominates the training data
   - Large class has very few samples
   - Model learns to predict Normal by default (safe bet)

2. MODEL BIAS
   - The magnitude head has highest bias for Normal class
   - This means even with ambiguous input, model predicts Normal

3. INSUFFICIENT LARGE EARTHQUAKE SAMPLES
   - Large earthquakes (M≥6.0) are rare events
   - Model hasn't seen enough examples to learn the pattern

4. POSSIBLE SOLUTIONS:
   a) Use stronger class weights during training
   b) Use focal loss to focus on hard examples
   c) Oversample Large class (SMOTE or augmentation)
   d) Undersample Normal class
   e) Use ensemble of models
   f) Collect more Large earthquake data
""")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
