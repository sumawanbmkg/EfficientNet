#!/usr/bin/env python3
"""
Investigate Training Data
Find out why model fails even with training spectrograms

Checks:
1. Validation split quality
2. Label quality
3. Data leakage
4. Class distribution
5. Sample quality
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import json

print("="*70)
print("INVESTIGATE TRAINING DATA - ROOT CAUSE ANALYSIS")
print("="*70)

# ============================================================================
# STEP 1: Load and Analyze Metadata
# ============================================================================

print(f"\n📊 Step 1: Loading metadata...")

metadata_file = 'dataset_unified/metadata/unified_metadata.csv'
df = pd.read_csv(metadata_file)

print(f"✅ Loaded {len(df)} samples")

# ============================================================================
# STEP 2: Check Class Distribution
# ============================================================================

print(f"\n📊 Step 2: Analyzing class distribution...")

print(f"\nMagnitude Classes:")
mag_dist = df['magnitude_class'].value_counts()
for cls, count in mag_dist.items():
    pct = count / len(df) * 100
    print(f"   {cls}: {count} ({pct:.1f}%)")

print(f"\nAzimuth Classes:")
az_dist = df['azimuth_class'].value_counts()
for cls, count in az_dist.items():
    pct = count / len(df) * 100
    print(f"   {cls}: {count} ({pct:.1f}%)")

# Check Normal class
normal_count = (df['magnitude_class'] == 'Normal').sum()
normal_pct = normal_count / len(df) * 100

print(f"\n🎯 Normal Class:")
print(f"   Count: {normal_count} ({normal_pct:.1f}%)")
print(f"   Status: {'✅ Sufficient' if normal_pct > 30 else '⚠️ Low' if normal_pct > 10 else '❌ Too Low'}")

# ============================================================================
# STEP 3: Analyze Data Split (Simulate Training Split)
# ============================================================================

print(f"\n📊 Step 3: Analyzing data split...")

from sklearn.model_selection import train_test_split

# Simulate the split used in training
# Check if Normal samples are in all splits

# Create binary labels
df['is_earthquake'] = (df['magnitude_class'] != 'Normal').astype(int)

# Split (same as training)
train_idx, temp_idx = train_test_split(
    range(len(df)), test_size=0.3, random_state=42,
    stratify=df['is_earthquake']
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42,
    stratify=df.iloc[temp_idx]['is_earthquake']
)

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]
test_df = df.iloc[test_idx]

print(f"\nSplit Sizes:")
print(f"   Train: {len(train_df)}")
print(f"   Val: {len(val_df)}")
print(f"   Test: {len(test_df)}")

print(f"\nNormal Samples in Each Split:")
train_normal = (train_df['magnitude_class'] == 'Normal').sum()
val_normal = (val_df['magnitude_class'] == 'Normal').sum()
test_normal = (test_df['magnitude_class'] == 'Normal').sum()

print(f"   Train: {train_normal} ({train_normal/len(train_df)*100:.1f}%)")
print(f"   Val: {val_normal} ({val_normal/len(val_df)*100:.1f}%)")
print(f"   Test: {test_normal} ({test_normal/len(test_df)*100:.1f}%)")

# Check if validation has diverse Normal samples
val_normal_samples = val_df[val_df['magnitude_class'] == 'Normal']
print(f"\nValidation Normal Samples:")
print(f"   Total: {len(val_normal_samples)}")
print(f"   Unique stations: {val_normal_samples['station'].nunique()}")
print(f"   Unique dates: {val_normal_samples['date'].nunique()}")

# Check if all Normal samples are from same station/date
if len(val_normal_samples) > 0:
    station_dist = val_normal_samples['station'].value_counts()
    print(f"\n   Station distribution:")
    for station, count in station_dist.head(5).items():
        print(f"      {station}: {count}")
    
    # Check if too concentrated
    max_station_pct = station_dist.max() / len(val_normal_samples) * 100
    if max_station_pct > 50:
        print(f"\n   ⚠️  WARNING: {max_station_pct:.1f}% from single station!")
        print(f"   This could cause overfitting!")

# ============================================================================
# STEP 4: Check for Data Leakage
# ============================================================================

print(f"\n📊 Step 4: Checking for data leakage...")

# Check if same station+date appears in multiple splits
def check_leakage(df1, df2, name1, name2):
    df1_keys = set(df1['station'] + '_' + df1['date'].astype(str))
    df2_keys = set(df2['station'] + '_' + df2['date'].astype(str))
    
    overlap = df1_keys & df2_keys
    
    if len(overlap) > 0:
        print(f"\n   ⚠️  LEAKAGE DETECTED: {name1} ↔ {name2}")
        print(f"   Overlapping samples: {len(overlap)}")
        print(f"   Examples: {list(overlap)[:3]}")
        return True
    else:
        print(f"   ✅ No leakage: {name1} ↔ {name2}")
        return False

has_leakage = False
has_leakage |= check_leakage(train_df, val_df, "Train", "Val")
has_leakage |= check_leakage(train_df, test_df, "Train", "Test")
has_leakage |= check_leakage(val_df, test_df, "Val", "Test")

if has_leakage:
    print(f"\n   🚨 DATA LEAKAGE FOUND!")
    print(f"   This explains why validation accuracy is misleading!")
else:
    print(f"\n   ✅ No data leakage detected")

# ============================================================================
# STEP 5: Check Label Quality
# ============================================================================

print(f"\n📊 Step 5: Checking label quality...")

# Check Normal samples
normal_samples = df[df['magnitude_class'] == 'Normal']

print(f"\nNormal Samples Analysis:")
print(f"   Total: {len(normal_samples)}")

# Check if Normal samples have consistent azimuth
normal_azimuth_dist = normal_samples['azimuth_class'].value_counts()
print(f"\n   Azimuth distribution:")
for az, count in normal_azimuth_dist.items():
    pct = count / len(normal_samples) * 100
    print(f"      {az}: {count} ({pct:.1f}%)")

# Check if all Normal samples have azimuth=Normal
normal_with_normal_az = (normal_samples['azimuth_class'] == 'Normal').sum()
normal_with_other_az = len(normal_samples) - normal_with_normal_az

print(f"\n   Consistency check:")
print(f"      Normal + Normal: {normal_with_normal_az} ({normal_with_normal_az/len(normal_samples)*100:.1f}%)")
print(f"      Normal + Other: {normal_with_other_az} ({normal_with_other_az/len(normal_samples)*100:.1f}%)")

if normal_with_other_az > 0:
    print(f"\n   ⚠️  WARNING: {normal_with_other_az} Normal samples have non-Normal azimuth!")
    print(f"   This is INCONSISTENT and could confuse the model!")

# ============================================================================
# STEP 6: Visual Inspection of Normal Samples
# ============================================================================

print(f"\n📊 Step 6: Visual inspection of Normal samples...")

# Load a few Normal samples and check if they look similar
normal_samples_to_check = normal_samples.head(10)

print(f"\nChecking {len(normal_samples_to_check)} Normal samples...")

sample_stats = []
for idx, sample in normal_samples_to_check.iterrows():
    spec_path = Path('dataset_unified') / sample['unified_path']
    
    if not spec_path.exists():
        continue
    
    img = Image.open(spec_path)
    img_array = np.array(img)
    
    stats = {
        'station': sample['station'],
        'date': sample['date'],
        'mean': img_array.mean(),
        'std': img_array.std(),
        'min': img_array.min(),
        'max': img_array.max()
    }
    sample_stats.append(stats)

if len(sample_stats) > 0:
    print(f"\n   Sample statistics:")
    means = [s['mean'] for s in sample_stats]
    stds = [s['std'] for s in sample_stats]
    
    print(f"      Mean: {np.mean(means):.1f} ± {np.std(means):.1f}")
    print(f"      Std: {np.mean(stds):.1f} ± {np.std(stds):.1f}")
    
    # Check if samples are too similar (possible duplicates)
    mean_variation = np.std(means)
    if mean_variation < 5:
        print(f"\n   ⚠️  WARNING: Very low variation in Normal samples!")
        print(f"   Samples might be too similar or duplicated!")

# ============================================================================
# STEP 7: Check Training vs Validation Similarity
# ============================================================================

print(f"\n📊 Step 7: Checking train/val similarity...")

# Check if validation Normal samples are similar to training Normal samples
train_normal = train_df[train_df['magnitude_class'] == 'Normal']
val_normal = val_df[val_df['magnitude_class'] == 'Normal']

print(f"\nTrain Normal Samples:")
print(f"   Count: {len(train_normal)}")
print(f"   Stations: {train_normal['station'].nunique()}")
print(f"   Date range: {train_normal['date'].min()} to {train_normal['date'].max()}")

print(f"\nVal Normal Samples:")
print(f"   Count: {len(val_normal)}")
print(f"   Stations: {val_normal['station'].nunique()}")
print(f"   Date range: {val_normal['date'].min()} to {val_normal['date'].max()}")

# Check if validation stations are in training
train_stations = set(train_normal['station'])
val_stations = set(val_normal['station'])

common_stations = train_stations & val_stations
print(f"\nCommon stations: {len(common_stations)}/{len(val_stations)}")

if len(common_stations) < len(val_stations):
    print(f"   ⚠️  WARNING: Validation has stations not in training!")
    print(f"   Unseen stations: {val_stations - train_stations}")

# ============================================================================
# FINAL ANALYSIS
# ============================================================================

print(f"\n{'='*70}")
print("FINAL ANALYSIS")
print(f"{'='*70}")

issues_found = []

# Issue 1: Data leakage
if has_leakage:
    issues_found.append("🔴 Data leakage detected")

# Issue 2: Label inconsistency
if normal_with_other_az > 0:
    issues_found.append(f"🔴 {normal_with_other_az} inconsistent Normal labels")

# Issue 3: Low Normal representation
if normal_pct < 30:
    issues_found.append(f"⚠️ Low Normal representation ({normal_pct:.1f}%)")

# Issue 4: Validation concentration
if len(val_normal_samples) > 0:
    max_station_pct = val_normal_samples['station'].value_counts().max() / len(val_normal_samples) * 100
    if max_station_pct > 50:
        issues_found.append(f"⚠️ Validation Normal samples too concentrated ({max_station_pct:.1f}% from one station)")

# Issue 5: Unseen stations in validation
if len(common_stations) < len(val_stations):
    issues_found.append(f"⚠️ Validation has unseen stations")

print(f"\n🔍 ISSUES FOUND: {len(issues_found)}")
if len(issues_found) > 0:
    for issue in issues_found:
        print(f"   {issue}")
else:
    print(f"   ✅ No major issues found")

print(f"\n🎯 ROOT CAUSE HYPOTHESIS:")

if has_leakage:
    print(f"   🔴 PRIMARY: Data leakage")
    print(f"   Model memorized validation set")
    print(f"   Cannot generalize to new samples")
elif normal_with_other_az > 0:
    print(f"   🔴 PRIMARY: Inconsistent labels")
    print(f"   Normal samples have non-Normal azimuth")
    print(f"   Model learned wrong patterns")
elif max_station_pct > 50:
    print(f"   ⚠️  PRIMARY: Validation overfitting")
    print(f"   Validation samples too concentrated")
    print(f"   Model overfitted to specific patterns")
else:
    print(f"   ⚠️  UNCLEAR: No obvious data issues")
    print(f"   Problem might be in model architecture or training")

print(f"\n💡 RECOMMENDED SOLUTION:")

if has_leakage:
    print(f"   1. Fix data split (remove leakage)")
    print(f"   2. Retrain model")
    print(f"   3. Verify on held-out samples")
elif normal_with_other_az > 0:
    print(f"   1. Fix inconsistent labels")
    print(f"   2. Ensure Normal + Normal consistency")
    print(f"   3. Retrain model")
elif max_station_pct > 50:
    print(f"   1. Rebalance validation set")
    print(f"   2. Ensure diverse samples")
    print(f"   3. Retrain model")
else:
    print(f"   1. Try different architecture (EfficientNet)")
    print(f"   2. Use focal loss with extreme gamma")
    print(f"   3. Use hard negative mining")

print(f"\n{'='*70}")
print("✅ INVESTIGATION COMPLETE!")
print(f"{'='*70}")

# Save report
report = {
    'total_samples': len(df),
    'normal_count': int(normal_count),
    'normal_percentage': float(normal_pct),
    'has_data_leakage': has_leakage,
    'inconsistent_labels': int(normal_with_other_az),
    'issues_found': issues_found,
    'train_normal': int(train_normal),
    'val_normal': int(val_normal),
    'test_normal': int(test_normal)
}

with open('training_data_investigation_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n💾 Report saved to: training_data_investigation_report.json")
