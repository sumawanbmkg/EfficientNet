#!/usr/bin/env python3
"""
Fix Data Split - Remove Data Leakage
Split by station+date instead of sample index to prevent leakage

This is THE REAL FIX!
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

print("="*70)
print("FIX DATA SPLIT - REMOVE DATA LEAKAGE")
print("="*70)

# ============================================================================
# STEP 1: Load Metadata
# ============================================================================

print(f"\n📊 Step 1: Loading metadata...")

metadata_file = 'dataset_unified/metadata/unified_metadata.csv'
df = pd.read_csv(metadata_file)

print(f"✅ Loaded {len(df)} samples")

# ============================================================================
# STEP 2: Create Unique Keys (station+date)
# ============================================================================

print(f"\n📊 Step 2: Creating unique keys...")

# Create unique key for each station+date combination
df['unique_key'] = df['station'] + '_' + df['date'].astype(str)

# Get unique keys
unique_keys = df['unique_key'].unique()
print(f"✅ Found {len(unique_keys)} unique station+date combinations")
print(f"   Total samples: {len(df)}")
print(f"   Samples per key: {len(df)/len(unique_keys):.1f} average")

# ============================================================================
# STEP 3: Create Binary Labels for Stratification
# ============================================================================

print(f"\n📊 Step 3: Creating labels for stratification...")

# For each unique key, determine if it's earthquake or normal
key_labels = {}
for key in unique_keys:
    key_samples = df[df['unique_key'] == key]
    # If ANY sample is earthquake, mark key as earthquake
    # This ensures we don't split earthquake events across train/val
    is_earthquake = (key_samples['magnitude_class'] != 'Normal').any()
    key_labels[key] = 1 if is_earthquake else 0

# Create DataFrame for splitting
keys_df = pd.DataFrame({
    'key': list(key_labels.keys()),
    'is_earthquake': list(key_labels.values())
})

print(f"\n   Unique keys by type:")
earthquake_keys = (keys_df['is_earthquake'] == 1).sum()
normal_keys = (keys_df['is_earthquake'] == 0).sum()
print(f"      Earthquake: {earthquake_keys} ({earthquake_keys/len(keys_df)*100:.1f}%)")
print(f"      Normal: {normal_keys} ({normal_keys/len(keys_df)*100:.1f}%)")

# ============================================================================
# STEP 4: Split by Unique Keys (NO LEAKAGE!)
# ============================================================================

print(f"\n📊 Step 4: Splitting by unique keys (NO LEAKAGE)...")

# Split unique keys (not samples!)
train_keys, temp_keys = train_test_split(
    keys_df['key'].values,
    test_size=0.3,
    random_state=42,
    stratify=keys_df['is_earthquake']
)

# Split temp into val and test
temp_keys_df = keys_df[keys_df['key'].isin(temp_keys)]
val_keys, test_keys = train_test_split(
    temp_keys_df['key'].values,
    test_size=0.5,
    random_state=42,
    stratify=temp_keys_df['is_earthquake']
)

print(f"\n   Split by keys:")
print(f"      Train keys: {len(train_keys)}")
print(f"      Val keys: {len(val_keys)}")
print(f"      Test keys: {len(test_keys)}")

# ============================================================================
# STEP 5: Get Samples for Each Split
# ============================================================================

print(f"\n📊 Step 5: Getting samples for each split...")

# Get samples for each split
train_df = df[df['unique_key'].isin(train_keys)].copy()
val_df = df[df['unique_key'].isin(val_keys)].copy()
test_df = df[df['unique_key'].isin(test_keys)].copy()

print(f"\n   Split by samples:")
print(f"      Train: {len(train_df)} samples")
print(f"      Val: {len(val_df)} samples")
print(f"      Test: {len(test_df)} samples")

# Check Normal distribution
train_normal = (train_df['magnitude_class'] == 'Normal').sum()
val_normal = (val_df['magnitude_class'] == 'Normal').sum()
test_normal = (test_df['magnitude_class'] == 'Normal').sum()

print(f"\n   Normal samples:")
print(f"      Train: {train_normal} ({train_normal/len(train_df)*100:.1f}%)")
print(f"      Val: {val_normal} ({val_normal/len(val_df)*100:.1f}%)")
print(f"      Test: {test_normal} ({test_normal/len(test_df)*100:.1f}%)")

# ============================================================================
# STEP 6: Verify No Leakage
# ============================================================================

print(f"\n📊 Step 6: Verifying no leakage...")

def check_leakage(keys1, keys2, name1, name2):
    overlap = set(keys1) & set(keys2)
    if len(overlap) > 0:
        print(f"   ❌ LEAKAGE: {name1} ↔ {name2}: {len(overlap)} keys")
        return True
    else:
        print(f"   ✅ No leakage: {name1} ↔ {name2}")
        return False

has_leakage = False
has_leakage |= check_leakage(train_keys, val_keys, "Train", "Val")
has_leakage |= check_leakage(train_keys, test_keys, "Train", "Test")
has_leakage |= check_leakage(val_keys, test_keys, "Val", "Test")

if has_leakage:
    print(f"\n   🚨 ERROR: Leakage still exists!")
    exit(1)
else:
    print(f"\n   ✅ SUCCESS: No leakage detected!")

# ============================================================================
# STEP 7: Check Validation Distribution
# ============================================================================

print(f"\n📊 Step 7: Checking validation distribution...")

val_normal_samples = val_df[val_df['magnitude_class'] == 'Normal']

if len(val_normal_samples) > 0:
    station_dist = val_normal_samples['station'].value_counts()
    print(f"\n   Validation Normal by station:")
    for station, count in station_dist.items():
        pct = count / len(val_normal_samples) * 100
        print(f"      {station}: {count} ({pct:.1f}%)")
    
    # Check concentration
    max_pct = station_dist.max() / len(val_normal_samples) * 100
    if max_pct > 50:
        print(f"\n   ⚠️  WARNING: {max_pct:.1f}% from single station")
        print(f"   Consider rebalancing if possible")
    else:
        print(f"\n   ✅ Good distribution (max {max_pct:.1f}% from single station)")

# ============================================================================
# STEP 8: Save Split Indices
# ============================================================================

print(f"\n📊 Step 8: Saving split indices...")

# Save indices
split_indices = {
    'train_indices': train_df.index.tolist(),
    'val_indices': val_df.index.tolist(),
    'test_indices': test_df.index.tolist(),
    'train_keys': train_keys.tolist(),
    'val_keys': val_keys.tolist(),
    'test_keys': test_keys.tolist()
}

output_dir = Path('dataset_unified/metadata')
output_dir.mkdir(parents=True, exist_ok=True)

split_file = output_dir / 'fixed_split_indices.json'
with open(split_file, 'w') as f:
    json.dump(split_indices, f, indent=2)

print(f"✅ Split indices saved to: {split_file}")

# Save split metadata
split_metadata = {
    'total_samples': len(df),
    'total_keys': len(unique_keys),
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df),
    'train_keys': len(train_keys),
    'val_keys': len(val_keys),
    'test_keys': len(test_keys),
    'train_normal': int(train_normal),
    'val_normal': int(val_normal),
    'test_normal': int(test_normal),
    'has_leakage': has_leakage,
    'split_method': 'by_station_date',
    'random_state': 42
}

metadata_file = output_dir / 'fixed_split_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(split_metadata, f, indent=2)

print(f"✅ Split metadata saved to: {metadata_file}")

# ============================================================================
# STEP 9: Create Split CSVs (for easy loading)
# ============================================================================

print(f"\n📊 Step 9: Creating split CSV files...")

train_df.to_csv(output_dir / 'train_split.csv', index=False)
val_df.to_csv(output_dir / 'val_split.csv', index=False)
test_df.to_csv(output_dir / 'test_split.csv', index=False)

print(f"✅ Split CSV files created:")
print(f"   - train_split.csv ({len(train_df)} samples)")
print(f"   - val_split.csv ({len(val_df)} samples)")
print(f"   - test_split.csv ({len(test_df)} samples)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")

print(f"\n✅ DATA SPLIT FIXED!")

print(f"\n📊 Split Statistics:")
print(f"   Method: Split by station+date (NO LEAKAGE)")
print(f"   Total samples: {len(df)}")
print(f"   Total unique keys: {len(unique_keys)}")

print(f"\n   Train:")
print(f"      Samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"      Keys: {len(train_keys)}")
print(f"      Normal: {train_normal} ({train_normal/len(train_df)*100:.1f}%)")

print(f"\n   Validation:")
print(f"      Samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"      Keys: {len(val_keys)}")
print(f"      Normal: {val_normal} ({val_normal/len(val_df)*100:.1f}%)")

print(f"\n   Test:")
print(f"      Samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
print(f"      Keys: {len(test_keys)}")
print(f"      Normal: {test_normal} ({test_normal/len(test_df)*100:.1f}%)")

print(f"\n✅ Verification:")
print(f"   Data leakage: {'❌ FOUND' if has_leakage else '✅ NONE'}")
print(f"   Splits are independent: ✅ YES")
print(f"   Ready for training: ✅ YES")

print(f"\n📁 Files Created:")
print(f"   1. fixed_split_indices.json - Split indices")
print(f"   2. fixed_split_metadata.json - Split metadata")
print(f"   3. train_split.csv - Training data")
print(f"   4. val_split.csv - Validation data")
print(f"   5. test_split.csv - Test data")

print(f"\n🚀 NEXT STEP:")
print(f"   Retrain model with fixed split!")
print(f"   Expected validation accuracy: 70-80% (realistic)")
print(f"   Expected to generalize to new data!")

print(f"\n{'='*70}")
print("✅ DATA SPLIT FIX COMPLETE!")
print(f"{'='*70}")
