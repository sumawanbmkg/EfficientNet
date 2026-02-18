#!/usr/bin/env python3
"""
Verify Training Data Consistency
Check if training data has consistent magnitude/azimuth labels
"""

import pandas as pd

print("="*70)
print("TRAINING DATA CONSISTENCY VERIFICATION")
print("="*70)

# Load dataset
df = pd.read_csv('dataset_unified/metadata/unified_metadata.csv')

print(f"\nTotal samples: {len(df)}")

# Check Normal data
normal_data = df[df['magnitude_class'] == 'Normal']
print(f"\n📊 Normal data (Magnitude=Normal): {len(normal_data)} samples")
print(f"   Azimuth=Normal: {len(normal_data[normal_data['azimuth_class'] == 'Normal'])}")
print(f"   Azimuth!=Normal: {len(normal_data[normal_data['azimuth_class'] != 'Normal'])}")

# Check Earthquake data
eq_data = df[df['magnitude_class'] != 'Normal']
print(f"\n📊 Earthquake data (Magnitude!=Normal): {len(eq_data)} samples")
print(f"   Azimuth=Normal: {len(eq_data[eq_data['azimuth_class'] == 'Normal'])}")
print(f"   Azimuth!=Normal: {len(eq_data[eq_data['azimuth_class'] != 'Normal'])}")

# Check consistency
normal_inconsistent = len(normal_data[normal_data['azimuth_class'] != 'Normal'])
eq_inconsistent = len(eq_data[eq_data['azimuth_class'] == 'Normal'])

print(f"\n{'='*70}")
print("CONSISTENCY RESULTS")
print(f"{'='*70}")

if normal_inconsistent == 0 and eq_inconsistent == 0:
    print("✅ PERFECT CONSISTENCY!")
    print("   All Normal samples have Azimuth=Normal")
    print("   All Earthquake samples have Azimuth!=Normal")
    print("\n🎯 CONCLUSION: NO RE-TRAINING NEEDED")
    print("   Training data is already 100% consistent!")
else:
    print("❌ INCONSISTENCY FOUND!")
    print(f"   Normal samples with wrong azimuth: {normal_inconsistent}")
    print(f"   Earthquake samples with wrong azimuth: {eq_inconsistent}")
    print("\n⚠️  CONCLUSION: May need data cleaning")

print(f"{'='*70}")
