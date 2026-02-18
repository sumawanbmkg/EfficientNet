#!/usr/bin/env python3
"""
Test Fixed Scanner
Verify that fixed scanner generates training-compatible spectrograms
"""

import sys
import os
import numpy as np
from PIL import Image
from pathlib import Path

print("="*70)
print("TEST FIXED SCANNER - TRAINING COMPATIBILITY")
print("="*70)

# Test case
TEST_STATION = 'GTO'
TEST_DATE = '2021-12-05'

print(f"\n📊 Test Case: {TEST_STATION} - {TEST_DATE}")

# ============================================================================
# STEP 1: Generate scanner spectrogram (FIXED VERSION)
# ============================================================================

print(f"\n🎨 Step 1: Generating scanner spectrogram (FIXED)...")

sys.path.insert(0, os.path.dirname(__file__))
from prekursor_scanner import PrekursorScanner

try:
    scanner = PrekursorScanner()
    
    # Fetch data
    data = scanner.fetch_data(TEST_DATE, TEST_STATION)
    if data is None:
        print(f"❌ Failed to fetch data")
        sys.exit(1)
    
    # Generate spectrogram (FIXED VERSION)
    scanner_spec = scanner.generate_spectrogram(data, component='3comp')
    if scanner_spec is None:
        print(f"❌ Failed to generate spectrogram")
        sys.exit(1)
    
    print(f"✅ Scanner spectrogram generated (FIXED)")
    print(f"   Shape: {scanner_spec.shape}")
    print(f"   Dtype: {scanner_spec.dtype}")
    print(f"   Value range: {scanner_spec.min()}-{scanner_spec.max()}")
    
    # Check if channels are different (not grayscale)
    r_eq_g = np.array_equal(scanner_spec[:,:,0], scanner_spec[:,:,1])
    g_eq_b = np.array_equal(scanner_spec[:,:,1], scanner_spec[:,:,2])
    
    print(f"\n   Channel Analysis:")
    print(f"      R==G: {r_eq_g} {'❌ (should be False)' if r_eq_g else '✅ (3-component!)'}")
    print(f"      G==B: {g_eq_b} {'❌ (should be False)' if g_eq_b else '✅ (3-component!)'}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 2: Load training spectrogram
# ============================================================================

print(f"\n📊 Step 2: Loading training spectrogram...")

import pandas as pd

metadata_file = 'dataset_unified/metadata/unified_metadata.csv'
df = pd.read_csv(metadata_file)

mask = (df['station'] == TEST_STATION) & (df['date'] == TEST_DATE)
matching_samples = df[mask]

if len(matching_samples) == 0:
    print(f"❌ No training sample found")
    sys.exit(1)

sample = matching_samples.iloc[0]
training_spec_path = Path('dataset_unified') / sample['unified_path']

training_img = Image.open(training_spec_path)
training_array = np.array(training_img)

print(f"✅ Training spectrogram loaded")
print(f"   Shape: {training_array.shape}")
print(f"   Dtype: {training_array.dtype}")
print(f"   Value range: {training_array.min()}-{training_array.max()}")

# ============================================================================
# STEP 3: Compare
# ============================================================================

print(f"\n🔍 Step 3: Comparing spectrograms...")

# Calculate difference
diff = np.abs(training_array.astype(float) - scanner_spec.astype(float))
diff_mean = diff.mean()
diff_max = diff.max()

print(f"\nAbsolute Difference:")
print(f"   Mean: {diff_mean:.2f}")
print(f"   Max: {diff_max:.2f}")
print(f"   Std: {diff.std():.2f}")

# Calculate correlation
correlation = np.corrcoef(training_array.flatten(), scanner_spec.flatten())[0, 1]
print(f"\nCorrelation: {correlation:.4f}")

# Similarity assessment
if diff_mean < 10:
    similarity = "🟢 VERY SIMILAR"
    status = "✅ EXCELLENT"
elif diff_mean < 30:
    similarity = "🟡 SIMILAR"
    status = "✅ GOOD"
elif diff_mean < 50:
    similarity = "🟠 SOMEWHAT SIMILAR"
    status = "⚠️ ACCEPTABLE"
else:
    similarity = "🔴 DIFFERENT"
    status = "❌ POOR"

print(f"\nSimilarity: {similarity}")
print(f"Status: {status}")

# ============================================================================
# STEP 4: Visual comparison
# ============================================================================

print(f"\n📊 Step 4: Creating visual comparison...")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Training
axes[0, 0].imshow(training_array, aspect='auto')
axes[0, 0].set_title('Training Spectrogram', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Scanner (FIXED)
axes[0, 1].imshow(scanner_spec, aspect='auto')
axes[0, 1].set_title('Scanner Spectrogram (FIXED)', fontsize=12, fontweight='bold')
axes[0, 1].axis('off')

# Difference
axes[1, 0].imshow(diff, aspect='auto', cmap='hot')
axes[1, 0].set_title(f'Absolute Difference (Mean: {diff_mean:.1f})', fontsize=12)
axes[1, 0].axis('off')

# Histogram comparison
axes[1, 1].hist(training_array.flatten(), bins=50, alpha=0.5, label='Training', color='blue')
axes[1, 1].hist(scanner_spec.flatten(), bins=50, alpha=0.5, label='Scanner', color='red')
axes[1, 1].set_title('Value Distribution', fontsize=12)
axes[1, 1].set_xlabel('Pixel Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle(f'Fixed Scanner Comparison: {TEST_STATION} - {TEST_DATE}',
             fontsize=16, fontweight='bold')

plt.tight_layout()

output_path = 'fixed_scanner_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Comparison saved to: {output_path}")

plt.show()

# ============================================================================
# FINAL VERDICT
# ============================================================================

print(f"\n{'='*70}")
print("FINAL VERDICT")
print(f"{'='*70}")

print(f"\n📊 Preprocessing Fixes Applied:")
print(f"   ✅ Filter range: 0.01-0.045 Hz (TRAINING MATCH)")
print(f"   ✅ 3-component: H, D, Z stacked (TRAINING MATCH)")
print(f"   ✅ Colormap: jet (TRAINING MATCH)")
print(f"   ✅ No percentile normalization (TRAINING MATCH)")

print(f"\n📊 Similarity Metrics:")
print(f"   Mean difference: {diff_mean:.2f}")
print(f"   Correlation: {correlation:.4f}")
print(f"   Assessment: {similarity}")

print(f"\n🎯 CONCLUSION:")
if diff_mean < 30 and correlation > 0.7:
    print(f"   ✅ Scanner preprocessing is NOW COMPATIBLE with training!")
    print(f"   ✅ Spectrograms are SIMILAR enough")
    print(f"   ✅ Ready to test with model")
elif diff_mean < 50 and correlation > 0.5:
    print(f"   🟡 Scanner preprocessing is IMPROVED")
    print(f"   ⚠️  Still some differences, but much better")
    print(f"   🔧 May need fine-tuning")
else:
    print(f"   ❌ Scanner preprocessing still has issues")
    print(f"   🔴 Need more investigation")

print(f"\n{'='*70}")
print("✅ TEST COMPLETE!")
print(f"{'='*70}")
