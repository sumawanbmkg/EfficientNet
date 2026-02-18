#!/usr/bin/env python3
"""
Visual Comparison: Training vs Scanner Spectrograms
Compare spectrograms side-by-side to verify preprocessing differences
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import pandas as pd

print("="*70)
print("VISUAL COMPARISON: TRAINING VS SCANNER SPECTROGRAMS")
print("="*70)

# Test case: GTO 2021-12-05 (Normal sample)
TEST_STATION = 'GTO'
TEST_DATE = '2021-12-05'

print(f"\n📊 Test Case: {TEST_STATION} - {TEST_DATE}")
print(f"   Expected: Normal / Normal")

# ============================================================================
# STEP 1: Find training spectrogram
# ============================================================================

print(f"\n🔍 Step 1: Finding training spectrogram...")

# Load metadata
metadata_file = 'dataset_unified/metadata/unified_metadata.csv'
if not os.path.exists(metadata_file):
    print(f"❌ Metadata not found: {metadata_file}")
    sys.exit(1)

df = pd.read_csv(metadata_file)

# Find matching sample
mask = (df['station'] == TEST_STATION) & (df['date'] == TEST_DATE)
matching_samples = df[mask]

if len(matching_samples) == 0:
    print(f"❌ No training sample found for {TEST_STATION} {TEST_DATE}")
    print(f"\nAvailable Normal samples:")
    normal_samples = df[df['magnitude_class'] == 'Normal'].head(10)
    for _, row in normal_samples.iterrows():
        print(f"   {row['station']} - {row['date']}")
    sys.exit(1)

# Get first matching sample
sample = matching_samples.iloc[0]
training_spec_path = Path('dataset_unified') / sample['unified_path']

if not training_spec_path.exists():
    print(f"❌ Training spectrogram not found: {training_spec_path}")
    sys.exit(1)

print(f"✅ Found training spectrogram: {training_spec_path}")
print(f"   Magnitude class: {sample['magnitude_class']}")
print(f"   Azimuth class: {sample['azimuth_class']}")

# Load training spectrogram
training_img = Image.open(training_spec_path)
training_array = np.array(training_img)

print(f"   Shape: {training_array.shape}")
print(f"   Dtype: {training_array.dtype}")
print(f"   Value range: {training_array.min()}-{training_array.max()}")

# ============================================================================
# STEP 2: Generate scanner spectrogram
# ============================================================================

print(f"\n🎨 Step 2: Generating scanner spectrogram...")

sys.path.insert(0, os.path.dirname(__file__))
from prekursor_scanner import PrekursorScanner

try:
    scanner = PrekursorScanner()
    
    # Fetch data
    data = scanner.fetch_data(TEST_DATE, TEST_STATION)
    if data is None:
        print(f"❌ Failed to fetch data")
        sys.exit(1)
    
    # Generate spectrogram
    scanner_spec = scanner.generate_spectrogram(data, component='Hcomp')
    if scanner_spec is None:
        print(f"❌ Failed to generate spectrogram")
        sys.exit(1)
    
    print(f"✅ Scanner spectrogram generated")
    print(f"   Shape: {scanner_spec.shape}")
    print(f"   Dtype: {scanner_spec.dtype}")
    print(f"   Value range: {scanner_spec.min()}-{scanner_spec.max()}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 3: Visual comparison
# ============================================================================

print(f"\n📊 Step 3: Creating visual comparison...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Training spectrogram
axes[0, 0].imshow(training_array, aspect='auto')
axes[0, 0].set_title('Training Spectrogram\n(Full Image)', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Show each channel separately
axes[0, 1].imshow(training_array[:, :, 0], aspect='auto', cmap='Reds')
axes[0, 1].set_title('Training - Red Channel', fontsize=10)
axes[0, 1].axis('off')

axes[0, 2].imshow(training_array[:, :, 1], aspect='auto', cmap='Greens')
axes[0, 2].set_title('Training - Green Channel', fontsize=10)
axes[0, 2].axis('off')

# Row 2: Scanner spectrogram
axes[1, 0].imshow(scanner_spec, aspect='auto')
axes[1, 0].set_title('Scanner Spectrogram\n(Full Image)', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# Show each channel separately
axes[1, 1].imshow(scanner_spec[:, :, 0], aspect='auto', cmap='Reds')
axes[1, 1].set_title('Scanner - Red Channel', fontsize=10)
axes[1, 1].axis('off')

axes[1, 2].imshow(scanner_spec[:, :, 1], aspect='auto', cmap='Greens')
axes[1, 2].set_title('Scanner - Green Channel', fontsize=10)
axes[1, 2].axis('off')

# Add title
fig.suptitle(f'Spectrogram Comparison: {TEST_STATION} - {TEST_DATE}\\n' +
             f'Training (Top) vs Scanner (Bottom)',
             fontsize=16, fontweight='bold')

plt.tight_layout()

# Save comparison
output_path = 'spectrogram_comparison_visual.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Comparison saved to: {output_path}")

plt.show()

# ============================================================================
# STEP 4: Statistical comparison
# ============================================================================

print(f"\n📊 Step 4: Statistical comparison...")

print(f"\nTraining Spectrogram:")
print(f"   Shape: {training_array.shape}")
print(f"   Mean: {training_array.mean():.2f}")
print(f"   Std: {training_array.std():.2f}")
print(f"   Min: {training_array.min()}")
print(f"   Max: {training_array.max()}")
print(f"   Unique values: {len(np.unique(training_array))}")

print(f"\nScanner Spectrogram:")
print(f"   Shape: {scanner_spec.shape}")
print(f"   Mean: {scanner_spec.mean():.2f}")
print(f"   Std: {scanner_spec.std():.2f}")
print(f"   Min: {scanner_spec.min()}")
print(f"   Max: {scanner_spec.max()}")
print(f"   Unique values: {len(np.unique(scanner_spec))}")

# Check if channels are identical (grayscale)
print(f"\nChannel Analysis:")
print(f"   Training - R==G: {np.array_equal(training_array[:,:,0], training_array[:,:,1])}")
print(f"   Training - G==B: {np.array_equal(training_array[:,:,1], training_array[:,:,2])}")
print(f"   Scanner - R==G: {np.array_equal(scanner_spec[:,:,0], scanner_spec[:,:,1])}")
print(f"   Scanner - G==B: {np.array_equal(scanner_spec[:,:,1], scanner_spec[:,:,2])}")

# ============================================================================
# STEP 5: Difference analysis
# ============================================================================

print(f"\n🔍 Step 5: Difference analysis...")

# Calculate absolute difference
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
elif diff_mean < 50:
    similarity = "🟡 SOMEWHAT SIMILAR"
elif diff_mean < 100:
    similarity = "🟠 DIFFERENT"
else:
    similarity = "🔴 VERY DIFFERENT"

print(f"\nSimilarity Assessment: {similarity}")

# ============================================================================
# FINAL VERDICT
# ============================================================================

print(f"\n{'='*70}")
print("FINAL VERDICT")
print(f"{'='*70}")

print(f"\n📊 Visual Comparison:")
print(f"   Training shape: {training_array.shape}")
print(f"   Scanner shape: {scanner_spec.shape}")
print(f"   Shape match: {'✅' if training_array.shape == scanner_spec.shape else '❌'}")

print(f"\n📊 Value Range:")
print(f"   Training: {training_array.min()}-{training_array.max()}")
print(f"   Scanner: {scanner_spec.min()}-{scanner_spec.max()}")
print(f"   Range match: {'✅' if training_array.min() == scanner_spec.min() and training_array.max() == scanner_spec.max() else '❌'}")

print(f"\n📊 Channel Structure:")
training_is_grayscale = np.array_equal(training_array[:,:,0], training_array[:,:,1])
scanner_is_grayscale = np.array_equal(scanner_spec[:,:,0], scanner_spec[:,:,1])
print(f"   Training is grayscale: {'✅' if training_is_grayscale else '❌'}")
print(f"   Scanner is grayscale: {'✅' if scanner_is_grayscale else '❌'}")

print(f"\n📊 Similarity:")
print(f"   Mean difference: {diff_mean:.2f}")
print(f"   Correlation: {correlation:.4f}")
print(f"   Assessment: {similarity}")

print(f"\n🎯 CONCLUSION:")
if diff_mean < 10 and correlation > 0.95:
    print(f"   ✅ Spectrograms are VERY SIMILAR")
    print(f"   ✅ Preprocessing is likely CORRECT")
    print(f"   ⚠️  Model failure must be due to other reasons")
elif diff_mean < 50 and correlation > 0.8:
    print(f"   🟡 Spectrograms are SOMEWHAT SIMILAR")
    print(f"   ⚠️  Minor preprocessing differences exist")
    print(f"   🔧 May need fine-tuning")
else:
    print(f"   ❌ Spectrograms are VERY DIFFERENT")
    print(f"   🔴 Preprocessing mismatch CONFIRMED")
    print(f"   🔧 MUST fix scanner preprocessing")

print(f"\n{'='*70}")
print("✅ VISUAL COMPARISON COMPLETE!")
print(f"{'='*70}")
print(f"\n📁 Comparison image saved to: {output_path}")
print(f"   Open this file to see visual differences!")
