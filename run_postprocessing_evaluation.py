#!/usr/bin/env python3
"""
Post-Processing Evaluation - Measure Impact on Validation Set
Evaluate consistency rules on validation data to ensure optimal performance

Author: Earthquake Prediction Research Team
Date: 2 February 2026
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from earthquake_cnn_v3 import EarthquakeCNNV3

print("="*70)
print("POST-PROCESSING EVALUATION - VALIDATION SET")
print("="*70)

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {DEVICE}")

# Load validation data
print("\n📊 Loading validation data...")
metadata_file = 'dataset_unified/metadata/unified_metadata.csv'
df = pd.read_csv(metadata_file)

# Split data (same as training)
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['magnitude_class'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['magnitude_class'])

print(f"   Validation samples: {len(val_df)}")

# Load model
print("\n🔮 Loading model...")
model_path = None

# Find best model
exp_dir = Path('experiments_v4')
if exp_dir.exists():
    exp_folders = sorted(exp_dir.glob('exp_v4_phase1_*'))
    if exp_folders:
        latest_exp = exp_folders[-1]
        model_path = latest_exp / 'best_model.pth'

if model_path is None or not model_path.exists():
    print("❌ Model not found! Please train model first.")
    sys.exit(1)

print(f"   Model: {model_path}")

# Create model
model = EarthquakeCNNV3(
    num_magnitude_classes=4,
    num_azimuth_classes=9,
    dropout_rate=0.3
)

# Load checkpoint
checkpoint = torch.load(model_path, map_location=DEVICE)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")

# Class mappings
magnitude_mapping = {
    0: 'Large',
    1: 'Medium',
    2: 'Moderate',
    3: 'Normal'
}

azimuth_mapping = {
    0: 'Normal',
    1: 'N',
    2: 'S',
    3: 'NW',
    4: 'W',
    5: 'E',
    6: 'NE',
    7: 'SE',
    8: 'SW'
}

# Evaluation metrics
results = {
    'total_samples': 0,
    'corrected_samples': 0,
    'correction_rate': 0.0,
    'corrections_by_rule': {
        'rule1': 0,  # Magnitude Normal → Azimuth Normal
        'rule2': 0,  # Azimuth Normal → Magnitude Normal
        'rule3': 0   # Low confidence → Both Normal
    },
    'raw_predictions': {
        'magnitude': [],
        'azimuth': [],
        'magnitude_conf': [],
        'azimuth_conf': []
    },
    'corrected_predictions': {
        'magnitude': [],
        'azimuth': [],
        'magnitude_conf': [],
        'azimuth_conf': []
    },
    'ground_truth': {
        'magnitude': [],
        'azimuth': []
    },
    'is_corrected': [],
    'is_precursor': []
}

print("\n🔍 Evaluating validation set...")
print("   Applying post-processing rules...")

# Process validation samples
for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Processing"):
    # Load spectrogram
    spec_path = f"dataset_unified/{row['unified_path']}"
    
    if not os.path.exists(spec_path):
        continue
    
    # Load image
    from PIL import Image
    img = Image.open(spec_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    # Prepare input
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        magnitude_logits, azimuth_logits = model(img_tensor)
        
        # Get probabilities
        magnitude_probs = torch.softmax(magnitude_logits, dim=1)
        azimuth_probs = torch.softmax(azimuth_logits, dim=1)
        
        # Get raw predictions
        magnitude_pred_raw = torch.argmax(magnitude_probs, dim=1).item()
        azimuth_pred_raw = torch.argmax(azimuth_probs, dim=1).item()
        
        magnitude_conf_raw = magnitude_probs[0, magnitude_pred_raw].item() * 100
        azimuth_conf_raw = azimuth_probs[0, azimuth_pred_raw].item() * 100
        
        # Apply consistency rules
        magnitude_pred = magnitude_pred_raw
        azimuth_pred = azimuth_pred_raw
        magnitude_conf = magnitude_conf_raw
        azimuth_conf = azimuth_conf_raw
        is_corrected = False
        correction_rule = None
        
        # Rule 1: Magnitude Normal → Azimuth Normal
        if magnitude_pred_raw == 3 and azimuth_pred_raw != 0:
            azimuth_pred = 0
            azimuth_conf = azimuth_probs[0, 0].item() * 100
            is_corrected = True
            correction_rule = 'rule1'
        
        # Rule 2: Azimuth Normal → Magnitude Normal
        elif azimuth_pred_raw == 0 and magnitude_pred_raw != 3:
            magnitude_pred = 3
            magnitude_conf = magnitude_probs[0, 3].item() * 100
            is_corrected = True
            correction_rule = 'rule2'
        
        # Rule 3: Low confidence → Both Normal
        elif magnitude_pred_raw != 3 and azimuth_pred_raw != 0:
            avg_conf = (magnitude_conf_raw + azimuth_conf_raw) / 2
            
            if avg_conf < 40.0:
                magnitude_pred = 3
                azimuth_pred = 0
                magnitude_conf = magnitude_probs[0, 3].item() * 100
                azimuth_conf = azimuth_probs[0, 0].item() * 100
                is_corrected = True
                correction_rule = 'rule3'
        
        # Calculate precursor status
        is_precursor = magnitude_pred != 3 and azimuth_pred != 0
        
        # Store results
        results['total_samples'] += 1
        if is_corrected:
            results['corrected_samples'] += 1
            results['corrections_by_rule'][correction_rule] += 1
        
        # Store predictions
        results['raw_predictions']['magnitude'].append(magnitude_pred_raw)
        results['raw_predictions']['azimuth'].append(azimuth_pred_raw)
        results['raw_predictions']['magnitude_conf'].append(magnitude_conf_raw)
        results['raw_predictions']['azimuth_conf'].append(azimuth_conf_raw)
        
        results['corrected_predictions']['magnitude'].append(magnitude_pred)
        results['corrected_predictions']['azimuth'].append(azimuth_pred)
        results['corrected_predictions']['magnitude_conf'].append(magnitude_conf)
        results['corrected_predictions']['azimuth_conf'].append(azimuth_conf)
        
        results['is_corrected'].append(is_corrected)
        results['is_precursor'].append(is_precursor)
        
        # Ground truth
        mag_class = row['magnitude_class']
        az_class = row['azimuth_class']
        
        # Convert to class IDs
        mag_id = {'Large': 0, 'Medium': 1, 'Moderate': 2, 'Normal': 3}[mag_class]
        az_id = {'Normal': 0, 'N': 1, 'S': 2, 'NW': 3, 'W': 4, 'E': 5, 'NE': 6, 'SE': 7, 'SW': 8}[az_class]
        
        results['ground_truth']['magnitude'].append(mag_id)
        results['ground_truth']['azimuth'].append(az_id)

# Calculate correction rate
results['correction_rate'] = (results['corrected_samples'] / results['total_samples']) * 100

# Print results
print("\n" + "="*70)
print("POST-PROCESSING EVALUATION RESULTS")
print("="*70)

print(f"\n📊 OVERALL STATISTICS:")
print(f"   Total samples: {results['total_samples']}")
print(f"   Corrected samples: {results['corrected_samples']}")
print(f"   Correction rate: {results['correction_rate']:.2f}%")

print(f"\n📋 CORRECTIONS BY RULE:")
print(f"   Rule 1 (Magnitude Normal → Azimuth Normal): {results['corrections_by_rule']['rule1']} ({results['corrections_by_rule']['rule1']/results['total_samples']*100:.2f}%)")
print(f"   Rule 2 (Azimuth Normal → Magnitude Normal): {results['corrections_by_rule']['rule2']} ({results['corrections_by_rule']['rule2']/results['total_samples']*100:.2f}%)")
print(f"   Rule 3 (Low confidence → Both Normal): {results['corrections_by_rule']['rule3']} ({results['corrections_by_rule']['rule3']/results['total_samples']*100:.2f}%)")

# Calculate accuracy before and after
from sklearn.metrics import accuracy_score, classification_report

# Before post-processing
raw_mag_acc = accuracy_score(results['ground_truth']['magnitude'], results['raw_predictions']['magnitude'])
raw_az_acc = accuracy_score(results['ground_truth']['azimuth'], results['raw_predictions']['azimuth'])

# After post-processing
corrected_mag_acc = accuracy_score(results['ground_truth']['magnitude'], results['corrected_predictions']['magnitude'])
corrected_az_acc = accuracy_score(results['ground_truth']['azimuth'], results['corrected_predictions']['azimuth'])

print(f"\n🎯 ACCURACY COMPARISON:")
print(f"   BEFORE Post-Processing:")
print(f"      Magnitude: {raw_mag_acc*100:.2f}%")
print(f"      Azimuth: {raw_az_acc*100:.2f}%")
print(f"   AFTER Post-Processing:")
print(f"      Magnitude: {corrected_mag_acc*100:.2f}%")
print(f"      Azimuth: {corrected_az_acc*100:.2f}%")
print(f"   IMPROVEMENT:")
print(f"      Magnitude: {(corrected_mag_acc - raw_mag_acc)*100:+.2f}%")
print(f"      Azimuth: {(corrected_az_acc - raw_az_acc)*100:+.2f}%")

# Check consistency
print(f"\n✅ CONSISTENCY CHECK:")
raw_inconsistent = 0
corrected_inconsistent = 0

for i in range(len(results['raw_predictions']['magnitude'])):
    # Raw predictions
    raw_mag = results['raw_predictions']['magnitude'][i]
    raw_az = results['raw_predictions']['azimuth'][i]
    
    # Check inconsistency
    if (raw_mag == 3 and raw_az != 0) or (raw_az == 0 and raw_mag != 3):
        raw_inconsistent += 1
    
    # Corrected predictions
    corr_mag = results['corrected_predictions']['magnitude'][i]
    corr_az = results['corrected_predictions']['azimuth'][i]
    
    # Check inconsistency
    if (corr_mag == 3 and corr_az != 0) or (corr_az == 0 and corr_mag != 3):
        corrected_inconsistent += 1

print(f"   BEFORE Post-Processing:")
print(f"      Inconsistent predictions: {raw_inconsistent} ({raw_inconsistent/results['total_samples']*100:.2f}%)")
print(f"   AFTER Post-Processing:")
print(f"      Inconsistent predictions: {corrected_inconsistent} ({corrected_inconsistent/results['total_samples']*100:.2f}%)")

# Interpretation
print(f"\n💡 INTERPRETATION:")
if results['correction_rate'] < 10:
    print(f"   ✅ EXCELLENT: Correction rate < 10%")
    print(f"      Model predictions are highly consistent.")
    print(f"      Post-processing is working optimally.")
elif results['correction_rate'] < 30:
    print(f"   ✅ GOOD: Correction rate < 30%")
    print(f"      Model predictions are mostly consistent.")
    print(f"      Post-processing is effective.")
else:
    print(f"   ⚠️  HIGH: Correction rate ≥ 30%")
    print(f"      Model predictions need frequent correction.")
    print(f"      Consider re-training with consistency loss.")

# Save results
output_dir = Path('postprocessing_evaluation')
output_dir.mkdir(exist_ok=True)

# Save detailed results
results_file = output_dir / 'evaluation_results.json'
with open(results_file, 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'total_samples': results['total_samples'],
        'corrected_samples': results['corrected_samples'],
        'correction_rate': results['correction_rate'],
        'corrections_by_rule': results['corrections_by_rule'],
        'accuracy': {
            'before': {
                'magnitude': float(raw_mag_acc),
                'azimuth': float(raw_az_acc)
            },
            'after': {
                'magnitude': float(corrected_mag_acc),
                'azimuth': float(corrected_az_acc)
            }
        },
        'consistency': {
            'before': {
                'inconsistent_count': raw_inconsistent,
                'inconsistent_rate': raw_inconsistent/results['total_samples']*100
            },
            'after': {
                'inconsistent_count': corrected_inconsistent,
                'inconsistent_rate': corrected_inconsistent/results['total_samples']*100
            }
        }
    }
    json.dump(json_results, f, indent=2)

print(f"\n💾 Results saved to: {results_file}")

# Create visualization
print(f"\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Post-Processing Evaluation Results', fontsize=16, fontweight='bold')

# 1. Correction rate by rule
ax1 = axes[0, 0]
rules = ['Rule 1\n(Mag→Az)', 'Rule 2\n(Az→Mag)', 'Rule 3\n(Low Conf)']
counts = [
    results['corrections_by_rule']['rule1'],
    results['corrections_by_rule']['rule2'],
    results['corrections_by_rule']['rule3']
]
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
bars = ax1.bar(rules, counts, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Number of Corrections', fontweight='bold')
ax1.set_title('Corrections by Rule', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({count/results["total_samples"]*100:.1f}%)',
            ha='center', va='bottom', fontweight='bold')

# 2. Accuracy comparison
ax2 = axes[0, 1]
categories = ['Magnitude', 'Azimuth']
before = [raw_mag_acc * 100, raw_az_acc * 100]
after = [corrected_mag_acc * 100, corrected_az_acc * 100]

x = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x - width/2, before, width, label='Before', color='#ff6b6b', alpha=0.7, edgecolor='black')
bars2 = ax2.bar(x + width/2, after, width, label='After', color='#4ecdc4', alpha=0.7, edgecolor='black')

ax2.set_ylabel('Accuracy (%)', fontweight='bold')
ax2.set_title('Accuracy: Before vs After Post-Processing', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 100)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Consistency comparison
ax3 = axes[1, 0]
consistency_data = [
    raw_inconsistent / results['total_samples'] * 100,
    corrected_inconsistent / results['total_samples'] * 100
]
labels = ['Before\nPost-Processing', 'After\nPost-Processing']
colors_cons = ['#ff6b6b', '#4ecdc4']
bars = ax3.bar(labels, consistency_data, color=colors_cons, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Inconsistent Predictions (%)', fontweight='bold')
ax3.set_title('Consistency: Before vs After', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, consistency_data):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%',
            ha='center', va='bottom', fontweight='bold')

# 4. Overall summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
POST-PROCESSING SUMMARY

Total Samples: {results['total_samples']}
Corrected: {results['corrected_samples']} ({results['correction_rate']:.1f}%)

CORRECTIONS BY RULE:
  Rule 1: {results['corrections_by_rule']['rule1']} ({results['corrections_by_rule']['rule1']/results['total_samples']*100:.1f}%)
  Rule 2: {results['corrections_by_rule']['rule2']} ({results['corrections_by_rule']['rule2']/results['total_samples']*100:.1f}%)
  Rule 3: {results['corrections_by_rule']['rule3']} ({results['corrections_by_rule']['rule3']/results['total_samples']*100:.1f}%)

ACCURACY IMPROVEMENT:
  Magnitude: {(corrected_mag_acc - raw_mag_acc)*100:+.2f}%
  Azimuth: {(corrected_az_acc - raw_az_acc)*100:+.2f}%

CONSISTENCY:
  Before: {raw_inconsistent/results['total_samples']*100:.1f}% inconsistent
  After: {corrected_inconsistent/results['total_samples']*100:.1f}% inconsistent
  Improvement: {(raw_inconsistent - corrected_inconsistent)/results['total_samples']*100:.1f}%

STATUS: {'✅ OPTIMAL' if results['correction_rate'] < 30 else '⚠️  NEEDS ATTENTION'}
"""

ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
        fontsize=11, verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save figure
fig_path = output_dir / 'evaluation_visualization.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"   Saved: {fig_path}")

plt.show()

# Final summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

if results['correction_rate'] < 10:
    print("\n✅ POST-PROCESSING IS OPTIMAL!")
    print("   - Correction rate < 10%")
    print("   - Model predictions are highly consistent")
    print("   - No re-training needed")
elif results['correction_rate'] < 30:
    print("\n✅ POST-PROCESSING IS EFFECTIVE!")
    print("   - Correction rate < 30%")
    print("   - Model predictions are mostly consistent")
    print("   - Post-processing is working well")
else:
    print("\n⚠️  POST-PROCESSING RATE IS HIGH")
    print("   - Correction rate ≥ 30%")
    print("   - Model predictions need frequent correction")
    print("   - Consider re-training with consistency loss")

print(f"\n🎯 RECOMMENDATION:")
if results['correction_rate'] < 30:
    print("   ✅ Continue using post-processing approach")
    print("   ✅ Model is performing optimally")
    print("   ✅ No changes needed")
else:
    print("   ⚠️  Consider re-training with consistency loss")
    print("   ⚠️  High correction rate indicates model issues")

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)
