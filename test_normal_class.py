#!/usr/bin/env python3
"""
Test Normal Class - CRITICAL TEST
Verify if model can predict "Normal" class correctly
"""

import pandas as pd
import sys
import os
from datetime import datetime

print("="*70)
print("NORMAL CLASS TEST - CRITICAL VERIFICATION")
print("="*70)

# Initialize scanner
sys.path.insert(0, os.path.dirname(__file__))
from prekursor_scanner import PrekursorScanner

print("\n🔮 Initializing scanner...")
try:
    scanner = PrekursorScanner()
    print("✅ Scanner initialized")
except Exception as e:
    print(f"❌ Failed to initialize scanner: {e}")
    sys.exit(1)

# Get normal dates from training data
print("\n📊 Loading training data to find normal dates...")
metadata_file = 'dataset_unified/metadata/unified_metadata.csv'
df = pd.read_csv(metadata_file)

# Get normal samples
normal_samples = df[df['magnitude_class'] == 'Normal'].copy()
print(f"   Found {len(normal_samples)} normal samples in training data")

# Select diverse normal samples (different stations and dates)
test_cases = []

# Get unique stations
stations = normal_samples['station'].unique()
print(f"   Stations with normal data: {len(stations)}")

# Select 10 normal samples from different stations
for station in stations[:10]:
    station_samples = normal_samples[normal_samples['station'] == station]
    if len(station_samples) > 0:
        # Get first sample
        sample = station_samples.iloc[0]
        test_cases.append({
            'station': sample['station'],
            'date': sample['date'],
            'expected_magnitude': 'Normal',
            'expected_azimuth': 'Normal'
        })

print(f"\n✅ Selected {len(test_cases)} normal test cases")
print(f"\n🔍 Test cases:")
for i, case in enumerate(test_cases, 1):
    print(f"   {i}. {case['station']} - {case['date']}")

# Run tests
print(f"\n{'='*70}")
print("RUNNING NORMAL CLASS TESTS")
print(f"{'='*70}")

results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"TEST {i}/{len(test_cases)}: {test_case['station']} - {test_case['date']}")
    print(f"Expected: Normal / Normal")
    print(f"{'='*70}")
    
    try:
        # Run scan (without visualization to speed up)
        # Fetch data
        data = scanner.fetch_data(test_case['date'], test_case['station'])
        if data is None:
            results.append({
                'id': i,
                'station': test_case['station'],
                'date': test_case['date'],
                'status': 'FAILED',
                'reason': 'Data not available'
            })
            print(f"\n❌ FAILED: Data not available")
            continue
        
        # Generate spectrogram
        spectrogram = scanner.generate_spectrogram(data, component='Hcomp')
        if spectrogram is None:
            results.append({
                'id': i,
                'station': test_case['station'],
                'date': test_case['date'],
                'status': 'FAILED',
                'reason': 'Spectrogram generation failed'
            })
            print(f"\n❌ FAILED: Spectrogram generation failed")
            continue
        
        # Predict (without visualization)
        pred = scanner.predict(spectrogram)
        
        result = {'predictions': pred}
        
        if result:
            pred = result['predictions']
            
            # Check if prediction matches expectation
            mag_is_normal = pred['magnitude']['class_id'] == 3  # Normal class
            az_is_normal = pred['azimuth']['class_id'] == 0     # Normal class
            
            results.append({
                'id': i,
                'station': test_case['station'],
                'date': test_case['date'],
                'expected_magnitude': 'Normal',
                'expected_azimuth': 'Normal',
                'predicted_magnitude': pred['magnitude']['class_name'],
                'predicted_azimuth': pred['azimuth']['class_name'],
                'magnitude_conf': pred['magnitude']['confidence'],
                'azimuth_conf': pred['azimuth']['confidence'],
                'magnitude_correct': mag_is_normal,
                'azimuth_correct': az_is_normal,
                'both_correct': mag_is_normal and az_is_normal,
                'is_corrected': pred['is_corrected'],
                'is_precursor': pred['is_precursor'],
                'status': 'SUCCESS'
            })
            
            mag_icon = "✅" if mag_is_normal else "❌"
            az_icon = "✅" if az_is_normal else "❌"
            
            print(f"\n📊 RESULT:")
            print(f"   Predicted: {pred['magnitude']['class_name']} ({pred['magnitude']['confidence']:.1f}%) {mag_icon}")
            print(f"              {pred['azimuth']['class_name']} ({pred['azimuth']['confidence']:.1f}%) {az_icon}")
            print(f"   Precursor: {'YES ⚠️' if pred['is_precursor'] else 'NO ✅'}")
            print(f"   Corrected: {'YES ⚠️' if pred['is_corrected'] else 'NO ✅'}")
            
            if not (mag_is_normal and az_is_normal):
                print(f"\n   🚨 FAILED: Model did NOT predict Normal!")
        else:
            results.append({
                'id': i,
                'station': test_case['station'],
                'date': test_case['date'],
                'status': 'FAILED',
                'reason': 'Data not available'
            })
            print(f"\n❌ FAILED: Data not available")
    
    except Exception as e:
        results.append({
            'id': i,
            'station': test_case['station'],
            'date': test_case['date'],
            'status': 'ERROR',
            'reason': str(e)
        })
        print(f"\n❌ ERROR: {e}")

# Analysis
print(f"\n{'='*70}")
print("NORMAL CLASS TEST - RESULTS ANALYSIS")
print(f"{'='*70}")

successful = [r for r in results if r['status'] == 'SUCCESS']
failed = [r for r in results if r['status'] != 'SUCCESS']

print(f"\n📊 OVERALL:")
print(f"   Total tests: {len(results)}")
print(f"   Successful: {len(successful)}")
print(f"   Failed: {len(failed)}")

if successful:
    # Calculate accuracy
    mag_correct = sum(1 for r in successful if r['magnitude_correct'])
    az_correct = sum(1 for r in successful if r['azimuth_correct'])
    both_correct = sum(1 for r in successful if r['both_correct'])
    
    print(f"\n🎯 ACCURACY:")
    print(f"   Magnitude (Normal): {mag_correct}/{len(successful)} ({mag_correct/len(successful)*100:.1f}%)")
    print(f"   Azimuth (Normal): {az_correct}/{len(successful)} ({az_correct/len(successful)*100:.1f}%)")
    print(f"   Both correct: {both_correct}/{len(successful)} ({both_correct/len(successful)*100:.1f}%)")
    
    # Check what model predicted instead
    print(f"\n🔍 PREDICTION ANALYSIS:")
    
    from collections import Counter
    mag_preds = [r['predicted_magnitude'] for r in successful]
    az_preds = [r['predicted_azimuth'] for r in successful]
    
    mag_counter = Counter(mag_preds)
    az_counter = Counter(az_preds)
    
    print(f"\n   Magnitude predictions:")
    for mag, count in mag_counter.most_common():
        print(f"      {mag}: {count}/{len(successful)} ({count/len(successful)*100:.1f}%)")
    
    print(f"\n   Azimuth predictions:")
    for az, count in az_counter.most_common():
        print(f"      {az}: {count}/{len(successful)} ({count/len(successful)*100:.1f}%)")
    
    # Check corrections
    corrected_count = sum(1 for r in successful if r['is_corrected'])
    precursor_count = sum(1 for r in successful if r['is_precursor'])
    
    print(f"\n⚠️  POST-PROCESSING:")
    print(f"   Corrected predictions: {corrected_count}/{len(successful)} ({corrected_count/len(successful)*100:.1f}%)")
    print(f"   Precursor detected: {precursor_count}/{len(successful)} ({precursor_count/len(successful)*100:.1f}%)")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for r in successful:
        mag_icon = "✅" if r['magnitude_correct'] else "❌"
        az_icon = "✅" if r['azimuth_correct'] else "❌"
        
        print(f"\n   {r['id']}. {r['station']} - {r['date']}")
        print(f"      Expected: Normal / Normal")
        print(f"      Predicted: {r['predicted_magnitude']} {mag_icon} / {r['predicted_azimuth']} {az_icon}")
        print(f"      Confidence: {r['magnitude_conf']:.1f}% / {r['azimuth_conf']:.1f}%")
        if r['is_corrected']:
            print(f"      ⚠️  Corrected by post-processing")
        if r['is_precursor']:
            print(f"      🚨 Incorrectly detected as precursor!")

# Save results
import json
results_file = 'normal_class_test_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {results_file}")

# Final verdict
print(f"\n{'='*70}")
print("FINAL VERDICT")
print(f"{'='*70}")

if successful:
    both_correct_pct = both_correct / len(successful) * 100
    
    print(f"\n🎯 CAN MODEL PREDICT 'NORMAL' CLASS?")
    
    if both_correct_pct >= 90:
        print(f"   ✅ YES - Model can predict Normal correctly ({both_correct_pct:.1f}%)")
    elif both_correct_pct >= 70:
        print(f"   ⚠️  PARTIALLY - Model sometimes predicts Normal ({both_correct_pct:.1f}%)")
    elif both_correct_pct >= 50:
        print(f"   ⚠️  POOR - Model rarely predicts Normal ({both_correct_pct:.1f}%)")
    else:
        print(f"   ❌ NO - Model CANNOT predict Normal ({both_correct_pct:.1f}%)")
        print(f"   🚨 CRITICAL ISSUE: Model fails on Normal class!")
    
    # Check if model always predicts earthquake
    if precursor_count == len(successful):
        print(f"\n   🚨 CRITICAL: Model ALWAYS predicts earthquake (precursor)!")
        print(f"   🚨 Model NEVER predicts Normal!")
    elif precursor_count > len(successful) * 0.5:
        print(f"\n   ⚠️  WARNING: Model mostly predicts earthquake ({precursor_count/len(successful)*100:.1f}%)")
    
    # Check post-processing impact
    if corrected_count > 0:
        print(f"\n   ⚠️  Post-processing corrected {corrected_count} predictions")
        print(f"   Without post-processing, accuracy would be even worse!")
else:
    print(f"\n❌ NO SUCCESSFUL TESTS - Cannot determine if model can predict Normal")

print(f"\n{'='*70}")
print("✅ NORMAL CLASS TEST COMPLETE!")
print(f"{'='*70}")
