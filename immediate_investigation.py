#!/usr/bin/env python3
"""
Immediate Investigation - Critical Model Analysis
1. Test dengan known earthquake dates
2. Test dengan normal dates
3. Analyze training data distribution
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import json

print("="*70)
print("IMMEDIATE INVESTIGATION - CRITICAL MODEL ANALYSIS")
print("="*70)

# ============================================================================
# PART 1: ANALYZE TRAINING DATA DISTRIBUTION
# ============================================================================

print("\n" + "="*70)
print("PART 1: TRAINING DATA DISTRIBUTION ANALYSIS")
print("="*70)

# Load training data
metadata_file = 'dataset_unified/metadata/unified_metadata.csv'
df = pd.read_csv(metadata_file)

print(f"\n📊 Total samples: {len(df)}")

# Magnitude distribution
print(f"\n📏 MAGNITUDE CLASS DISTRIBUTION:")
mag_dist = df['magnitude_class'].value_counts()
for mag_class, count in mag_dist.items():
    percentage = count / len(df) * 100
    print(f"   {mag_class:12s}: {count:4d} ({percentage:5.1f}%)")

# Azimuth distribution
print(f"\n🧭 AZIMUTH CLASS DISTRIBUTION:")
az_dist = df['azimuth_class'].value_counts()
for az_class, count in az_dist.items():
    percentage = count / len(df) * 100
    print(f"   {az_class:12s}: {count:4d} ({percentage:5.1f}%)")

# Combined distribution (Magnitude + Azimuth)
print(f"\n🔗 COMBINED DISTRIBUTION (Top 10):")
combined = df.groupby(['magnitude_class', 'azimuth_class']).size().sort_values(ascending=False)
print("\n   Magnitude    Azimuth      Count    Percentage")
print("   " + "-"*60)
for (mag, az), count in combined.head(10).items():
    percentage = count / len(df) * 100
    print(f"   {mag:12s} {az:12s} {count:4d}     {percentage:5.1f}%")

# Check for bias
print(f"\n⚠️  BIAS ANALYSIS:")
dominant_combo = combined.head(1)
dominant_mag, dominant_az = dominant_combo.index[0]
dominant_count = dominant_combo.values[0]
dominant_pct = dominant_count / len(df) * 100

print(f"   Most common combination: {dominant_mag} + {dominant_az}")
print(f"   Count: {dominant_count} ({dominant_pct:.1f}%)")

if dominant_pct > 30:
    print(f"   🚨 WARNING: Dominant class > 30% - HIGH BIAS RISK!")
else:
    print(f"   ✅ OK: Dominant class < 30% - acceptable balance")

# Check Normal class
normal_count = len(df[df['magnitude_class'] == 'Normal'])
normal_pct = normal_count / len(df) * 100
print(f"\n📊 NORMAL CLASS:")
print(f"   Count: {normal_count} ({normal_pct:.1f}%)")
print(f"   Should be: ~45% (888/1972)")

# ============================================================================
# PART 2: GET KNOWN EARTHQUAKE DATES
# ============================================================================

print("\n" + "="*70)
print("PART 2: KNOWN EARTHQUAKE DATES")
print("="*70)

# Get known earthquake dates from training data
print(f"\n📋 Using training data for earthquake events")

# Get earthquake samples (not Normal)
eq_samples = df[df['magnitude_class'] != 'Normal'].copy()
print(f"   Found {len(eq_samples)} earthquake samples in training data")

# Select diverse samples
test_earthquakes = []

# Get samples with different magnitudes
for mag_class in ['Large', 'Medium', 'Moderate']:
    mag_samples = eq_samples[eq_samples['magnitude_class'] == mag_class]
    if len(mag_samples) > 0:
        # Get first sample
        sample = mag_samples.iloc[0]
        test_earthquakes.append({
            'date': sample['date'],
            'station': sample['station'],
            'magnitude': sample['magnitude'],
            'azimuth': sample['azimuth'],
            'type': 'earthquake',
            'expected_mag': mag_class,
            'expected_az': sample['azimuth_class']
        })

print(f"\n✅ Selected {len(test_earthquakes)} earthquake events for testing")
print(f"\n🔍 Sample earthquake events:")
for eq in test_earthquakes:
    print(f"   {eq['station']} - {eq['date']}: {eq['expected_mag']} / {eq['expected_az']}")

# ============================================================================
# PART 3: GET KNOWN NORMAL DATES
# ============================================================================

print("\n" + "="*70)
print("PART 3: KNOWN NORMAL DATES")
print("="*70)

# Load quiet days
quiet_file = 'quiet_days.csv'
if os.path.exists(quiet_file):
    df_quiet = pd.read_csv(quiet_file)
    print(f"\n📋 Loaded {len(df_quiet)} quiet days")
    
    # Show first few
    print(f"\n🔍 Sample quiet days:")
    print(df_quiet.head(10).to_string(index=False))
    
    # Select 5 normal dates for testing
    test_normal = []
    
    # Get diverse stations
    stations = ['SCN', 'GTO', 'ALR', 'SBG', 'CLP']
    for i, station in enumerate(stations[:5]):
        if i < len(df_quiet):
            date = df_quiet.iloc[i]['date'] if 'date' in df_quiet.columns else df_quiet.iloc[i][0]
            test_normal.append({
                'date': date,
                'station': station,
                'type': 'normal',
                'expected_mag': 'Normal',
                'expected_az': 'Normal'
            })
    
    print(f"\n✅ Selected {len(test_normal)} normal dates for testing")
else:
    print(f"\n❌ Quiet days file not found: {quiet_file}")
    # Create some normal dates manually
    test_normal = [
        {'date': '2019-01-15', 'station': 'SCN', 'type': 'normal', 'expected_mag': 'Normal', 'expected_az': 'Normal'},
        {'date': '2019-06-20', 'station': 'GTO', 'type': 'normal', 'expected_mag': 'Normal', 'expected_az': 'Normal'},
        {'date': '2020-03-10', 'station': 'ALR', 'type': 'normal', 'expected_mag': 'Normal', 'expected_az': 'Normal'},
        {'date': '2020-09-05', 'station': 'SBG', 'type': 'normal', 'expected_mag': 'Normal', 'expected_az': 'Normal'},
        {'date': '2021-02-14', 'station': 'CLP', 'type': 'normal', 'expected_mag': 'Normal', 'expected_az': 'Normal'},
    ]
    print(f"\n⚠️  Using manually created normal dates: {len(test_normal)}")

# ============================================================================
# PART 4: TEST WITH SCANNER
# ============================================================================

print("\n" + "="*70)
print("PART 4: TESTING WITH SCANNER")
print("="*70)

# Combine test cases
all_tests = test_earthquakes + test_normal

print(f"\n📋 Total test cases: {len(all_tests)}")
print(f"   Earthquake events: {len(test_earthquakes)}")
print(f"   Normal dates: {len(test_normal)}")

# Save test cases
test_cases_file = 'immediate_test_cases.json'
with open(test_cases_file, 'w') as f:
    json.dump(all_tests, f, indent=2, default=str)

print(f"\n💾 Test cases saved to: {test_cases_file}")

# Now run scanner tests
print(f"\n🔮 Initializing scanner...")

sys.path.insert(0, os.path.dirname(__file__))
from prekursor_scanner import PrekursorScanner

try:
    scanner = PrekursorScanner()
    print("✅ Scanner initialized")
except Exception as e:
    print(f"❌ Failed to initialize scanner: {e}")
    sys.exit(1)

# Run tests
results = []

for i, test_case in enumerate(all_tests, 1):
    print(f"\n{'='*70}")
    print(f"TEST {i}/{len(all_tests)}: {test_case['station']} - {test_case['date']}")
    print(f"Type: {test_case['type'].upper()}")
    print(f"Expected: Magnitude={test_case['expected_mag']}, Azimuth={test_case['expected_az']}")
    print(f"{'='*70}")
    
    try:
        # Convert date to string if needed
        date_str = str(test_case['date'])
        if 'T' in date_str:
            date_str = date_str.split('T')[0]
        
        # Run scan
        result = scanner.scan(date_str, test_case['station'], save_results=False)
        
        if result:
            pred = result['predictions']
            
            # Check if prediction matches expectation
            mag_match = pred['magnitude']['class_name'].startswith(test_case['expected_mag'])
            az_match = pred['azimuth']['class_name'].startswith(test_case['expected_az'])
            
            results.append({
                'id': i,
                'type': test_case['type'],
                'station': test_case['station'],
                'date': date_str,
                'expected_magnitude': test_case['expected_mag'],
                'expected_azimuth': test_case['expected_az'],
                'predicted_magnitude': pred['magnitude']['class_name'],
                'predicted_azimuth': pred['azimuth']['class_name'],
                'magnitude_conf': pred['magnitude']['confidence'],
                'azimuth_conf': pred['azimuth']['confidence'],
                'magnitude_match': mag_match,
                'azimuth_match': az_match,
                'is_corrected': pred['is_corrected'],
                'is_precursor': pred['is_precursor'],
                'status': 'SUCCESS'
            })
            
            match_icon_mag = "✅" if mag_match else "❌"
            match_icon_az = "✅" if az_match else "❌"
            
            print(f"\n📊 RESULT:")
            print(f"   Expected: {test_case['expected_mag']} / {test_case['expected_az']}")
            print(f"   Predicted: {pred['magnitude']['class_name']} ({pred['magnitude']['confidence']:.1f}%) {match_icon_mag}")
            print(f"              {pred['azimuth']['class_name']} ({pred['azimuth']['confidence']:.1f}%) {match_icon_az}")
            print(f"   Precursor: {'YES ⚠️' if pred['is_precursor'] else 'NO ✅'}")
            print(f"   Corrected: {'YES ⚠️' if pred['is_corrected'] else 'NO ✅'}")
        else:
            results.append({
                'id': i,
                'type': test_case['type'],
                'station': test_case['station'],
                'date': date_str,
                'status': 'FAILED',
                'reason': 'Data not available'
            })
            print(f"\n❌ FAILED: Data not available")
    
    except Exception as e:
        results.append({
            'id': i,
            'type': test_case['type'],
            'station': test_case['station'],
            'date': date_str,
            'status': 'ERROR',
            'reason': str(e)
        })
        print(f"\n❌ ERROR: {e}")

# ============================================================================
# PART 5: ANALYSIS OF RESULTS
# ============================================================================

print(f"\n{'='*70}")
print("PART 5: RESULTS ANALYSIS")
print(f"{'='*70}")

successful = [r for r in results if r['status'] == 'SUCCESS']
failed = [r for r in results if r['status'] != 'SUCCESS']

print(f"\n📊 OVERALL:")
print(f"   Total tests: {len(results)}")
print(f"   Successful: {len(successful)}")
print(f"   Failed: {len(failed)}")

if successful:
    # Separate by type
    earthquake_results = [r for r in successful if r['type'] == 'earthquake']
    normal_results = [r for r in successful if r['type'] == 'normal']
    
    print(f"\n📈 BY TYPE:")
    print(f"   Earthquake tests: {len(earthquake_results)}")
    print(f"   Normal tests: {len(normal_results)}")
    
    # Earthquake results
    if earthquake_results:
        print(f"\n🌍 EARTHQUAKE TESTS:")
        mag_correct = sum(1 for r in earthquake_results if r['magnitude_match'])
        az_correct = sum(1 for r in earthquake_results if r['azimuth_match'])
        
        print(f"   Magnitude accuracy: {mag_correct}/{len(earthquake_results)} ({mag_correct/len(earthquake_results)*100:.1f}%)")
        print(f"   Azimuth accuracy: {az_correct}/{len(earthquake_results)} ({az_correct/len(earthquake_results)*100:.1f}%)")
        
        print(f"\n   Detailed results:")
        for r in earthquake_results:
            mag_icon = "✅" if r['magnitude_match'] else "❌"
            az_icon = "✅" if r['azimuth_match'] else "❌"
            print(f"   {r['id']}. {r['station']} - {r['date']}")
            print(f"      Expected: {r['expected_magnitude']} / {r['expected_azimuth']}")
            print(f"      Predicted: {r['predicted_magnitude']} {mag_icon} / {r['predicted_azimuth']} {az_icon}")
    
    # Normal results
    if normal_results:
        print(f"\n✅ NORMAL TESTS:")
        mag_correct = sum(1 for r in normal_results if r['magnitude_match'])
        az_correct = sum(1 for r in normal_results if r['azimuth_match'])
        
        print(f"   Magnitude accuracy: {mag_correct}/{len(normal_results)} ({mag_correct/len(normal_results)*100:.1f}%)")
        print(f"   Azimuth accuracy: {az_correct}/{len(normal_results)} ({az_correct/len(normal_results)*100:.1f}%)")
        
        print(f"\n   Detailed results:")
        for r in normal_results:
            mag_icon = "✅" if r['magnitude_match'] else "❌"
            az_icon = "✅" if r['azimuth_match'] else "❌"
            print(f"   {r['id']}. {r['station']} - {r['date']}")
            print(f"      Expected: {r['expected_magnitude']} / {r['expected_azimuth']}")
            print(f"      Predicted: {r['predicted_magnitude']} {mag_icon} / {r['predicted_azimuth']} {az_icon}")
    
    # Check for bias
    print(f"\n🔍 BIAS ANALYSIS:")
    all_mag_preds = [r['predicted_magnitude'] for r in successful]
    all_az_preds = [r['predicted_azimuth'] for r in successful]
    
    from collections import Counter
    mag_counter = Counter(all_mag_preds)
    az_counter = Counter(all_az_preds)
    
    print(f"\n   Magnitude predictions:")
    for mag, count in mag_counter.most_common():
        print(f"      {mag}: {count}/{len(successful)} ({count/len(successful)*100:.1f}%)")
    
    print(f"\n   Azimuth predictions:")
    for az, count in az_counter.most_common():
        print(f"      {az}: {count}/{len(successful)} ({count/len(successful)*100:.1f}%)")
    
    # Check if model always predicts same thing
    if len(mag_counter) == 1:
        print(f"\n   🚨 WARNING: Model ALWAYS predicts {list(mag_counter.keys())[0]} for magnitude!")
    if len(az_counter) == 1:
        print(f"\n   🚨 WARNING: Model ALWAYS predicts {list(az_counter.keys())[0]} for azimuth!")

# Save results
results_file = 'immediate_investigation_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {results_file}")

print(f"\n{'='*70}")
print("✅ IMMEDIATE INVESTIGATION COMPLETE!")
print(f"{'='*70}")
