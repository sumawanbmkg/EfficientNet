#!/usr/bin/env python3
"""
Test Random Scanner - 10 Random Tests
Test scanner dengan 10 kombinasi random stasiun dan tanggal
"""

import random
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add path
sys.path.insert(0, os.path.dirname(__file__))

from prekursor_scanner import PrekursorScanner

print("="*70)
print("RANDOM SCANNER TEST - 10 SAMPLES")
print("="*70)

# Load station list
station_file = 'intial/lokasi_stasiun.csv'
df_stations = pd.read_csv(station_file, sep=';')
stations = []
for _, row in df_stations.iterrows():
    code = str(row['Kode Stasiun']).strip()
    if code and code != 'nan':
        stations.append(code)

print(f"\n📍 Available stations: {len(stations)}")
print(f"   {', '.join(sorted(stations))}")

# Generate random dates (2018-2023)
start_date = datetime(2018, 1, 1)
end_date = datetime(2023, 12, 31)

# Generate 10 random combinations
random.seed(42)  # For reproducibility
test_cases = []

for i in range(10):
    # Random station
    station = random.choice(stations)
    
    # Random date
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    random_date = start_date + timedelta(days=random_days)
    
    test_cases.append({
        'id': i + 1,
        'station': station,
        'date': random_date.strftime('%Y-%m-%d')
    })

print(f"\n🎲 Generated 10 random test cases:")
for case in test_cases:
    print(f"   {case['id']}. {case['station']} - {case['date']}")

# Initialize scanner
print(f"\n🔮 Initializing scanner...")
try:
    scanner = PrekursorScanner()
    print("✅ Scanner initialized")
except Exception as e:
    print(f"❌ Failed to initialize scanner: {e}")
    sys.exit(1)

# Run tests
print(f"\n{'='*70}")
print("RUNNING TESTS")
print(f"{'='*70}")

results = []

for case in test_cases:
    print(f"\n{'='*70}")
    print(f"TEST {case['id']}/10: {case['station']} - {case['date']}")
    print(f"{'='*70}")
    
    try:
        # Run scan (without saving to avoid clutter)
        result = scanner.scan(case['date'], case['station'], save_results=False)
        
        if result:
            pred = result['predictions']
            
            # Store result
            results.append({
                'id': case['id'],
                'station': case['station'],
                'date': case['date'],
                'magnitude_class': pred['magnitude']['class_name'],
                'magnitude_conf': pred['magnitude']['confidence'],
                'azimuth_class': pred['azimuth']['class_name'],
                'azimuth_conf': pred['azimuth']['confidence'],
                'is_corrected': pred['is_corrected'],
                'is_precursor': pred['is_precursor'],
                'data_coverage': result['data_quality']['coverage'],
                'status': 'SUCCESS'
            })
            
            print(f"\n✅ RESULT:")
            print(f"   Magnitude: {pred['magnitude']['class_name']} ({pred['magnitude']['confidence']:.1f}%)")
            print(f"   Azimuth: {pred['azimuth']['class_name']} ({pred['azimuth']['confidence']:.1f}%)")
            print(f"   Precursor: {'YES ⚠️' if pred['is_precursor'] else 'NO ✅'}")
            print(f"   Corrected: {'YES ⚠️' if pred['is_corrected'] else 'NO ✅'}")
            print(f"   Data Coverage: {result['data_quality']['coverage']:.1f}%")
        else:
            results.append({
                'id': case['id'],
                'station': case['station'],
                'date': case['date'],
                'status': 'FAILED',
                'reason': 'Data not available'
            })
            print(f"\n❌ FAILED: Data not available")
    
    except Exception as e:
        results.append({
            'id': case['id'],
            'station': case['station'],
            'date': case['date'],
            'status': 'ERROR',
            'reason': str(e)
        })
        print(f"\n❌ ERROR: {e}")

# Summary
print(f"\n{'='*70}")
print("TEST SUMMARY")
print(f"{'='*70}")

successful = [r for r in results if r['status'] == 'SUCCESS']
failed = [r for r in results if r['status'] != 'SUCCESS']

print(f"\n📊 OVERALL:")
print(f"   Total tests: {len(results)}")
print(f"   Successful: {len(successful)} ✅")
print(f"   Failed: {len(failed)} ❌")

if successful:
    print(f"\n📈 SUCCESSFUL TESTS:")
    
    precursor_count = sum(1 for r in successful if r['is_precursor'])
    corrected_count = sum(1 for r in successful if r['is_corrected'])
    
    print(f"   Precursor detected: {precursor_count}/{len(successful)} ({precursor_count/len(successful)*100:.1f}%)")
    print(f"   Corrected predictions: {corrected_count}/{len(successful)} ({corrected_count/len(successful)*100:.1f}%)")
    
    avg_mag_conf = sum(r['magnitude_conf'] for r in successful) / len(successful)
    avg_az_conf = sum(r['azimuth_conf'] for r in successful) / len(successful)
    avg_coverage = sum(r['data_coverage'] for r in successful) / len(successful)
    
    print(f"   Avg magnitude confidence: {avg_mag_conf:.1f}%")
    print(f"   Avg azimuth confidence: {avg_az_conf:.1f}%")
    print(f"   Avg data coverage: {avg_coverage:.1f}%")
    
    print(f"\n📋 DETAILED RESULTS:")
    for r in successful:
        precursor_icon = "⚠️" if r['is_precursor'] else "✅"
        corrected_icon = "⚠️" if r['is_corrected'] else "✅"
        
        print(f"\n   {r['id']}. {r['station']} - {r['date']}")
        print(f"      Magnitude: {r['magnitude_class']} ({r['magnitude_conf']:.1f}%)")
        print(f"      Azimuth: {r['azimuth_class']} ({r['azimuth_conf']:.1f}%)")
        print(f"      Precursor: {precursor_icon} {'YES' if r['is_precursor'] else 'NO'}")
        print(f"      Corrected: {corrected_icon} {'YES' if r['is_corrected'] else 'NO'}")
        print(f"      Coverage: {r['data_coverage']:.1f}%")

if failed:
    print(f"\n❌ FAILED TESTS:")
    for r in failed:
        print(f"   {r['id']}. {r['station']} - {r['date']}: {r.get('reason', 'Unknown error')}")

# Save results
import json
output_file = 'random_scanner_test_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_file}")

print(f"\n{'='*70}")
print("✅ RANDOM TEST COMPLETE!")
print(f"{'='*70}")
