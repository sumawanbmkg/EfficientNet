#!/usr/bin/env python3
"""
Quick Validation Test
Test validation system dengan beberapa events
"""

import pandas as pd
from pathlib import Path

print("="*70)
print("QUICK VALIDATION TEST")
print("="*70)

# Load earthquake events
print("\n📊 Loading earthquake events...")
metadata_file = 'dataset_unified/metadata/test_split.csv'

if not Path(metadata_file).exists():
    print(f"❌ File not found: {metadata_file}")
    exit(1)

df = pd.read_csv(metadata_file)
print(f"✅ Loaded {len(df)} samples")

# Filter earthquake events
earthquakes = df[df['magnitude_class'] != 'Normal'].copy()
print(f"✅ Found {len(earthquakes)} earthquake samples")

# Get unique events (by station + date)
earthquakes['date_only'] = pd.to_datetime(earthquakes['date']).dt.date
unique_events = earthquakes.groupby(['station', 'date_only']).first().reset_index()
print(f"✅ Found {len(unique_events)} unique earthquake events")

# Show first 10 events
print(f"\n📋 First 10 Earthquake Events:")
print("="*70)
for idx, row in unique_events.head(10).iterrows():
    print(f"{idx+1}. {row['station']} - {row['date_only']} - {row['magnitude_class']} - {row['azimuth_class']}")

print(f"\n{'='*70}")
print("✅ Validation system ready!")
print(f"{'='*70}")

print(f"\n💡 To run full validation:")
print(f"   python validate_historical_earthquakes.py --max-events 5")
