import pandas as pd

df = pd.read_csv('dataset_unified/metadata/unified_metadata.csv')

# Separate precursor and normal
precursor = df[df['magnitude_class'] != 'Normal']
normal = df[df['magnitude_class'] == 'Normal']

print("="*70)
print("DATASET REALITY CHECK")
print("="*70)

print(f"\nTotal samples: {len(df)}")
print(f"├── Precursor samples: {len(precursor)}")
print(f"└── Normal samples: {len(normal)}")

# Count unique earthquake events
events = precursor.groupby(['date', 'magnitude']).size().reset_index(name='samples')
print(f"\n{'='*70}")
print(f"UNIQUE EARTHQUAKE EVENTS")
print(f"{'='*70}")
print(f"Total unique events: {len(events)}")

# Breakdown by magnitude class
print(f"\nBreakdown by magnitude class:")
for mag_class in ['Moderate', 'Medium', 'Large']:
    subset = precursor[precursor['magnitude_class'] == mag_class]
    unique_events = subset.groupby(['date', 'magnitude']).size()
    print(f"  {mag_class:10s}: {len(unique_events):3d} events → {len(subset):4d} samples")

# Samples per event
print(f"\nSamples per event statistics:")
print(f"  Mean: {events['samples'].mean():.1f}")
print(f"  Median: {events['samples'].median():.1f}")
print(f"  Min: {events['samples'].min()}")
print(f"  Max: {events['samples'].max()}")

# Show how samples are created
print(f"\n{'='*70}")
print(f"HOW 1,972 SAMPLES ARE CREATED")
print(f"{'='*70}")

# Count stations and hours
unique_stations = precursor['station'].nunique()
unique_hours = precursor['hour'].nunique() if 'hour' in precursor.columns else 'N/A'

print(f"\nFor each earthquake event:")
print(f"  × {unique_stations} stations (different locations)")
print(f"  × Multiple hours (6-hour window)")
print(f"  = Multiple samples per event")

# Example
example_event = events.iloc[0]
example_samples = precursor[(precursor['date'] == example_event['date']) & 
                            (precursor['magnitude'] == example_event['magnitude'])]
print(f"\nExample: Event on {example_event['date']}, M{example_event['magnitude']}")
print(f"  Stations: {example_samples['station'].unique()}")
print(f"  Hours: {sorted(example_samples['hour'].unique()) if 'hour' in example_samples.columns else 'N/A'}")
print(f"  Total samples: {len(example_samples)}")

print(f"\n{'='*70}")
print(f"CRITICAL FINDING")
print(f"{'='*70}")
print(f"\n⚠️  ACTUAL EARTHQUAKE EVENTS: {len(events)}")
print(f"⚠️  TOTAL SAMPLES (after windowing): {len(precursor)}")
print(f"⚠️  MULTIPLICATION FACTOR: {len(precursor) / len(events):.1f}x")
print(f"\nThis means:")
print(f"  - Each earthquake creates ~{len(precursor) / len(events):.0f} samples")
print(f"  - Through: Multiple stations × Multiple time windows")
print(f"  - This is WINDOWING, not independent events!")

# Save results
results = {
    'total_samples': len(df),
    'precursor_samples': len(precursor),
    'normal_samples': len(normal),
    'unique_earthquake_events': len(events),
    'multiplication_factor': len(precursor) / len(events),
    'events_by_magnitude': {
        'Moderate': len(precursor[precursor['magnitude_class'] == 'Moderate'].groupby(['date', 'magnitude'])),
        'Medium': len(precursor[precursor['magnitude_class'] == 'Medium'].groupby(['date', 'magnitude'])),
        'Large': len(precursor[precursor['magnitude_class'] == 'Large'].groupby(['date', 'magnitude']))
    }
}

import json
with open('dataset_reality_check.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: dataset_reality_check.json")
