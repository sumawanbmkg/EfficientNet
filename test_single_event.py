"""
Test single event processing dengan binary reading yang sudah diperbaiki
"""
import sys
sys.path.insert(0, 'intial')

from geomagnetic_dataset_generator_ssh_v2 import GeomagneticDatasetGeneratorSSH_V2
import pandas as pd

# Create generator
generator = GeomagneticDatasetGeneratorSSH_V2(output_dir='test_output')

# Read event list
df = pd.read_excel('intial/event_list.xlsx')

# Test dengan 1 event: SCN 2018-01-17 Hour 19
test_event = df[(df['Stasiun'] == 'SCN') & 
                (pd.to_datetime(df['Tanggal']).dt.strftime('%Y-%m-%d') == '2018-01-17') &
                (df['Jam'] == 19)].iloc[0]

print(f"Testing event: {test_event['Stasiun']} - {test_event['Tanggal']} Hour {test_event['Jam']}")

# Process
from geomagnetic_fetcher import GeomagneticDataFetcher

with GeomagneticDataFetcher() as fetcher:
    result = generator.process_event(fetcher, test_event)
    
    if result and result != 'SKIPPED':
        print("\n[SUCCESS] Event processed successfully!")
        print(f"Spectrogram saved: {result['spectrogram_file']}")
        print(f"H mean: {result['h_mean']:.2f} nT")
        print(f"D mean: {result['d_mean']:.2f} nT")
        print(f"Z mean: {result['z_mean']:.2f} nT")
        print(f"Samples: {result['samples']}")
        print("\nPlease check the spectrogram image in test_output/spectrograms/")
    elif result == 'SKIPPED':
        print("\n[SKIPPED] Event already processed")
    else:
        print("\n[FAILED] Event processing failed")
