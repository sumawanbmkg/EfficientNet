"""
Test CNN format - 224x224, no axis, no text
"""
import sys
sys.path.insert(0, 'intial')

from geomagnetic_dataset_generator_ssh_v21 import GeomagneticDatasetGeneratorSSH_V21
import pandas as pd
from PIL import Image

# Create generator
generator = GeomagneticDatasetGeneratorSSH_V21(output_dir='test_cnn_output')

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
        print(f"Image size: {result['image_size']}")
        
        # Verify image
        img_path = f"test_cnn_output/spectrograms/{result['spectrogram_file']}"
        img = Image.open(img_path)
        
        print(f"\n[VERIFICATION]")
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        print(f"Image format: {img.format}")
        
        if img.size == (224, 224) and img.mode == 'RGB':
            print(f"\n[OK] Image format is CORRECT for CNN!")
            print(f"   - Size: 224x224 pixels")
            print(f"   - Mode: RGB")
            print(f"   - Ready for CNN training!")
        else:
            print(f"\n[WARNING] Image format needs adjustment")
            
    elif result == 'SKIPPED':
        print("\n[SKIPPED] Event already processed")
    else:
        print("\n[FAILED] Event processing failed")
