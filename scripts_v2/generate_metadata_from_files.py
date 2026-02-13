"""
Quick Script: Generate metadata.csv from existing PNG files
Untuk dataset_moderate dan dataset_medium_new yang belum punya metadata
"""
import os
import pandas as pd
import re

def generate_metadata_from_files(folder_path, magnitude_class):
    """Parse filename untuk extract metadata"""
    spectrograms_path = os.path.join(folder_path, 'spectrograms')
    
    if not os.path.exists(spectrograms_path):
        print(f"Folder {spectrograms_path} tidak ditemukan")
        return
    
    files = [f for f in os.listdir(spectrograms_path) if f.endswith('.png')]
    
    metadata = []
    for filename in files:
        # Format: STATION_YYYYMMDD_HXX_M4.5.png
        # Contoh: SCN_20230104_H02_M4.5.png
        match = re.match(r'([A-Z]+)_(\d{8})_H(\d{2})_M([\d\.]+)\.png', filename)
        
        if match:
            station = match.group(1)
            date_str = match.group(2)
            hour = int(match.group(3))
            magnitude = float(match.group(4))
            
            # Format date
            date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            metadata.append({
                'filename': filename,
                'filepath': f"spectrograms/{filename}",
                'station': station,
                'date': date,
                'hour': hour,
                'magnitude': magnitude,
                'magnitude_class': magnitude_class,
                'azimuth': 0,
                'azimuth_class': 'Unknown'
            })
    
    if metadata:
        df = pd.DataFrame(metadata)
        output_path = os.path.join(folder_path, 'metadata.csv')
        df.to_csv(output_path, index=False)
        print(f"✓ Generated {output_path} with {len(metadata)} entries")
    else:
        print(f"✗ No valid files found in {spectrograms_path}")

# Generate untuk Moderate
generate_metadata_from_files('dataset_moderate', 'Moderate')

# Generate untuk Medium (jika ada)
generate_metadata_from_files('dataset_medium_new', 'Medium')

print("\nDone!")
