#!/usr/bin/env python3
"""
Verify Missing Data - Compare event_list.xlsx with generated dataset
Find which station-date-hour combinations are missing

Date: 6 February 2026
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime

print("=" * 70)
print("VERIFIKASI DATA YANG BELUM DI-GENERATE")
print("=" * 70)

# 1. Load event list
print("\n[1] Loading event_list.xlsx...")
event_df = pd.read_excel('intial/event_list.xlsx')
print(f"    Total events in list: {len(event_df)}")

# Normalize station names
event_df['Stasiun'] = event_df['Stasiun'].str.upper().str.strip()

# Create unique key: Station_Date_Hour
event_df['date_str'] = pd.to_datetime(event_df['Tanggal']).dt.strftime('%Y%m%d')
event_df['key'] = event_df['Stasiun'] + '_' + event_df['date_str'] + '_H' + event_df['Jam'].astype(str).str.zfill(2)

print(f"    Unique station-date-hour combinations: {event_df['key'].nunique()}")

# 2. Scan existing spectrograms in dataset_unified
print("\n[2] Scanning existing spectrograms in dataset_unified...")

existing_keys = set()
dataset_path = Path('dataset_unified/spectrograms')

if dataset_path.exists():
    for mag_folder in dataset_path.iterdir():
        if mag_folder.is_dir():
            for azi_folder in mag_folder.iterdir():
                if azi_folder.is_dir():
                    for png_file in azi_folder.glob('*.png'):
                        # Parse filename: SCN_20180117_H19_3comp_spec.png
                        name = png_file.stem
                        parts = name.split('_')
                        if len(parts) >= 3:
                            station = parts[0]
                            date = parts[1]
                            hour = parts[2]
                            key = f"{station}_{date}_{hour}"
                            existing_keys.add(key)

print(f"    Found {len(existing_keys)} spectrograms in dataset_unified")

# 3. Also check dataset_spectrogram_ssh_v22
print("\n[3] Scanning dataset_spectrogram_ssh_v22...")
ssh_path = Path('dataset_spectrogram_ssh_v22/spectrograms')

if ssh_path.exists():
    for mag_folder in ssh_path.iterdir():
        if mag_folder.is_dir():
            for azi_folder in mag_folder.iterdir():
                if azi_folder.is_dir():
                    for png_file in azi_folder.glob('*.png'):
                        name = png_file.stem
                        parts = name.split('_')
                        if len(parts) >= 3:
                            station = parts[0]
                            date = parts[1]
                            hour = parts[2]
                            key = f"{station}_{date}_{hour}"
                            existing_keys.add(key)

print(f"    Total unique spectrograms (combined): {len(existing_keys)}")

# 4. Find missing data
print("\n[4] Finding missing data...")

required_keys = set(event_df['key'].unique())
missing_keys = required_keys - existing_keys
found_keys = required_keys & existing_keys

print(f"    Required: {len(required_keys)}")
print(f"    Found: {len(found_keys)}")
print(f"    Missing: {len(missing_keys)}")
print(f"    Coverage: {len(found_keys)/len(required_keys)*100:.1f}%")

# 5. Create missing data CSV
print("\n[5] Creating missing_data.csv...")

missing_records = []
# Filter out NaN keys
missing_keys = {k for k in missing_keys if isinstance(k, str)}
for key in sorted(missing_keys):
    # Parse key: STATION_YYYYMMDD_HXX
    parts = key.split('_')
    station = parts[0]
    date_str = parts[1]
    hour = parts[2].replace('H', '')
    
    # Get original event info
    event_info = event_df[event_df['key'] == key].iloc[0] if key in event_df['key'].values else None
    
    missing_records.append({
        'station': station,
        'date': f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
        'hour': int(hour),
        'date_raw': date_str,
        'magnitude': event_info['Mag'] if event_info is not None else None,
        'azimuth': event_info['Azm'] if event_info is not None else None,
        'key': key
    })

missing_df = pd.DataFrame(missing_records)
missing_df = missing_df.sort_values(['station', 'date', 'hour'])
missing_df.to_csv('missing_data_updated.csv', index=False)

print(f"    Saved to: missing_data_updated.csv")

# 6. Summary by station
print("\n[6] Summary by Station:")
print("-" * 50)

station_summary = missing_df.groupby('station').agg({
    'date': 'count',
    'magnitude': 'mean'
}).rename(columns={'date': 'missing_count', 'magnitude': 'avg_mag'})

station_summary = station_summary.sort_values('missing_count', ascending=False)
print(station_summary.to_string())

# 7. Summary by year
print("\n[7] Summary by Year:")
print("-" * 50)

missing_df['year'] = missing_df['date'].str[:4]
year_summary = missing_df.groupby('year').size()
print(year_summary.to_string())

# 8. Print sample of missing data
print("\n[8] Sample Missing Data (first 20):")
print("-" * 70)
print(missing_df[['station', 'date', 'hour', 'magnitude', 'azimuth']].head(20).to_string(index=False))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total events in event_list.xlsx: {len(event_df)}")
print(f"Unique station-date-hour required: {len(required_keys)}")
print(f"Already generated: {len(found_keys)}")
print(f"Missing (need raw data): {len(missing_keys)}")
print(f"Coverage: {len(found_keys)/len(required_keys)*100:.1f}%")
print(f"\nOutput file: missing_data.csv")
print("=" * 70)
