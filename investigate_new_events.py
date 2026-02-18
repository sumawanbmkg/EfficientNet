"""
Investigasi data event gempa baru dari new_event.csv
"""

import pandas as pd
import numpy as np
from collections import Counter

# Load data
df = pd.read_csv('new_event.csv')

print("=" * 70)
print("INVESTIGASI DATA EVENT GEMPA BARU")
print("=" * 70)

# Basic info
print(f"\nTotal records: {len(df)}")
print(f"\nKolom: {list(df.columns)}")

# Unique values
print("\n" + "=" * 70)
print("RINGKASAN DATA")
print("=" * 70)

print(f"\nStasiun unik: {df['Stasiun'].unique()}")
print(f"Tanggal unik: {sorted(df['Tanggal'].unique())}")
print(f"Magnitude unik: {sorted(df['Mag'].unique())}")
print(f"Azimuth unik: {sorted(df['Azm'].unique())}")

# Group by event (unique combination of date + azimuth + magnitude)
print("\n" + "=" * 70)
print("ANALISIS EVENT GEMPA")
print("=" * 70)

events = df.groupby(['Tanggal', 'Azm', 'Mag']).agg({
    'Jam': ['count', 'min', 'max', list]
}).reset_index()
events.columns = ['Tanggal', 'Azimuth', 'Magnitude', 'Jam_Count', 'Jam_Min', 'Jam_Max', 'Jam_List']

print(f"\nJumlah event unik: {len(events)}")
print("\nDetail setiap event:")
for _, row in events.iterrows():
    # Convert azimuth to direction
    azm = row['Azimuth']
    if 337.5 <= azm or azm < 22.5:
        direction = 'N'
    elif 22.5 <= azm < 67.5:
        direction = 'NE'
    elif 67.5 <= azm < 112.5:
        direction = 'E'
    elif 112.5 <= azm < 157.5:
        direction = 'SE'
    elif 157.5 <= azm < 202.5:
        direction = 'S'
    elif 202.5 <= azm < 247.5:
        direction = 'SW'
    elif 247.5 <= azm < 292.5:
        direction = 'W'
    else:
        direction = 'NW'
    
    # Magnitude class
    mag = row['Magnitude']
    if mag >= 6.0:
        mag_class = 'Large'
    elif mag >= 5.0:
        mag_class = 'Medium'
    else:
        mag_class = 'Moderate'
    
    print(f"\n  Event: {row['Tanggal']}")
    print(f"    Magnitude: {mag} ({mag_class})")
    print(f"    Azimuth: {azm}° ({direction})")
    print(f"    Jam tersedia: {row['Jam_Count']} jam (H{int(row['Jam_Min'])}-H{int(row['Jam_Max'])})")

# Summary by magnitude class
print("\n" + "=" * 70)
print("DISTRIBUSI MAGNITUDE CLASS")
print("=" * 70)

def get_mag_class(mag):
    if mag >= 6.0:
        return 'Large'
    elif mag >= 5.0:
        return 'Medium'
    else:
        return 'Moderate'

df['magnitude_class'] = df['Mag'].apply(get_mag_class)
mag_dist = Counter(df['magnitude_class'])
print(f"\nDistribusi per jam:")
for cls, count in sorted(mag_dist.items()):
    print(f"  {cls}: {count} jam")

# Summary by azimuth class
print("\n" + "=" * 70)
print("DISTRIBUSI AZIMUTH CLASS")
print("=" * 70)

def get_azi_class(azm):
    if 337.5 <= azm or azm < 22.5:
        return 'N'
    elif 22.5 <= azm < 67.5:
        return 'NE'
    elif 67.5 <= azm < 112.5:
        return 'E'
    elif 112.5 <= azm < 157.5:
        return 'SE'
    elif 157.5 <= azm < 202.5:
        return 'S'
    elif 202.5 <= azm < 247.5:
        return 'SW'
    elif 247.5 <= azm < 292.5:
        return 'W'
    else:
        return 'NW'

df['azimuth_class'] = df['Azm'].apply(get_azi_class)
azi_dist = Counter(df['azimuth_class'])
print(f"\nDistribusi per jam:")
for cls, count in sorted(azi_dist.items()):
    print(f"  {cls}: {count} jam")

# Check existing dataset
print("\n" + "=" * 70)
print("PERBANDINGAN DENGAN DATASET EXISTING")
print("=" * 70)

try:
    existing = pd.read_csv('dataset_unified/metadata/train_split.csv')
    existing_val = pd.read_csv('dataset_unified/metadata/val_split.csv')
    existing_all = pd.concat([existing, existing_val])
    
    print(f"\nDataset existing:")
    print(f"  Total samples: {len(existing_all)}")
    
    existing_mag = Counter(existing_all['magnitude_class'])
    print(f"\n  Magnitude distribution:")
    for cls, count in sorted(existing_mag.items()):
        print(f"    {cls}: {count}")
    
    # Check if BYN station exists
    if 'station' in existing_all.columns:
        stations = existing_all['station'].unique()
        print(f"\n  Stasiun existing: {list(stations)}")
        if 'BYN' in stations:
            print("  [OK] BYN sudah ada di dataset")
        else:
            print("  [NEW] BYN adalah stasiun BARU!")
    
except Exception as e:
    print(f"Error loading existing dataset: {e}")

# Potential issues
print("\n" + "=" * 70)
print("POTENSI MASALAH & CATATAN")
print("=" * 70)

print("""
1. STASIUN BARU: BYN (Banyuwangi?) - perlu data geomagnetik dari stasiun ini
   
2. DATA HOURLY: Setiap event memiliki multiple jam (precursor window)
   - Ini bagus untuk training karena menambah variasi temporal
   
3. MAGNITUDE:
   - M5.2 (Medium class) - 204 jam dari 2 event berbeda azimuth
   - M4.9 (Moderate class) - 252 jam dari 2 event berbeda azimuth
   
4. AZIMUTH BARU:
   - 38° (NE) - baru
   - 340° (NW) - sudah ada
   - 157° (SE) - sudah ada  
   - 46° (NE) - baru

5. TANGGAL: September 2018 (1-6 Sept)
   - Perlu cek apakah data geomagnetik tersedia untuk periode ini

6. CATATAN PENTING:
   - Data ini akan menambah 456 samples baru
   - Terutama membantu class Moderate (saat ini hanya 16 samples)
   - Akan menambah variasi stasiun (BYN baru)
""")

print("\n" + "=" * 70)
print("REKOMENDASI")
print("=" * 70)
print("""
1. Cek ketersediaan data geomagnetik BYN untuk Sept 2018
2. Generate spectrogram untuk setiap jam
3. Tambahkan ke dataset_unified
4. Re-run SMOTE dengan data baru
5. Re-train model
""")
