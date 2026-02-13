"""
Script Phase 2 - Step 1: Merge and Validate Datasets
Menggabungkan data dari 3 sumber utama:
1. dataset_unified (Data lama/training set v1)
2. dataset_new_events (Hasil scan SSH terbaru)
3. dataset_missing_filled (Hasil recovery dari local storage)

Tujuan:
- Menciptakan Single Source of Truth: dataset_consolidation/
- Quality Control: Cek file corrupt/size 0
- Deduplikasi: Hapus event ganda berdasarkan (Station + Date + Hour)
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import logging

# Konfigurasi
SOURCE_DIRS = {
    'normal_new': 'dataset_normal_new',  # Prioritas data tenang baru (2023-2025)
    'moderate': 'dataset_moderate',
    'medium_new': 'dataset_medium_new',
    'new_events': 'dataset_new_events',
    'missing_filled': 'dataset_missing_filled',
    'unified': 'dataset_unified'         # Legacy data sebagai fallback
}

OUTPUT_DIR = 'dataset_consolidation'
LOG_FILE = 'scripts_v2/logs/1_merge_datasets.log'

# Setup Logging
os.makedirs('scripts_v2/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_image(image_path):
    """Cek apakah file gambar valid dan tidak corrupt"""
    try:
        if not os.path.exists(image_path):
            return False
        
        # Cek ukuran file
        if os.path.getsize(image_path) < 1024:  # Kurang dari 1KB probably corrupt
            return False
            
        # Cek apakah bisa dibuka sebagai gambar
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.warning(f"Corrupt image found: {image_path} ({e})")
        return False

def reclassify_magnitude(mag):
    """Klasifikasi ulang Magnitude berdasarkan standar Production V2"""
    try:
        m = float(mag)
        if m >= 6.0: return 'Large'
        elif m >= 5.0: return 'Medium'
        elif m >= 4.0: return 'Moderate'
        elif m > 0: return 'Normal' # Small -> Normal
        return 'Normal'
    except:
        return 'Normal'

def standardize_metadata(df, source_name):
    """Standarisasi kolom metadata dari berbagai sumber"""
    # Pastikan kolom essential ada
    required_cols = ['station', 'date', 'hour', 'azimuth', 'magnitude', 
                     'azimuth_class', 'magnitude_class']
    
    # Mapping nama kolom jika beda (contoh: 'stasiun' -> 'station')
    col_mapping = {
        'stasiun': 'station',
        'Stasiun': 'station',
        'Tanggal': 'date',
        'Jam': 'hour',
        'Mag': 'magnitude',
        'Azm': 'azimuth',
        'magnitudo': 'magnitude'
    }
    df = df.rename(columns=col_mapping)
    
    # Isi missing columns dengan default jika perlu
    for col in required_cols:
        if col not in df.columns:
            if col == 'azimuth_class':
                df[col] = 'Unknown' # Nanti bisa direcompute
            elif col == 'magnitude_class':
                df[col] = 'Normal'
            else:
                df[col] = np.nan

    
    # Mapping untuk berbagai variasi nama kolom dari dataset berbeda
    # 1. Path Gambar
    if 'unified_path' in df.columns:
        df['original_relative_path'] = df['unified_path']
    elif 'filepath' in df.columns: # Format dataset_new_events
        df['original_relative_path'] = df['filepath']
    elif 'filename' in df.columns: # Format missing_filled
        df['original_relative_path'] = df['filename']
        
    # 2. Magnitude Class
    if 'magnitude_class' not in df.columns:
        if 'mag_class' in df.columns:
             df['magnitude_class'] = df['mag_class']
        else:
             from scipy import signal # Dummy import to signify change
             df['magnitude_class'] = 'Normal' # Default
             
    # 3. Azimuth Class
    if 'azimuth_class' not in df.columns:
        if 'azm_class' in df.columns:
             df['azimuth_class'] = df['azm_class']
        elif 'azimuth_category' in df.columns:
             df['azimuth_class'] = df['azimuth_category']
        else: 
             df['azimuth_class'] = 'Normal'

    # RE-CLASSIFY MAGNITUDE (CRITICAL FIX)
    # Jangan percaya label bawaan, hitung ulang dari nilai numerik
    if 'magnitude' in df.columns:
        df['magnitude_class'] = df['magnitude'].apply(reclassify_magnitude)

    # Filter kolom wajib
    required_cols = ['station', 'date', 'magnitude', 'azimuth', 'original_relative_path', 'magnitude_class', 'azimuth_class']
    
    # Pastikan 'hour' ada dan integer
    if 'hour' not in df.columns:
        df['hour'] = 0
    else:
        # Bersihkan jam (kadang string 'H01')
        def clean_hour(h):
            try: return int(str(h).replace('H',''))
            except: return 0
        df['hour'] = df['hour'].apply(clean_hour)
        
    # Buat ID unik jika belum ada
        
    # Buat ID unik jika belum ada
    if 'id' not in df.columns:
        df['id'] = df.apply(lambda row: f"{row['station']}_{str(row['date']).replace('-','')}_{source_name}_{np.random.randint(1000,9999)}", axis=1)

    # Return hanya kolom standar + id + source
    final_cols = ['id', 'source_dataset'] + required_cols + ['hour']
    # Pastikan semua ada
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    return df[final_cols].copy()

def main():
    logger.info("="*50)
    logger.info("PHASE 2 - STEP 1: MERGE DATASETS")
    logger.info("="*50)
    
    # 1. Persiapan Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'spectrograms'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'metadata'), exist_ok=True)
    
    all_metadata = []
    stats = {'total_processed': 0, 'valid': 0, 'corrupt': 0, 'duplicate': 0}
    
    # Set unique identifier untuk deduplikasi
    processed_ids = set() # Format: STATION_YYYYMMDD_HH

    # 2. Iterasi Setiap Sumber Data
    for name, path in SOURCE_DIRS.items():
        logger.info(f"Processing source: {name} ({path})")
        
        metadata_path = None
        # Cari file metadata (csv)
        potential_meta = [
            os.path.join(path, 'metadata', 'unified_metadata.csv'),
            os.path.join(path, 'metadata.csv'),
            os.path.join(path, 'metadata', 'processed_events.csv')
        ]
        
        for p in potential_meta:
            if os.path.exists(p):
                metadata_path = p
                break
        
        if not metadata_path:
            logger.warning(f"No metadata found for {name}, skipping...")
            continue
            
        df = pd.read_csv(metadata_path)
        logger.info(f"  Loaded metadata: {len(df)} rows")
        
        # Standardize
        df = standardize_metadata(df, name)
        
        # Copy file & update metadata
        valid_rows = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Merging {name}"):
            stats['total_processed'] += 1
            
            # Construct Unique ID
            date_clean = str(row['date']).replace('-', '')
            try:
                hour_clean = int(row['hour'])
            except:
                hour_clean = 0
            event_id = f"{row['station']}_{date_clean}_{hour_clean:02d}"
            
            # Cek Deduplikasi (Prioritas: Unified > New > Missing)
            # Masalah: Data Normal lama mungkin punya Hour=0 semua, jadi dianggap duplikat
            # Solusi: Jika Magnitude < 4.0 (Normal), tambahkan suffix unik agar tidak kena deduplikasi
            #         Kecuali jika benar-benar duplikat file path
            
            is_normal = False
            try:
                mag_val = float(row['magnitude'])
                if mag_val < 4.0: is_normal = True
            except:
                is_normal = True
                
            if event_id in processed_ids:
                if is_normal:
                    # Untuk Normal, kita izinkan multiple entries per hari (asal beda jam/file)
                    # Tambahkan suffix random/counter
                    event_id = f"{event_id}_{np.random.randint(1000,9999)}"
                else:
                    # Untuk Prekursor (Event Gempa), kita strict deduplikasi
                    # Data baru (New Scan) mungkin lebih bagus dari Unified lama
                    # Tapi karena loop source kita berurutan (Unified dulu), maka Unified menang.
                    # Jika kita ingin New Scan menang, urutan SOURCE_DIRS harus diubah atau logic di sini.
                    # Asumsi sekarang: Unified adalah 'Gold Standard' lama, jadi keep.
                    stats['duplicate'] += 1
                    continue
            
            # Cari file gambar asli
            # Logika pencarian file gambar agak rumit karena struktur folder beda-beda
            # Kita coba cari rekursif atau berdasarkan pola nama file
            
            # Coba construct nama file standar dulu
            # Format umum: STATION_YYYYMMDD_Hxx_*.png
            # Atau cari berdasarkan 'original_relative_path'
            
            src_image_path = None
            
            # Strategi 1: Pakai path dari metadata jika valid
            # Strategi 1: Pakai path dari metadata jika valid
            if pd.notna(row['original_relative_path']):
                # Cek apakah relative path itu full path dari root dataset
                base_name = os.path.basename(str(row['original_relative_path']))
                cand1 = os.path.join(path, str(row['original_relative_path'])) # unified case: spectrograms/xyz.png
                cand2 = os.path.join(path, 'spectrograms', base_name) # consolidated style
                cand3 = os.path.join(path, 'Large', base_name) # Folder structure baru (Large)
                cand4 = os.path.join(path, 'Medium', base_name) # Folder structure baru (Medium)
                # Cek folder berjenjang lainnya dari new_events
                
                # Special handle untuk new_events yang punya struktur folder aneh (Medium\SE\...)
                # Kita coba ambil pure filename dan cari rekursif
                
                if os.path.exists(cand1): src_image_path = cand1
                elif os.path.exists(cand2): src_image_path = cand2
                elif os.path.exists(cand3): src_image_path = cand3
                elif os.path.exists(cand4): src_image_path = cand4
            
            # Strategi 2: Cari brute force di folder spectrograms berdasarkan nama file ID
            if not src_image_path:
                # Pola pencarian: STATION_YYYYMMDD_HH
                pattern = f"{row['station']}_{date_clean}_H{hour_clean:02d}"
                # Search
                search_dir = os.path.join(path, 'spectrograms')
                if os.path.exists(search_dir):
                    for root, dirs, files in os.walk(search_dir):
                        for f in files:
                            if f.startswith(pattern) and f.endswith('.png'):
                                src_image_path = os.path.join(root, f)
                                break
                        if src_image_path: break
            
            # Jika gambar tidak ketemu atau corrupt
            if not src_image_path or not validate_image(src_image_path):
                stats['corrupt'] += 1
                continue
                
            # Copy ke Consolidation Folder
            # Standardize filename: STATION_YYYYMMDD_HH_MAG_AZM.png
            dest_filename = f"{event_id}_M{row['magnitude']}_AZ{row['azimuth']}.png"
            dest_path = os.path.join(OUTPUT_DIR, 'spectrograms', dest_filename)
            
            try:
                shutil.copy2(src_image_path, dest_path)
                
                # Update Metadata
                row_dict = row.to_dict()
                row_dict['consolidation_path'] = f"spectrograms/{dest_filename}"
                row_dict['event_id'] = event_id
                valid_rows.append(row_dict)
                
                processed_ids.add(event_id)
                stats['valid'] += 1
                
            except Exception as e:
                logger.error(f"Failed to copy {src_image_path}: {e}")
                stats['corrupt'] += 1

        all_metadata.extend(valid_rows)

    # 3. Save Final Metadata
    final_df = pd.DataFrame(all_metadata)
    final_meta_path = os.path.join(OUTPUT_DIR, 'metadata.csv')
    final_df.to_csv(final_meta_path, index=False)
    
    logger.info("="*50)
    logger.info("MERGE COMPLETE")
    logger.info(f"Total Processed: {stats['total_processed']}")
    logger.info(f"Valid Validated: {len(final_df)}")
    logger.info(f"Corrupt/Missing: {stats['corrupt']}")
    logger.info(f"Duplicates:      {stats['duplicate']}")
    logger.info(f"Output: {OUTPUT_DIR}")
    
    # Print Class Distribution
    if not final_df.empty:
        logger.info("\nClass Distribution (Magnitude):")
        dist = final_df['magnitude_class'].value_counts()
        for idx, val in dist.items():
            logger.info(f"  {idx}: {val}")

if __name__ == "__main__":
    main()
