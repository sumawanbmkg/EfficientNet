"""
Scan Medium Earthquake Dataset from local 'missing' directory
=============================================================
Complementing the remote scan with local raw data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import struct
import matplotlib.pyplot as plt
from scipy import signal
import logging
from tqdm import tqdm
import gzip

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Input & Output
REPO_FILE = 'earthquake_catalog_2018_2025_merged.csv'
STATION_FILE = 'mdata2/lokasi_stasiun.csv'
LOCAL_DATA_DIR = 'missing'
OUTPUT_DIR = 'dataset_medium_new' # Same output folder to consolidate
TARGET_COUNT = 500

# Baselines (Same as scanner)
BASELINES = {
    'GTO': {'H': 40000, 'Z': 30000},
    'GSI': {'H': 40000, 'Z': 30000},
    'ALR': {'H': 38000, 'Z': 32000},
    'AMB': {'H': 38000, 'Z': 32000},
    'SCN': {'H': 38000, 'Z': 32000},
    'MLB': {'H': 38000, 'Z': 32000},
    'SBG': {'H': 38000, 'Z': 32000},
    'YOG': {'H': 38000, 'Z': 32000},
    'TRT': {'H': 38000, 'Z': 32000},
    'LWA': {'H': 38000, 'Z': 32000},
    'KPY': {'H': 38000, 'Z': 32000},
    'JYP': {'H': 38000, 'Z': 32000},
    'TRD': {'H': 38000, 'Z': 32000},
    'KUP': {'H': 38000, 'Z': 32000},
    'DEFAULT': {'H': 38000, 'Z': 32000} 
}

class LocalMediumScanner:
    def __init__(self):
        self.stations_df = self.load_stations()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'spectrograms'), exist_ok=True)
        
    def load_stations(self):
        try:
            df = pd.read_csv(STATION_FILE, sep=';')
            df['Latitude'] = df['Latitude'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
            df['Longitude'] = df['Longitude'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
            df = df[df['Latitude'] != '']
            df['Latitude'] = df['Latitude'].astype(float)
            df['Longitude'] = df['Longitude'].astype(float)
            df['Kode Stasiun'] = df['Kode Stasiun'].str.strip()
            return df
        except:
            return pd.DataFrame()

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def find_nearest_station(self, eq_lat, eq_lon):
        if self.stations_df.empty: return None, 9999
        min_dist = 9999
        nearest = None
        for idx, row in self.stations_df.iterrows():
            dist = self.haversine(eq_lat, eq_lon, row['Latitude'], row['Longitude'])
            if dist < min_dist:
                min_dist = dist
                nearest = row['Kode Stasiun']
        if min_dist < 600:
            return nearest, min_dist
        return None, min_dist

    def generate_spectrogram(self, H_data, Z_data, output_path):
        if H_data is None or Z_data is None or len(H_data) < 100:
            return False
        try:
            valid_mask = (H_data > 0) & ~np.isnan(H_data) & ~np.isnan(Z_data)
            if np.sum(valid_mask) < 100: return False
            zh_ratio = np.abs(Z_data[valid_mask] / H_data[valid_mask])
            fs = 1 
            nperseg = min(256, len(zh_ratio) // 4)
            f, t, Sxx = signal.spectrogram(zh_ratio, fs=fs, nperseg=nperseg)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
            plt.axis('off')
            plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            return True
        except:
            return False

    def process_event(self, event):
        try:
            dt_obj = pd.to_datetime(event['datetime'])
        except:
            return None
            
        station, dist = self.find_nearest_station(event['Latitude'], event['Longitude'])
        if not station:
            return None
            
        target_dt = dt_obj - timedelta(hours=1)
        date_str = target_dt.strftime("%Y-%m-%d")
        yy = target_dt.year % 100
        mm = target_dt.month
        dd = target_dt.day
        hour = target_dt.hour
        
        # Local Path Formats
        filename_local = f"S{yy:02d}{mm:02d}{dd:02d}.{station}"
        
        # Check in 'missing'
        path_missing = os.path.join(LOCAL_DATA_DIR, station, filename_local)
        # Check in 'mdata2' (with potential .gz)
        path_mdata = os.path.join('mdata2', station, filename_local)
        path_mdata_gz = path_mdata + ".gz"
        
        local_path = None
        is_gz = False
        
        if os.path.exists(path_missing):
            local_path = path_missing
        elif os.path.exists(path_mdata):
            local_path = path_mdata
        elif os.path.exists(path_mdata_gz):
            local_path = path_mdata_gz
            is_gz = True
        
        if not local_path:
            return None

        save_filename = f"{station}_{target_dt.strftime('%Y%m%d')}_H{hour:02d}_M{event['Magnitude']:.1f}_local.png"
        save_path = os.path.join(OUTPUT_DIR, 'spectrograms', save_filename)
        
        if os.path.exists(save_path):
            return self._create_metadata(event, station, date_str, hour, save_filename)

        # Parse Local Binary
        try:
            if is_gz:
                with gzip.open(local_path, 'rb') as f:
                    binary_data = f.read()
            else:
                with open(local_path, 'rb') as f:
                    binary_data = f.read()
                
            baseline = BASELINES.get(station, BASELINES['DEFAULT'])
            header_size = 32
            record_size = 17
            data_start = binary_data[header_size:]
            
            start_idx = hour * 3600
            end_idx = (hour + 1) * 3600
            
            H_data = []
            Z_data = []
            num_records = len(data_start) // record_size
            
            if start_idx >= num_records: return None 
                
            for i in range(start_idx, min(end_idx, num_records)):
                offset = i * record_size
                record = data_start[offset:offset+record_size]
                h_dev = struct.unpack('<h', record[0:2])[0] * 0.1
                z_dev = struct.unpack('<h', record[4:6])[0] * 0.1
                H_data.append(baseline['H'] + h_dev)
                Z_data.append(baseline['Z'] + z_dev)
                
            if self.generate_spectrogram(np.array(H_data), np.array(Z_data), save_path):
                return self._create_metadata(event, station, date_str, hour, save_filename)
        except:
            pass
        return None

    def _create_metadata(self, event, station, date, hour, filename):
        return {
            'filename': filename,
            'filepath': f"spectrograms/{filename}",
            'station': station,
            'date': date,
            'hour': hour,
            'magnitude': event['Magnitude'],
            'magnitude_class': 'Medium',
            'azimuth': 0, 
            'azimuth_class': 'Unknown',
            'source': 'local'
        }

def main():
    logger.info("PHASE 2 - LOCAL MEDIUM DATASET SCANNER")
    
    # Load existing metadata to avoid duplication and see current count
    existing_meta_path = os.path.join(OUTPUT_DIR, 'metadata.csv')
    existing_results = []
    if os.path.exists(existing_meta_path):
        existing_df = pd.read_csv(existing_meta_path)
        existing_results = existing_df.to_dict('records')
        logger.info(f"Existing Medium samples: {len(existing_results)}")
    
    if len(existing_results) >= TARGET_COUNT:
        logger.info("Target already reached.")
        # return # User wants to scan local anyway to improve
    
    df = pd.read_csv(REPO_FILE)
    df['dt'] = pd.to_datetime(df['datetime'])
    df = df[(df['Magnitude'] >= 5.0) & (df['Magnitude'] < 6.0)]
    
    # Shuffle
    df = df.sample(frac=1, random_state=42)
    
    scanner = LocalMediumScanner()
    new_results = []
    
    total_added = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scanning Local Missing"):
        res = scanner.process_event(row)
        if res:
            # Check if filename already in existing_results
            if not any(e['filename'] == res['filename'] for e in existing_results):
                new_results.append(res)
                total_added += 1
                if (len(existing_results) + total_added) >= TARGET_COUNT:
                    pass # Keep scanning to get more quality local data
            
    if new_results:
        combined_results = existing_results + new_results
        pd.DataFrame(combined_results).to_csv(existing_meta_path, index=False)
        logger.info(f"SUCCESS. Added {total_added} local samples. Total: {len(combined_results)}")
    else:
        logger.warning("No new local data found.")

if __name__ == "__main__":
    main()
