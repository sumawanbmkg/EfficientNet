"""
Generate Moderate Earthquake Dataset (M4.5 - M4.9) - FIXED VERSION
==================================================================
Script khusus untuk menscan data gempa Moderate (M4.x) dari repositor_gempa_bumi 2018-2025.csv.
Menggunakan logika parsing binary dan path yang sudah terbukti berhasil.

Fitur:
1. Load Koordinat Akurat dari mdata2/lokasi_stasiun.csv
2. Parsing Binary SData yang benar (extension .[STATION_CODE])
3. Matplotlib Spectrogram Generation (bukan numpy raw)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import paramiko
from io import BytesIO
import struct
import matplotlib.pyplot as plt
from scipy import signal
import logging
from tqdm import tqdm

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Konfigurasi SSH (Ditarik dari ENV untuk Keamanan GitHub)
SSH_CONFIG = {
    'hostname': os.getenv('SSH_HOSTNAME', '202.90.198.224'),
    'port': int(os.getenv('SSH_PORT', 4343)),
    'username': os.getenv('SSH_USERNAME', 'precursor'),
    'password': os.getenv('SSH_PASSWORD', 'otomatismon')
}

# Input & Output
REPO_FILE = 'earthquake_catalog_2018_2025_merged.csv'
STATION_FILE = 'mdata2/lokasi_stasiun.csv'
STATION_FILE = 'mdata2/lokasi_stasiun.csv'
OUTPUT_DIR = 'dataset_moderate'
TARGET_COUNT = 500

# Baselines (Penting untuk decoding binary)
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
    # Default fallback
    'DEFAULT': {'H': 38000, 'Z': 32000} 
}

class ModerateScanner:
    def __init__(self):
        self.ssh_client = None
        self.sftp_client = None
        self.stations_df = self.load_stations()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'spectrograms'), exist_ok=True)
        
    def load_stations(self):
        try:
            df = pd.read_csv(STATION_FILE, sep=';')
            # Clean data (remove spaces, parse float)
            df['Latitude'] = df['Latitude'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
            df['Longitude'] = df['Longitude'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
            df = df[df['Latitude'] != ''] # Remove empty
            df['Latitude'] = df['Latitude'].astype(float)
            df['Longitude'] = df['Longitude'].astype(float)
            df['Kode Stasiun'] = df['Kode Stasiun'].str.strip()
            logger.info(f"Loaded {len(df)} stations coordinates.")
            return df
        except Exception as e:
            logger.error(f"Failed to load stations: {e}")
            return pd.DataFrame()

    def connect(self):
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(**SSH_CONFIG, timeout=20)
            self.sftp_client = self.ssh_client.open_sftp()
            logger.info("SSH Connected.")
            return True
        except Exception as e:
            logger.error(f"SSH Connect Error: {e}")
            return False
            
    def close(self):
        if self.sftp_client: self.sftp_client.close()
        if self.ssh_client: self.ssh_client.close()
        
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
                
        # Threshold: 500 km untuk Moderate
        if min_dist < 500:
            return nearest, min_dist
        return None, min_dist

    def generate_spectrogram(self, H_data, Z_data, output_path, title=""):
        if H_data is None or Z_data is None or len(H_data) < 100:
            return False
        
        try:
            # Calculate Ratio (mirip generate_dataset_from_scan)
            valid_mask = (H_data > 0) & ~np.isnan(H_data) & ~np.isnan(Z_data)
            if np.sum(valid_mask) < 100: return False
            
            # Simple Ratio or Raw Data?
            # Original script uses Z/H Ratio Spectrogram? Let's check logic.
            # "zh_ratio = np.abs(Z_data / H_data)" -> Spectrogram of Ratio.
            # But wait, trainer uses RGB (H, D, Z).
            # Let's stick to generating 3-channel Spectrogram if possible, or 
            # if the original used ratio-only spec, we follow that.
            # Original code: `zh_ratio = np.abs(Z_data[valid_mask] / H_data[valid_mask])`
            # `signal.spectrogram(zh_ratio ...)`
            # AND `im = ax.pcolormesh(..., cmap='jet')` -> This creates colored single channel image.
            # This matches what we have in dataset_unified (png images).
            
            zh_ratio = np.abs(Z_data[valid_mask] / H_data[valid_mask])
            fs = 1 
            nperseg = min(256, len(zh_ratio) // 4)
            f, t, Sxx = signal.spectrogram(zh_ratio, fs=fs, nperseg=nperseg)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
            plt.axis('off') # Clean image for training
            
            plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            return True
            
        except Exception as e:
            return False

    def process_event(self, event):
        try:
            dt_obj = pd.to_datetime(event['datetime'])
        except:
            return None
            
        station, dist = self.find_nearest_station(event['Latitude'], event['Longitude'])
        if not station:
            return None
            
        # Target: 1 Jam Sebelum Gempa
        target_dt = dt_obj - timedelta(hours=1)
        date_str = target_dt.strftime("%Y-%m-%d")
        yy = target_dt.year % 100
        mm = target_dt.month
        dd = target_dt.day
        hour = target_dt.hour
        
        # Path Format Fix: /DATA/{station}/SData/{yy}{mm}/S{yy}{mm}{dd}.{station}
        foldername = f"{yy:02d}{mm:02d}"
        filename_remote = f"S{yy:02d}{mm:02d}{dd:02d}.{station}"
        remote_path = f"/home/precursor/SEISMO/DATA/{station}/SData/{foldername}/{filename_remote}"
        
        # Output Filename
        save_filename = f"{station}_{target_dt.strftime('%Y%m%d')}_H{hour:02d}_M{event['Magnitude']:.1f}.png"
        save_path = os.path.join(OUTPUT_DIR, 'spectrograms', save_filename)
        
        if os.path.exists(save_path):
            return self._create_metadata(event, station, date_str, hour, save_filename)

        # Fetch & Parse
        try:
            with BytesIO() as buffer:
                self.sftp_client.getfo(remote_path, buffer)
                buffer.seek(0)
                binary_data = buffer.read()
                
            baseline = BASELINES.get(station, BASELINES['DEFAULT'])
            header_size = 32
            record_size = 17
            data_start = binary_data[header_size:]
            
            # Extract specific hour
            start_idx = hour * 3600
            end_idx = (hour + 1) * 3600
            
            H_data = []
            Z_data = []
            
            num_records = len(data_start) // record_size
            
            if start_idx >= num_records:
                return None # Data jam tersebut tidak ada
                
            for i in range(start_idx, min(end_idx, num_records)):
                offset = i * record_size
                record = data_start[offset:offset+record_size]
                
                h_dev = struct.unpack('<h', record[0:2])[0] * 0.1
                z_dev = struct.unpack('<h', record[4:6])[0] * 0.1
                
                H_data.append(baseline['H'] + h_dev)
                Z_data.append(baseline['Z'] + z_dev)
                
            H_arr = np.array(H_data)
            Z_arr = np.array(Z_data)
            
            if self.generate_spectrogram(H_arr, Z_arr, save_path):
                return self._create_metadata(event, station, date_str, hour, save_filename)
            
        except Exception as e:
            # logger.debug(f"Fetch failed {remote_path}: {e}")
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
            'magnitude_class': 'Moderate', # FORCE LABEL
            'azimuth': 0, 
            'azimuth_class': 'Unknown'
        }

def main():
    logger.info("PHASE 2 - MODERATE DATASET GENERATOR (FIXED)")
    
    df = pd.read_csv(REPO_FILE)
    df['dt'] = pd.to_datetime(df['datetime'])
    
    # Priority 2023-2025 first
    df_new = df[df['dt'].dt.year >= 2023]
    df_old = df[df['dt'].dt.year < 2023]
    
    # Target M4.5 - M4.9
    df_new = df_new[(df_new['Magnitude'] >= 4.5) & (df_new['Magnitude'] < 5.0)]
    df_old = df_old[(df_old['Magnitude'] >= 4.5) & (df_old['Magnitude'] < 5.0)]
    
    df = pd.concat([df_new, df_old])
    
    logger.info(f"Candidates (M4.5-4.9, >=2023): {len(df)}")
    
    # Shuffle candidates to get a good spread
    df = df.sample(frac=1, random_state=42)
    
    scanner = ModerateScanner()
    if not scanner.connect(): return
        
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scanning"):
        res = scanner.process_event(row)
        if res:
            results.append(res)
            if len(results) >= TARGET_COUNT:
                break
            
    scanner.close()
    
    if results:
        meta_path = os.path.join(OUTPUT_DIR, 'metadata.csv')
        pd.DataFrame(results).to_csv(meta_path, index=False)
        logger.info(f"SUCCESS. Generated {len(results)} samples.")
    else:
        logger.warning("No data generated.")

if __name__ == "__main__":
    main()
