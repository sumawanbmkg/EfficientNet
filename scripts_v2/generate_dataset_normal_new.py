"""
Generate Normal Dataset (New Scan 2023-2025)
============================================
Script ini bertujuan untuk mengambil sampel data Normal (Non-Prekursor) dari 
periode 2023-2025 agar memiliki kualitas visual yang setara dengan dataset Gempa baru.

Logika:
1. Random Sampling tanggal & jam (2023-2025).
2. Exclusion Check: Memastikan TIDAK ADA gempa M>4.0 di radius 1000km pada jam tersebut.
3. Download & Process: Menggunakan protokol yang sama dengan generate_dataset_moderate.
"""

import os
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
import random

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
REPO_FILE = 'mdata2/repositor_gempa_bumi 2018-2025.csv'
STATION_FILE = 'mdata2/lokasi_stasiun.csv'
OUTPUT_DIR = 'dataset_normal_new' 
TARGET_COUNT = 1000

# Baselines
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

class NormalScanner:
    def __init__(self):
        self.ssh_client = None
        self.sftp_client = None
        self.stations_df = self.load_stations()
        self.earthquake_df = self.load_earthquakes() # Untuk exclusion check
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

    def load_earthquakes(self):
        try:
            df = pd.read_csv(REPO_FILE)
            df['dt'] = pd.to_datetime(df['Date time']).dt.tz_localize(None)
            return df
        except Exception as e:
            logger.error(f"Failed to load earthquakes: {e}")
            return pd.DataFrame()

    def connect(self):
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(**SSH_CONFIG, timeout=20)
            self.sftp_client = self.ssh_client.open_sftp()
            return True
        except Exception as e:
            logger.error(f"SSH Error: {e}")
            return False
            
    def close(self):
        if self.sftp_client: self.sftp_client.close()
        if self.ssh_client: self.ssh_client.close()
        
    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def is_quiet_period(self, station_lat, station_lon, target_time):
        """
        Cek apakah ada gempa M>4.0 di radius 1000km dalam window +/- 3 jam
        """
        if self.earthquake_df.empty: return True # Blind assumption
        
        # Pastikan target_time naive
        if target_time.tzinfo is not None:
            target_time = target_time.replace(tzinfo=None)

        # Filter waktu (+/- 3 jam)
        start_win = target_time - timedelta(hours=3)
        end_win = target_time + timedelta(hours=3)
        
        candidates = self.earthquake_df[
            (self.earthquake_df['dt'] >= start_win) & 
            (self.earthquake_df['dt'] <= end_win) &
            (self.earthquake_df['Magnitude'] >= 4.0)
        ]
        
        if candidates.empty: return True
        
        # Cek jarak
        for _, eq in candidates.iterrows():
            dist = self.haversine(station_lat, station_lon, eq['Latitude'], eq['Longitude'])
            if dist < 1000: # Ada gempa dekat!
                return False
                
        return True

    def generate_spectrogram(self, H_data, Z_data, output_path):
        if len(H_data) < 100: return False
        try:
            valid_mask = (H_data > 0) & ~np.isnan(H_data) & ~np.isnan(Z_data)
            if np.sum(valid_mask) < 100: return False
            
            zh_ratio = np.abs(Z_data[valid_mask] / H_data[valid_mask])
            f, t, Sxx = signal.spectrogram(zh_ratio, fs=1, nperseg=min(256, len(zh_ratio)//4))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='jet')
            plt.axis('off')
            plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            return True
        except:
            return False

    def process_random_slot(self):
        # Pick Random Station
        if self.stations_df.empty: return None
        station_row = self.stations_df.sample(1).iloc[0]
        station = station_row['Kode Stasiun']
        
        # Pick Random Date (2023-2025)
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2025, 12, 31)
        random_days = random.randint(0, (end_date - start_date).days)
        target_date = start_date + timedelta(days=random_days)
        hour = random.randint(0, 23)
        target_dt = target_date.replace(hour=hour, minute=0, second=0)
        
        # 1. Exclusion Check
        if not self.is_quiet_period(station_row['Latitude'], station_row['Longitude'], target_dt):
            return None # Skip, noisy period
            
        # 2. Fetch Data
        yy = target_dt.year % 100
        mm = target_dt.month
        dd = target_dt.day
        
        foldername = f"{yy:02d}{mm:02d}"
        filename_remote = f"S{yy:02d}{mm:02d}{dd:02d}.{station}"
        remote_path = f"/home/precursor/SEISMO/DATA/{station}/SData/{foldername}/{filename_remote}"
        
        save_filename = f"{station}_{target_dt.strftime('%Y%m%d')}_H{hour:02d}_Normal.png"
        save_path = os.path.join(OUTPUT_DIR, 'spectrograms', save_filename)
        
        if os.path.exists(save_path): return None # Already exists
        
        try:
            with BytesIO() as buffer:
                self.sftp_client.getfo(remote_path, buffer)
                buffer.seek(0)
                binary_data = buffer.read()
                
            baseline = BASELINES.get(station, BASELINES['DEFAULT'])
            data_start = binary_data[32:] # Header skip
            
            start_idx = hour * 3600
            end_idx = (hour + 1) * 3600
            if start_idx >= len(data_start) // 17: return None
            
            H_data = []
            Z_data = []
            
            for i in range(start_idx, min(end_idx, len(data_start) // 17)):
                offset = i * 17
                record = data_start[offset:offset+17]
                h_dev = struct.unpack('<h', record[0:2])[0] * 0.1
                z_dev = struct.unpack('<h', record[4:6])[0] * 0.1
                H_data.append(baseline['H'] + h_dev)
                Z_data.append(baseline['Z'] + z_dev)
                
            if self.generate_spectrogram(np.array(H_data), np.array(Z_data), save_path):
                return {
                    'filename': save_filename,
                    'filepath': f"spectrograms/{save_filename}",
                    'station': station,
                    'date': target_dt.strftime("%Y-%m-%d"),
                    'hour': hour,
                    'magnitude': 0,
                    'magnitude_class': 'Normal',
                    'azimuth': 0,
                    'azimuth_class': 'Unknown'
                }
                
        except Exception as e:
            pass # File not found or connection error
            
        return None

def main():
    logger.info("PHASE 2 - NORMAL DATASET GENERATOR (Random Sampling 2023-2025)")
    scanner = NormalScanner()
    if not scanner.connect(): return
    
    results = []
    attempts = 0
    pbar = tqdm(total=TARGET_COUNT, desc="Generated Normal")
    
    while len(results) < TARGET_COUNT and attempts < TARGET_COUNT * 20:
        attempts += 1
        try:
            res = scanner.process_random_slot()
            if res:
                results.append(res)
                pbar.update(1)
        except Exception as e:
            logger.error(f"Error in sampling loop: {e}")
            continue
            
    pbar.close()
    scanner.close()
    
    if results:
        meta_path = os.path.join(OUTPUT_DIR, 'metadata.csv')
        pd.DataFrame(results).to_csv(meta_path, index=False)
        logger.info(f"SUCCESS. Generated {len(results)} Normal samples.")

if __name__ == "__main__":
    main()
