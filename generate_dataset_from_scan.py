"""
Generate Spectrogram Dataset from Scan Results
===============================================
Script untuk membuat dataset spectrogram dari hasil scan precursor.
Menggunakan data dari new_event_scanned.csv dan menghasilkan spectrogram
untuk training model.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SSH Configuration
SSH_CONFIG = {
    'host': '202.90.198.224',
    'port': 4343,
    'username': 'precursor',
    'password': 'otomatismon'
}

# Station baselines
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
    'LPS': {'H': 38000, 'Z': 32000},
    'SRG': {'H': 38000, 'Z': 32000},
    'SKB': {'H': 38000, 'Z': 32000},
    'CLP': {'H': 38000, 'Z': 32000},
    'LUT': {'H': 38000, 'Z': 32000},
    'SMI': {'H': 38000, 'Z': 32000},
    'SRO': {'H': 38000, 'Z': 32000},
    'TNT': {'H': 38000, 'Z': 32000},
    'TND': {'H': 40000, 'Z': 30000},
    'LWK': {'H': 38000, 'Z': 32000},
    'PLU': {'H': 38000, 'Z': 32000},
    'TRD': {'H': 38000, 'Z': 32000},
    'JYP': {'H': 38000, 'Z': 32000},
}


class SpectrogramGenerator:
    """Generate spectrograms from geomagnetic data."""
    
    def __init__(self, output_dir='dataset_new_events'):
        self.output_dir = output_dir
        self.ssh_client = None
        self.sftp_client = None
        os.makedirs(output_dir, exist_ok=True)
        
    def connect(self):
        """Connect to SSH server."""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(
                hostname=SSH_CONFIG['host'],
                port=SSH_CONFIG['port'],
                username=SSH_CONFIG['username'],
                password=SSH_CONFIG['password'],
                timeout=60
            )
            self.sftp_client = self.ssh_client.open_sftp()
            logger.info("Connected to SSH server")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from SSH server."""
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
    
    def fetch_hourly_data(self, date, hour, station):
        """Fetch 1 hour of data for spectrogram generation."""
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        yy = date.year % 100
        mm = date.month
        dd = date.day
        
        filename = f"S{yy:02d}{mm:02d}{dd:02d}.{station}"
        remote_path = f"/home/precursor/SEISMO/DATA/{station}/SData/{yy:02d}{mm:02d}/{filename}"
        
        try:
            with BytesIO() as buffer:
                self.sftp_client.getfo(remote_path, buffer)
                buffer.seek(0)
                binary_data = buffer.read()
                
                # Parse binary data
                baseline = BASELINES.get(station, {'H': 38000, 'Z': 32000})
                header_size = 32
                record_size = 17
                data_start = binary_data[header_size:]
                
                # Extract hour data (3600 seconds)
                start_idx = hour * 3600
                end_idx = (hour + 1) * 3600
                
                H_data = []
                Z_data = []
                
                for i in range(start_idx, min(end_idx, len(data_start) // record_size)):
                    offset = i * record_size
                    record = data_start[offset:offset+record_size]
                    
                    if len(record) < record_size:
                        break
                    
                    h_dev = struct.unpack('<h', record[0:2])[0] * 0.1
                    z_dev = struct.unpack('<h', record[4:6])[0] * 0.1
                    
                    H_data.append(baseline['H'] + h_dev)
                    Z_data.append(baseline['Z'] + z_dev)
                
                return np.array(H_data), np.array(Z_data)
                
        except Exception as e:
            logger.debug(f"Error fetching {remote_path}: {e}")
            return None, None
    
    def generate_spectrogram(self, H_data, Z_data, output_path, title=""):
        """Generate and save spectrogram image."""
        if H_data is None or Z_data is None or len(H_data) < 100:
            return False
        
        try:
            # Calculate Z/H ratio
            valid_mask = (H_data > 0) & ~np.isnan(H_data) & ~np.isnan(Z_data)
            if np.sum(valid_mask) < 100:
                return False
            
            zh_ratio = np.abs(Z_data[valid_mask] / H_data[valid_mask])
            
            # Generate spectrogram
            fs = 1  # 1 Hz sampling
            nperseg = min(256, len(zh_ratio) // 4)
            
            f, t, Sxx = signal.spectrogram(zh_ratio, fs=fs, nperseg=nperseg)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot spectrogram
            im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                             shading='gouraud', cmap='jet')
            
            ax.set_ylabel('Frequency [Hz]')
            ax.set_xlabel('Time [s]')
            ax.set_title(title)
            
            plt.colorbar(im, ax=ax, label='Power [dB]')
            
            # Save
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating spectrogram: {e}")
            return False
    
    def get_magnitude_class(self, magnitude):
        """Classify magnitude into categories."""
        if magnitude >= 7.0:
            return 'Large'
        elif magnitude >= 6.0:
            return 'Medium'
        elif magnitude >= 5.0:
            return 'Moderate'
        else:
            return 'Small'
    
    def get_azimuth_class(self, azimuth):
        """Classify azimuth into 8 directions."""
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        idx = int((azimuth + 22.5) / 45) % 8
        return directions[idx]
    
    def process_scan_results(self, scan_file='new_event_scanned.csv'):
        """Process scan results and generate spectrograms."""
        if not os.path.exists(scan_file):
            logger.error(f"Scan file not found: {scan_file}")
            return
        
        df = pd.read_csv(scan_file)
        logger.info(f"Loaded {len(df)} events from {scan_file}")
        
        if not self.connect():
            return
        
        try:
            generated = 0
            failed = 0
            
            for idx, row in df.iterrows():
                station = row['Stasiun']
                date = row['Tanggal']
                hour = int(row['Jam'])
                azimuth = row['Azm']
                magnitude = row['Mag']
                
                # Get classes
                mag_class = self.get_magnitude_class(magnitude)
                azm_class = self.get_azimuth_class(azimuth)
                
                # Create output directory structure
                class_dir = os.path.join(self.output_dir, mag_class, azm_class)
                os.makedirs(class_dir, exist_ok=True)
                
                # Generate filename
                date_str = date.replace('-', '')
                filename = f"{station}_{date_str}_H{hour:02d}_M{magnitude:.1f}.png"
                output_path = os.path.join(class_dir, filename)
                
                # Skip if already exists
                if os.path.exists(output_path):
                    logger.info(f"Skipping existing: {filename}")
                    continue
                
                # Fetch data and generate spectrogram
                H_data, Z_data = self.fetch_hourly_data(date, hour, station)
                
                title = f"{station} {date} H{hour:02d} - M{magnitude:.1f} Azm:{azimuth:.0f}°"
                
                if self.generate_spectrogram(H_data, Z_data, output_path, title):
                    generated += 1
                    logger.info(f"[{idx+1}/{len(df)}] Generated: {filename}")
                else:
                    failed += 1
                    logger.warning(f"[{idx+1}/{len(df)}] Failed: {filename}")
                
                # Progress save every 50
                if (idx + 1) % 50 == 0:
                    logger.info(f"Progress: {generated} generated, {failed} failed")
            
            logger.info(f"\nComplete! Generated: {generated}, Failed: {failed}")
            
        finally:
            self.disconnect()
    
    def create_metadata(self, scan_file='new_event_scanned.csv'):
        """Create metadata CSV for the dataset."""
        if not os.path.exists(scan_file):
            return
        
        df = pd.read_csv(scan_file)
        
        metadata = []
        for idx, row in df.iterrows():
            station = row['Stasiun']
            date = row['Tanggal']
            hour = int(row['Jam'])
            azimuth = row['Azm']
            magnitude = row['Mag']
            
            mag_class = self.get_magnitude_class(magnitude)
            azm_class = self.get_azimuth_class(azimuth)
            
            date_str = date.replace('-', '')
            filename = f"{station}_{date_str}_H{hour:02d}_M{magnitude:.1f}.png"
            filepath = os.path.join(mag_class, azm_class, filename)
            
            metadata.append({
                'filename': filename,
                'filepath': filepath,
                'station': station,
                'date': date,
                'hour': hour,
                'azimuth': azimuth,
                'magnitude': magnitude,
                'mag_class': mag_class,
                'azm_class': azm_class
            })
        
        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv(os.path.join(self.output_dir, 'metadata.csv'), index=False)
        logger.info(f"Created metadata.csv with {len(meta_df)} entries")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate spectrogram dataset from scan results')
    parser.add_argument('--input', type=str, default='new_event_scanned.csv', help='Input scan file')
    parser.add_argument('--output', type=str, default='dataset_new_events', help='Output directory')
    parser.add_argument('--metadata-only', action='store_true', help='Only create metadata')
    
    args = parser.parse_args()
    
    generator = SpectrogramGenerator(output_dir=args.output)
    
    if args.metadata_only:
        generator.create_metadata(args.input)
    else:
        generator.process_scan_results(args.input)
        generator.create_metadata(args.input)


if __name__ == '__main__':
    main()
