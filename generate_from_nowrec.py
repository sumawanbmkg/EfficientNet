#!/usr/bin/env python3
"""
Generate Spectrograms from Nowrec Data via SSH
Fetches data from /home/precursor/SEISMO/DATA/{station}/Nowrec/

Date: 6 February 2026
"""

import os
import sys
import paramiko
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram
from datetime import datetime
from pathlib import Path
from io import BytesIO
import gzip
import struct
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

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
    'CLP': {'H': 38000, 'Z': 32000},
    'KPY': {'H': 38000, 'Z': 32000},
    'LUT': {'H': 38000, 'Z': 32000},
    'LWA': {'H': 38000, 'Z': 32000},
    'LWK': {'H': 38000, 'Z': 32000},
    'PLU': {'H': 38000, 'Z': 32000},
    'SKB': {'H': 38000, 'Z': 32000},
    'TRT': {'H': 38000, 'Z': 32000},
    'YOG': {'H': 38000, 'Z': 32000},
    'AMB': {'H': 38000, 'Z': 32000},
}

class NowrecProcessor:
    def __init__(self, output_dir='dataset_nowrec'):
        self.output_dir = Path(output_dir)
        self.pc3_low = 0.01
        self.pc3_high = 0.045
        self.sampling_rate = 1.0
        self.image_size = 224
        
        self.ssh = None
        self.sftp = None
        self.stats = {'processed': 0, 'failed': 0}
        
        (self.output_dir / 'spectrograms').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metadata').mkdir(parents=True, exist_ok=True)
    
    def connect(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(
            hostname=SSH_CONFIG['host'],
            port=SSH_CONFIG['port'],
            username=SSH_CONFIG['username'],
            password=SSH_CONFIG['password'],
            timeout=30
        )
        self.sftp = self.ssh.open_sftp()
        logger.info("Connected to server")
    
    def disconnect(self):
        if self.sftp:
            self.sftp.close()
        if self.ssh:
            self.ssh.close()
        logger.info("Disconnected")
    
    def parse_binary(self, binary_data, station):
        """Parse binary data from Nowrec file"""
        baseline = BASELINES.get(station, {'H': 38000, 'Z': 32000})
        num_seconds = 86400
        
        H = np.full(num_seconds, np.nan)
        D = np.full(num_seconds, np.nan)
        Z = np.full(num_seconds, np.nan)
        
        header_size = 32
        record_size = 17
        data_start = binary_data[header_size:]
        max_records = min(num_seconds, len(data_start) // record_size)
        
        for i in range(max_records):
            offset = i * record_size
            record = data_start[offset:offset+record_size]
            if len(record) < record_size:
                break
            
            h_dev = struct.unpack('<h', record[0:2])[0] * 0.1
            d_dev = struct.unpack('<h', record[2:4])[0] * 0.1
            z_dev = struct.unpack('<h', record[4:6])[0] * 0.1
            
            H[i] = baseline['H'] + h_dev
            D[i] = d_dev
            Z[i] = baseline['Z'] + z_dev
        
        return H, D, Z

    def _bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = max(0.001, min(lowcut / nyq, 0.99))
        high = max(low + 0.001, min(highcut / nyq, 0.99))
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def _get_magnitude_class(self, mag):
        if mag < 5.0:
            return "mag_4.0-4.9"
        elif mag < 6.0:
            return "mag_5.0-5.9"
        elif mag < 7.0:
            return "mag_6.0-6.9"
        else:
            return "mag_7.0+"
    
    def _get_azimuth_class(self, azm):
        if azm is None or np.isnan(azm):
            return "azi_unknown"
        azm = azm % 360
        if 337.5 <= azm or azm < 22.5:
            return "azi_N"
        elif 22.5 <= azm < 67.5:
            return "azi_NE"
        elif 67.5 <= azm < 112.5:
            return "azi_E"
        elif 112.5 <= azm < 157.5:
            return "azi_SE"
        elif 157.5 <= azm < 202.5:
            return "azi_S"
        elif 202.5 <= azm < 247.5:
            return "azi_SW"
        elif 247.5 <= azm < 292.5:
            return "azi_W"
        elif 292.5 <= azm < 337.5:
            return "azi_NW"
        return "azi_unknown"
    
    def generate_spectrogram(self, H, D, Z, station, date_str, hour, mag, azm):
        """Generate 3-component spectrogram"""
        start_idx = hour * 3600
        end_idx = start_idx + 3600
        
        if end_idx > len(H):
            return None
        
        h_hour = H[start_idx:end_idx].copy()
        d_hour = D[start_idx:end_idx].copy()
        z_hour = Z[start_idx:end_idx].copy()
        
        if np.sum(np.isfinite(h_hour)) < 1800:
            return None
        
        # Interpolate NaN
        for arr in [h_hour, d_hour, z_hour]:
            nans = np.isnan(arr)
            if np.any(nans) and not np.all(nans):
                arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
        
        # Remove baseline
        h_hour -= np.nanmean(h_hour)
        d_hour -= np.nanmean(d_hour)
        z_hour -= np.nanmean(z_hour)
        
        try:
            h_filt = self._bandpass_filter(h_hour, self.pc3_low, self.pc3_high, self.sampling_rate)
            d_filt = self._bandpass_filter(d_hour, self.pc3_low, self.pc3_high, self.sampling_rate)
            z_filt = self._bandpass_filter(z_hour, self.pc3_low, self.pc3_high, self.sampling_rate)
        except:
            return None
        
        # Spectrogram
        nperseg, noverlap = 256, 128
        f_h, t_h, Sxx_h = spectrogram(h_filt, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap)
        f_d, t_d, Sxx_d = spectrogram(d_filt, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap)
        f_z, t_z, Sxx_z = spectrogram(z_filt, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap)
        
        freq_mask = (f_h >= self.pc3_low) & (f_h <= self.pc3_high)
        Sxx_h, Sxx_d, Sxx_z = Sxx_h[freq_mask], Sxx_d[freq_mask], Sxx_z[freq_mask]
        
        # Convert to dB and normalize
        def to_db_norm(arr):
            db = 10 * np.log10(arr + 1e-10)
            vmin, vmax = np.percentile(db, [2, 98])
            return np.clip((db - vmin) / (vmax - vmin + 1e-10), 0, 1)
        
        rgb = np.stack([to_db_norm(Sxx_h), to_db_norm(Sxx_d), to_db_norm(Sxx_z)], axis=-1)
        
        # Save figure
        fig = plt.figure(figsize=(self.image_size/100, self.image_size/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(rgb, aspect='auto', origin='lower', interpolation='bilinear')
        ax.axis('off')
        
        mag_class = self._get_magnitude_class(mag)
        azi_class = self._get_azimuth_class(azm)
        out_folder = self.output_dir / 'spectrograms' / mag_class / azi_class
        out_folder.mkdir(parents=True, exist_ok=True)
        
        filename = f"{station}_{date_str}_H{hour:02d}_3comp_spec.png"
        out_path = out_folder / filename
        plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        return str(out_path)

    def process_all(self):
        """Process all files from nowrec_available.csv"""
        logger.info("=" * 60)
        logger.info("GENERATING SPECTROGRAMS FROM NOWREC")
        logger.info("=" * 60)
        
        # Load available files
        avail_df = pd.read_csv('nowrec_available.csv')
        logger.info(f"Files to process: {len(avail_df)}")
        
        self.connect()
        metadata_records = []
        
        try:
            for idx, row in avail_df.iterrows():
                station = row['station']
                remote_path = row['path']
                hour = int(row['hour'])
                mag = float(row['magnitude'])
                azm = float(row['azimuth']) if pd.notna(row['azimuth']) else None
                date_str = row['date'].replace('-', '')
                
                # Handle hour 24
                if hour == 24:
                    hour = 23
                
                logger.info(f"[{idx+1}/{len(avail_df)}] {row['filename']} H{hour}")
                
                try:
                    # Download file
                    with BytesIO() as buf:
                        self.sftp.getfo(remote_path, buf)
                        buf.seek(0)
                        binary_data = buf.read()
                    
                    # Check if compressed
                    if remote_path.endswith('.gz'):
                        binary_data = gzip.decompress(binary_data)
                    
                    # Parse binary
                    H, D, Z = self.parse_binary(binary_data, station)
                    
                    # Generate spectrogram
                    result = self.generate_spectrogram(H, D, Z, station, date_str, hour, mag, azm)
                    
                    if result:
                        self.stats['processed'] += 1
                        metadata_records.append({
                            'station': station,
                            'date': row['date'],
                            'hour': hour,
                            'magnitude': mag,
                            'azimuth': azm,
                            'mag_class': self._get_magnitude_class(mag),
                            'azi_class': self._get_azimuth_class(azm),
                            'spectrogram_path': result
                        })
                        logger.info(f"  ✓ Generated")
                    else:
                        self.stats['failed'] += 1
                        logger.warning(f"  ✗ Failed (data quality)")
                        
                except Exception as e:
                    self.stats['failed'] += 1
                    logger.error(f"  ✗ Error: {e}")
        
        finally:
            self.disconnect()
        
        # Save metadata
        if metadata_records:
            meta_df = pd.DataFrame(metadata_records)
            meta_df.to_csv(self.output_dir / 'metadata' / 'processed_events.csv', index=False)
        
        logger.info("\n" + "=" * 60)
        logger.info("COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed: {self.stats['processed']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Output: {self.output_dir}")
        
        return self.stats


if __name__ == "__main__":
    processor = NowrecProcessor(output_dir='dataset_nowrec')
    processor.process_all()
