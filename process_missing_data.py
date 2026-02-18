#!/usr/bin/env python3
"""
Process Missing Raw Data to Spectrograms
Reads raw binary files from 'missing' folder and generates spectrograms

Date: 6 February 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram
from datetime import datetime
from pathlib import Path
import logging

# Add intial to path for read_mdata
sys.path.insert(0, 'intial')
from read_mdata import read_604rcsv_new_python

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MissingDataProcessor:
    def __init__(self, input_dir='missing', output_dir='dataset_missing'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # PC3 frequency range
        self.pc3_low = 0.01
        self.pc3_high = 0.045
        self.sampling_rate = 1.0
        self.image_size = 224
        
        # Load event list for labels
        self.event_df = pd.read_excel('intial/event_list.xlsx')
        self.event_df['Stasiun'] = self.event_df['Stasiun'].str.upper().str.strip()
        self.event_df['date_str'] = pd.to_datetime(self.event_df['Tanggal']).dt.strftime('%Y%m%d')
        
        # Create output directories
        self._create_directories()
        
        # Stats
        self.stats = {'processed': 0, 'failed': 0, 'skipped': 0}
        
    def _create_directories(self):
        (self.output_dir / 'spectrograms').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metadata').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'logs').mkdir(parents=True, exist_ok=True)

    def _bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        """Apply bandpass filter"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.99))
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def _get_magnitude_class(self, mag):
        """Get magnitude class folder name"""
        if mag < 5.0:
            return "mag_4.0-4.9"
        elif mag < 6.0:
            return "mag_5.0-5.9"
        elif mag < 7.0:
            return "mag_6.0-6.9"
        else:
            return "mag_7.0+"
    
    def _get_azimuth_class(self, azm):
        """Get azimuth class folder name"""
        if azm is None or np.isnan(azm):
            return "azi_unknown"
        
        # Normalize to 0-360
        azm = azm % 360
        
        # 8 directions + center
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
        else:
            return "azi_unknown"
    
    def _generate_spectrogram(self, H, D, Z, station, date_str, hour, mag, azm):
        """Generate 3-component spectrogram"""
        # Extract 1 hour of data
        start_idx = hour * 3600
        end_idx = start_idx + 3600
        
        if end_idx > len(H):
            logger.warning(f"Not enough data for hour {hour}")
            return None
        
        h_hour = H[start_idx:end_idx]
        d_hour = D[start_idx:end_idx]
        z_hour = Z[start_idx:end_idx]
        
        # Check for valid data
        if np.sum(np.isfinite(h_hour)) < 1800:  # At least 50% valid
            logger.warning(f"Too many NaN values in data")
            return None
        
        # Interpolate NaN values
        for arr in [h_hour, d_hour, z_hour]:
            nans = np.isnan(arr)
            if np.any(nans) and not np.all(nans):
                arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
        
        # Remove baseline
        h_hour = h_hour - np.nanmean(h_hour)
        d_hour = d_hour - np.nanmean(d_hour)
        z_hour = z_hour - np.nanmean(z_hour)
        
        # Apply bandpass filter
        try:
            h_filt = self._bandpass_filter(h_hour, self.pc3_low, self.pc3_high, self.sampling_rate)
            d_filt = self._bandpass_filter(d_hour, self.pc3_low, self.pc3_high, self.sampling_rate)
            z_filt = self._bandpass_filter(z_hour, self.pc3_low, self.pc3_high, self.sampling_rate)
        except Exception as e:
            logger.warning(f"Filter error: {e}")
            return None

        # Generate spectrograms
        nperseg = 256
        noverlap = 128
        
        f_h, t_h, Sxx_h = spectrogram(h_filt, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap)
        f_d, t_d, Sxx_d = spectrogram(d_filt, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap)
        f_z, t_z, Sxx_z = spectrogram(z_filt, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap)
        
        # Filter to PC3 range
        freq_mask = (f_h >= self.pc3_low) & (f_h <= self.pc3_high)
        Sxx_h = Sxx_h[freq_mask, :]
        Sxx_d = Sxx_d[freq_mask, :]
        Sxx_z = Sxx_z[freq_mask, :]
        f_h = f_h[freq_mask]
        
        # Convert to dB
        Sxx_h_db = 10 * np.log10(Sxx_h + 1e-10)
        Sxx_d_db = 10 * np.log10(Sxx_d + 1e-10)
        Sxx_z_db = 10 * np.log10(Sxx_z + 1e-10)
        
        # Normalize to 0-1
        def normalize(arr):
            vmin, vmax = np.percentile(arr, [2, 98])
            arr = np.clip(arr, vmin, vmax)
            return (arr - vmin) / (vmax - vmin + 1e-10)
        
        Sxx_h_norm = normalize(Sxx_h_db)
        Sxx_d_norm = normalize(Sxx_d_db)
        Sxx_z_norm = normalize(Sxx_z_db)
        
        # Create RGB image (H=R, D=G, Z=B)
        rgb_image = np.stack([Sxx_h_norm, Sxx_d_norm, Sxx_z_norm], axis=-1)
        
        # Create figure
        fig = plt.figure(figsize=(self.image_size/100, self.image_size/100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(rgb_image, aspect='auto', origin='lower', interpolation='bilinear')
        ax.axis('off')
        
        # Get output path
        mag_class = self._get_magnitude_class(mag)
        azi_class = self._get_azimuth_class(azm)
        
        out_folder = self.output_dir / 'spectrograms' / mag_class / azi_class
        out_folder.mkdir(parents=True, exist_ok=True)
        
        filename = f"{station}_{date_str}_H{hour:02d}_3comp_spec.png"
        out_path = out_folder / filename
        
        plt.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        return str(out_path)

    def _parse_filename(self, filename):
        """Parse filename like S210224.LUT to get date and station"""
        # Format: SYYMMDD.STN
        name = Path(filename).stem  # S210224
        ext = Path(filename).suffix[1:]  # LUT
        
        yy = int(name[1:3])
        mm = int(name[3:5])
        dd = int(name[5:7])
        
        year = 2000 + yy
        station = ext.upper()
        
        return year, mm, dd, station
    
    def _find_events_for_date(self, station, date_str):
        """Find events in event_list for this station and date"""
        events = self.event_df[
            (self.event_df['Stasiun'] == station) & 
            (self.event_df['date_str'] == date_str)
        ]
        return events
    
    def process_all(self):
        """Process all raw data files in missing folder"""
        logger.info("=" * 60)
        logger.info("PROCESSING MISSING RAW DATA")
        logger.info("=" * 60)
        
        metadata_records = []
        
        # Scan all station folders
        for station_folder in self.input_dir.iterdir():
            if not station_folder.is_dir():
                continue
            
            station = station_folder.name.upper()
            logger.info(f"\nProcessing station: {station}")
            
            # Process each raw file
            for raw_file in station_folder.glob("S*.*"):
                if raw_file.suffix.lower() in ['.gz', '.npz']:
                    continue
                
                try:
                    year, month, day, stn = self._parse_filename(raw_file.name)
                    date_str = f"{year}{month:02d}{day:02d}"
                    
                    logger.info(f"  Reading: {raw_file.name} ({year}-{month:02d}-{day:02d})")
                    
                    # Read binary data
                    data = read_604rcsv_new_python(year, month, day, station, str(self.input_dir))
                    
                    H = data['H']
                    D = data['D']
                    Z = data['Z']
                    
                    # Find events for this date
                    events = self._find_events_for_date(station, date_str)
                    
                    if len(events) == 0:
                        logger.warning(f"    No events found in event_list for {station} {date_str}")
                        self.stats['skipped'] += 1
                        continue
                    
                    # Process each event (hour)
                    for _, event in events.iterrows():
                        hour = int(event['Jam'])
                        mag = float(event['Mag'])
                        azm = float(event['Azm']) if pd.notna(event['Azm']) else None
                        
                        # Handle hour 24 -> hour 0 of next day (skip for now)
                        if hour == 24:
                            hour = 23  # Use hour 23 as approximation
                        
                        logger.info(f"    Processing hour {hour}, Mag={mag}, Azm={azm}")
                        
                        result = self._generate_spectrogram(H, D, Z, station, date_str, hour, mag, azm)
                        
                        if result:
                            self.stats['processed'] += 1
                            metadata_records.append({
                                'station': station,
                                'date': f"{year}-{month:02d}-{day:02d}",
                                'hour': hour,
                                'magnitude': mag,
                                'azimuth': azm,
                                'mag_class': self._get_magnitude_class(mag),
                                'azi_class': self._get_azimuth_class(azm),
                                'spectrogram_path': result
                            })
                            logger.info(f"      ✓ Generated: {Path(result).name}")
                        else:
                            self.stats['failed'] += 1
                            logger.warning(f"      ✗ Failed to generate spectrogram")
                            
                except Exception as e:
                    logger.error(f"  Error processing {raw_file.name}: {e}")
                    self.stats['failed'] += 1
        
        # Save metadata
        if metadata_records:
            meta_df = pd.DataFrame(metadata_records)
            meta_path = self.output_dir / 'metadata' / 'processed_events.csv'
            meta_df.to_csv(meta_path, index=False)
            logger.info(f"\nMetadata saved to: {meta_path}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed: {self.stats['processed']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Skipped: {self.stats['skipped']}")
        logger.info(f"Output directory: {self.output_dir}")
        
        return self.stats


if __name__ == "__main__":
    processor = MissingDataProcessor(input_dir='missing', output_dir='dataset_missing')
    processor.process_all()
