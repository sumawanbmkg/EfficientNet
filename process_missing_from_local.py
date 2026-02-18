"""
Process Missing Data from Local mdata2/ folder
Generate spectrograms for events listed in missing_data_updated.csv

This script:
1. Reads missing_data_updated.csv (specific events that need spectrograms)
2. Checks if local data exists in mdata2/{station}/
3. Generates spectrograms for available data
4. Reports which data is still missing (needs SSH fetch)

Author: Kiro
Date: 2026-02-11
"""

import os
import sys
import gzip
import struct
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram
from datetime import datetime
from pathlib import Path
import logging
import shutil
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_missing_local.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MissingDataProcessor:
    """Process missing data events using local mdata2/ files"""
    
    # Station baselines
    BASELINES = {
        'GTO': {'H': 40000, 'Z': 30000},
        'GSI': {'H': 40000, 'Z': 30000},
        'ALR': {'H': 38000, 'Z': 32000},
        'AMB': {'H': 38000, 'Z': 32000},
        'CLP': {'H': 38000, 'Z': 32000},
        'TND': {'H': 40000, 'Z': 30000},
        'SCN': {'H': 38000, 'Z': 32000},
        'MLB': {'H': 38000, 'Z': 32000},
        'SBG': {'H': 38000, 'Z': 32000},
        'YOG': {'H': 38000, 'Z': 32000},
        'LWK': {'H': 38000, 'Z': 32000},
        'SKB': {'H': 38000, 'Z': 32000},
        'TRT': {'H': 38000, 'Z': 32000},
        'PLU': {'H': 38000, 'Z': 32000},
        'LWA': {'H': 38000, 'Z': 32000},
        'KPY': {'H': 38000, 'Z': 32000},
        'LPS': {'H': 38000, 'Z': 32000},
        'SRG': {'H': 38000, 'Z': 32000},
        'LUT': {'H': 38000, 'Z': 32000},
        'SMI': {'H': 38000, 'Z': 32000},
        'TNT': {'H': 38000, 'Z': 32000},
        'SRO': {'H': 38000, 'Z': 32000},
        'TRD': {'H': 38000, 'Z': 32000},
        'JYP': {'H': 38000, 'Z': 32000},
        'BYN': {'H': 38000, 'Z': 32000},
        'KPG': {'H': 38000, 'Z': 32000},
        'TNG': {'H': 38000, 'Z': 32000},
        'TUN': {'H': 38000, 'Z': 32000},
    }
    
    def __init__(self, mdata_dir='mdata2', missing_dir='missing', output_dir='dataset_missing_filled'):
        self.mdata_dir = mdata_dir
        self.missing_dir = missing_dir
        self.output_dir = output_dir
        
        # PC3 frequency range
        self.pc3_low = 0.01
        self.pc3_high = 0.045
        self.sampling_rate = 1.0
        
        # CNN input size
        self.image_size = 224
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'local_available': 0,
            'processed_success': 0,
            'processed_failed': 0,
            'still_missing': 0
        }
        
        # Create output directories
        self._create_directories()
        
        logger.info(f"MissingDataProcessor initialized")
        logger.info(f"Input directory: {mdata_dir}")
        logger.info(f"Output directory: {output_dir}")
    
    def _create_directories(self):
        """Create output directory structure"""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/spectrograms",
            f"{self.output_dir}/spectrograms/by_azimuth",
            f"{self.output_dir}/spectrograms/by_magnitude",
            f"{self.output_dir}/metadata",
        ]
        
        # Azimuth classes
        for az in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
            dirs.append(f"{self.output_dir}/spectrograms/by_azimuth/{az}")
        
        # Magnitude classes
        for mag in ['Small', 'Moderate', 'Medium', 'Large', 'Major']:
            dirs.append(f"{self.output_dir}/spectrograms/by_magnitude/{mag}")
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def get_local_filepath(self, station, date):
        """Get local file path for a given station and date - check both mdata2 and missing folders"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        yy = date.year % 100
        mm = date.month
        dd = date.day
        
        # Format variations to check
        filename_gz = f"S{yy:02d}{mm:02d}{dd:02d}.{station}.gz"
        filename_stn = f"S{yy:02d}{mm:02d}{dd:02d}.{station}"
        
        # Check paths in order of priority
        paths_to_check = [
            # 1. missing folder - direct file
            os.path.join(self.missing_dir, station, filename_gz),
            os.path.join(self.missing_dir, station, filename_stn),
            # 2. missing folder - subfolder by month (GSI format)
            os.path.join(self.missing_dir, station, f"{yy:02d}{mm:02d}", filename_gz),
            os.path.join(self.missing_dir, station, f"{yy:02d}{mm:02d}", filename_stn),
            # 3. mdata2 folder
            os.path.join(self.mdata_dir, station, filename_gz),
            os.path.join(self.mdata_dir, station, filename_stn),
        ]
        
        for path in paths_to_check:
            if os.path.exists(path):
                logger.debug(f"Found file: {path}")
                return path
        
        return None
    
    def read_binary_file(self, filepath, station):
        """Read binary geomagnetic data file (supports both .gz and uncompressed)"""
        num_seconds = 86400
        baseline = self.BASELINES.get(station, {'H': 38000, 'Z': 32000})
        
        data = {
            'Hcomp': np.full(num_seconds, np.nan),
            'Dcomp': np.full(num_seconds, np.nan),
            'Zcomp': np.full(num_seconds, np.nan),
        }
        
        try:
            # Check if gzipped or not
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    binary_data = f.read()
            else:
                with open(filepath, 'rb') as f:
                    binary_data = f.read()
            
            # Skip 32-byte header
            header_size = 32
            record_size = 17
            data_start = binary_data[header_size:]
            
            max_records = min(num_seconds, len(data_start) // record_size)
            
            for i in range(max_records):
                offset = i * record_size
                record = data_start[offset:offset+record_size]
                
                if len(record) < record_size:
                    break
                
                # Parse components (little-endian, signed int16)
                h_dev = struct.unpack('<h', record[0:2])[0] * 0.1
                d_dev = struct.unpack('<h', record[2:4])[0] * 0.1
                z_dev = struct.unpack('<h', record[4:6])[0] * 0.1
                
                # Add baseline
                data['Hcomp'][i] = baseline['H'] + h_dev
                data['Dcomp'][i] = d_dev
                data['Zcomp'][i] = baseline['Z'] + z_dev
            
            valid_samples = np.sum(~np.isnan(data['Hcomp']))
            logger.debug(f"Read {valid_samples} samples from {filepath}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return None
    
    def classify_azimuth(self, azimuth):
        """Classify azimuth to 8 classes"""
        azimuth = float(azimuth) % 360
        if 337.5 <= azimuth or azimuth < 22.5:
            return 'N'
        elif 22.5 <= azimuth < 67.5:
            return 'NE'
        elif 67.5 <= azimuth < 112.5:
            return 'E'
        elif 112.5 <= azimuth < 157.5:
            return 'SE'
        elif 157.5 <= azimuth < 202.5:
            return 'S'
        elif 202.5 <= azimuth < 247.5:
            return 'SW'
        elif 247.5 <= azimuth < 292.5:
            return 'W'
        else:
            return 'NW'
    
    def classify_magnitude(self, magnitude):
        """Classify magnitude to 5 classes"""
        mag = float(magnitude) if pd.notna(magnitude) else 5.0
        if mag < 4.0:
            return 'Small'
        elif mag < 5.0:
            return 'Moderate'
        elif mag < 6.0:
            return 'Medium'
        elif mag < 7.0:
            return 'Large'
        else:
            return 'Major'
    
    def apply_pc3_filter(self, data):
        """Apply PC3 bandpass filter"""
        if len(data) < 100:
            return data
        
        data_clean = np.nan_to_num(data, nan=np.nanmean(data))
        
        nyquist = self.sampling_rate / 2
        low = self.pc3_low / nyquist
        high = self.pc3_high / nyquist
        
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        
        if low >= high:
            return data_clean
        
        try:
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, data_clean)
            return filtered
        except:
            return data_clean
    
    def generate_spectrogram_image(self, data, station, date_str, hour):
        """Generate and save spectrogram image"""
        # Handle hour 24 -> use hour 23 data (last hour of day)
        actual_hour = min(hour, 23) if hour == 24 else hour
        
        # Extract hour data
        start_idx = actual_hour * 3600
        end_idx = start_idx + 3600
        
        h_hour = data['Hcomp'][start_idx:end_idx]
        d_hour = data['Dcomp'][start_idx:end_idx]
        z_hour = data['Zcomp'][start_idx:end_idx]
        
        if np.sum(~np.isnan(h_hour)) < 100:
            logger.warning(f"Insufficient data for {station} {date_str} H{hour}")
            return None
        
        # Apply PC3 filter
        h_pc3 = self.apply_pc3_filter(h_hour)
        d_pc3 = self.apply_pc3_filter(d_hour)
        z_pc3 = self.apply_pc3_filter(z_hour)
        
        # Generate spectrograms
        nperseg = min(256, len(h_pc3) // 4)
        noverlap = nperseg // 2
        
        f_h, t_h, Sxx_h = spectrogram(h_pc3, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap)
        f_d, t_d, Sxx_d = spectrogram(d_pc3, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap)
        f_z, t_z, Sxx_z = spectrogram(z_pc3, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap)
        
        # Limit to PC3 frequency range
        freq_mask = (f_h >= self.pc3_low) & (f_h <= self.pc3_high)
        f_pc3 = f_h[freq_mask]
        Sxx_h_pc3 = Sxx_h[freq_mask, :]
        Sxx_d_pc3 = Sxx_d[freq_mask, :]
        Sxx_z_pc3 = Sxx_z[freq_mask, :]
        
        # Convert to dB
        Sxx_h_db = 10 * np.log10(Sxx_h_pc3 + 1e-10)
        Sxx_d_db = 10 * np.log10(Sxx_d_pc3 + 1e-10)
        Sxx_z_db = 10 * np.log10(Sxx_z_pc3 + 1e-10)
        
        # Create figure
        fig_size = self.image_size / 100.0
        fig, axes = plt.subplots(3, 1, figsize=(fig_size, fig_size))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        
        axes[0].pcolormesh(t_h, f_pc3, Sxx_h_db, shading='gouraud', cmap='jet')
        axes[0].axis('off')
        
        axes[1].pcolormesh(t_d, f_pc3, Sxx_d_db, shading='gouraud', cmap='jet')
        axes[1].axis('off')
        
        axes[2].pcolormesh(t_z, f_pc3, Sxx_z_db, shading='gouraud', cmap='jet')
        axes[2].axis('off')
        
        # Save
        filename = f"{station}_{date_str.replace('-', '')}_H{hour:02d}_3comp_spec.png"
        filepath = os.path.join(self.output_dir, 'spectrograms', filename)
        
        plt.savefig(filepath, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Resize to exact 224x224
        img = Image.open(filepath)
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            img.save(filepath)
        
        # Convert to RGB
        img = Image.open(filepath)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            img.save(filepath)
        
        return filepath
    
    def process_event(self, row):
        """Process a single missing data event"""
        station = row['station']
        date = row['date']
        hour = int(row['hour'])
        magnitude = row['magnitude']
        azimuth = row['azimuth']
        
        # Check if local file exists
        local_path = self.get_local_filepath(station, date)
        
        if local_path is None:
            logger.info(f"[MISSING] {station} {date} - No local file")
            self.stats['still_missing'] += 1
            return None, 'no_local_file'
        
        self.stats['local_available'] += 1
        logger.info(f"[FOUND] {station} {date} - Processing...")
        
        # Read binary data
        data = self.read_binary_file(local_path, station)
        
        if data is None:
            logger.error(f"[FAILED] {station} {date} - Read error")
            self.stats['processed_failed'] += 1
            return None, 'read_error'
        
        # Generate spectrogram
        date_str = str(date)
        image_path = self.generate_spectrogram_image(data, station, date_str, hour)
        
        if image_path is None:
            logger.error(f"[FAILED] {station} {date} H{hour} - Spectrogram error")
            self.stats['processed_failed'] += 1
            return None, 'spectrogram_error'
        
        # Classify
        azimuth_class = self.classify_azimuth(azimuth)
        magnitude_class = self.classify_magnitude(magnitude)
        
        # Copy to class folders
        filename = os.path.basename(image_path)
        az_dest = os.path.join(self.output_dir, 'spectrograms', 'by_azimuth', azimuth_class, filename)
        mag_dest = os.path.join(self.output_dir, 'spectrograms', 'by_magnitude', magnitude_class, filename)
        shutil.copy2(image_path, az_dest)
        shutil.copy2(image_path, mag_dest)
        
        self.stats['processed_success'] += 1
        logger.info(f"[OK] {station} {date} H{hour} -> {azimuth_class}/{magnitude_class}")
        
        return {
            'station': station,
            'date': date_str,
            'hour': hour,
            'azimuth': azimuth,
            'magnitude': magnitude,
            'azimuth_class': azimuth_class,
            'magnitude_class': magnitude_class,
            'spectrogram_file': filename,
            'source': 'local_mdata2'
        }, 'success'
    
    def process_missing_list(self, missing_csv_path):
        """Process all events from missing data CSV"""
        logger.info("="*80)
        logger.info("MISSING DATA PROCESSOR - Starting")
        logger.info("="*80)
        
        # Load missing data list
        logger.info(f"Loading missing data list: {missing_csv_path}")
        df = pd.read_csv(missing_csv_path)
        
        self.stats['total_events'] = len(df)
        logger.info(f"Total missing events: {len(df)}")
        
        # Process each event
        processed_list = []
        still_missing_list = []
        
        for idx, row in df.iterrows():
            if (idx + 1) % 20 == 0:
                logger.info(f"Progress: {idx+1}/{len(df)}")
            
            result, status = self.process_event(row)
            
            if result:
                processed_list.append(result)
            elif status == 'no_local_file':
                still_missing_list.append(row.to_dict())
        
        # Save results
        if processed_list:
            processed_df = pd.DataFrame(processed_list)
            processed_path = os.path.join(self.output_dir, 'metadata', 'processed_events.csv')
            processed_df.to_csv(processed_path, index=False)
            logger.info(f"Processed events saved to {processed_path}")
        
        if still_missing_list:
            still_missing_df = pd.DataFrame(still_missing_list)
            still_missing_path = os.path.join(self.output_dir, 'metadata', 'still_missing_need_ssh.csv')
            still_missing_df.to_csv(still_missing_path, index=False)
            logger.info(f"Still missing (need SSH) saved to {still_missing_path}")
        
        # Print summary
        logger.info("="*80)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Total events: {self.stats['total_events']}")
        logger.info(f"Local files available: {self.stats['local_available']}")
        logger.info(f"Successfully processed: {self.stats['processed_success']}")
        logger.info(f"Processing failed: {self.stats['processed_failed']}")
        logger.info(f"Still missing (need SSH): {self.stats['still_missing']}")
        logger.info(f"Output directory: {self.output_dir}")
        
        return processed_list, still_missing_list


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process missing data from local mdata2/')
    parser.add_argument('--input', default='missing_data_updated.csv',
                        help='Path to missing data CSV')
    parser.add_argument('--output', default='dataset_missing_filled',
                        help='Output directory')
    
    args = parser.parse_args()
    
    processor = MissingDataProcessor(
        mdata_dir='mdata2',
        output_dir=args.output
    )
    
    processed, still_missing = processor.process_missing_list(args.input)
    
    print(f"\n" + "="*60)
    print(f"SUMMARY")
    print(f"="*60)
    print(f"Successfully processed from local: {len(processed)}")
    print(f"Still need SSH fetch: {len(still_missing)}")
    print(f"Output saved to: {args.output}")


if __name__ == '__main__':
    main()
