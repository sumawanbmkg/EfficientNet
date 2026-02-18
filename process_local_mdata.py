"""
Process Local Geomagnetic Data from mdata2/ folder
Generate spectrograms for earthquake precursor detection

This script processes local .gz files from mdata2/ folder and generates
spectrograms that can be merged with SSH-scanned data.

Features:
- Reads gzipped binary files from mdata2/{station}/S{yymmdd}.{station}.gz
- Matches with earthquake catalog to find precursor windows (4-20 days before)
- Calculates Z/H ratio maximum per hour
- Generates 224x224 spectrograms for CNN training
- Output format compatible with existing dataset

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
from datetime import datetime, timedelta
from pathlib import Path
import logging
import math
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_local_mdata.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LocalMdataProcessor:
    """Process local geomagnetic data from mdata2/ folder"""
    
    # Station baselines (from geomagnetic_fetcher.py)
    BASELINES = {
        'GTO': {'H': 40000, 'Z': 30000},
        'GSI': {'H': 40000, 'Z': 30000},
        'ALR': {'H': 38000, 'Z': 32000},
        'AMB': {'H': 38000, 'Z': 32000},
        'BTN': {'H': 38000, 'Z': 32000},
        'CLP': {'H': 38000, 'Z': 32000},
        'TND': {'H': 40000, 'Z': 30000},
        'SCN': {'H': 38000, 'Z': 32000},
        'MLB': {'H': 38000, 'Z': 32000},
        'SBG': {'H': 38000, 'Z': 32000},
        'YOG': {'H': 38000, 'Z': 32000},
        'MJB': {'H': 38000, 'Z': 32000},
        'LWK': {'H': 38000, 'Z': 32000},
        'SMG': {'H': 38000, 'Z': 32000},
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
    }
    
    # Station coordinates (from lokasi_stasiun.csv)
    STATION_COORDS = {
        'SBG': (5.87679, 95.3382),
        'SCN': (-0.545875, 100.298),
        'KPY': (-3.67999, 102.582),
        'LWA': (-5.01744, 104.058),
        'LPS': (-5.7887, 105.583),
        'SRG': (-6.17132, 106.051),
        'SKB': (-7.07442, 106.531),
        'CLP': (-7.7194, 109.015),
        'YOG': (-7.73119, 110.354),
        'TRT': (-7.70543, 112.635),
        'LUT': (-8.21959, 116.407),
        'ALR': (-8.14423, 124.59),
        'SMI': (-7.66885, 131.579),
        'SRO': (-0.862834, 131.259),
        'TNT': (0.812571, 127.367),
        'TND': (1.29462, 124.925),
        'GTO': (0.555972, 123.141),
        'LWK': (-1.00019, 122.784),
        'PLU': (-0.62027, 119.859),
        'TRD': (2.13628, 117.424),
        'JYP': (-2.51447, 140.704),
        'AMB': (-3.67585, 128.111),
        'GSI': (1.30396, 97.5755),
        'MLB': (4.04902, 96.2477),
    }
    
    def __init__(self, mdata_dir='mdata2', output_dir='dataset_local_mdata'):
        self.mdata_dir = mdata_dir
        self.output_dir = output_dir
        
        # PC3 frequency range
        self.pc3_low = 0.01
        self.pc3_high = 0.045
        self.sampling_rate = 1.0
        
        # CNN input size
        self.image_size = 224
        
        # Precursor window (days before earthquake)
        self.precursor_min_days = 4
        self.precursor_max_days = 20
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'precursors_found': 0,
            'spectrograms_generated': 0
        }
        
        # Create output directories
        self._create_directories()
        
        logger.info(f"LocalMdataProcessor initialized")
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
    
    def read_binary_gz(self, filepath, station):
        """Read gzipped binary geomagnetic data file"""
        num_seconds = 86400
        baseline = self.BASELINES.get(station, {'H': 38000, 'Z': 32000})
        
        data = {
            'Hcomp': np.full(num_seconds, np.nan),
            'Dcomp': np.full(num_seconds, np.nan),
            'Zcomp': np.full(num_seconds, np.nan),
        }
        
        try:
            with gzip.open(filepath, 'rb') as f:
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
    
    def calculate_zh_ratio_hourly(self, data):
        """Calculate Z/H ratio maximum per hour"""
        zh_hourly = []
        
        for hour in range(24):
            start_idx = hour * 3600
            end_idx = start_idx + 3600
            
            h_hour = data['Hcomp'][start_idx:end_idx]
            z_hour = data['Zcomp'][start_idx:end_idx]
            
            # Remove NaN
            valid_mask = ~np.isnan(h_hour) & ~np.isnan(z_hour) & (h_hour != 0)
            
            if np.sum(valid_mask) < 100:
                zh_hourly.append({'hour': hour, 'zh_max': np.nan, 'valid': False})
                continue
            
            h_valid = h_hour[valid_mask]
            z_valid = z_hour[valid_mask]
            
            # Calculate Z/H ratio
            zh_ratio = np.abs(z_valid / h_valid)
            zh_max = np.max(zh_ratio)
            
            zh_hourly.append({
                'hour': hour,
                'zh_max': zh_max,
                'zh_mean': np.mean(zh_ratio),
                'valid': True
            })
        
        return zh_hourly
    
    def calculate_azimuth(self, station_lat, station_lon, eq_lat, eq_lon):
        """Calculate azimuth from station to earthquake"""
        lat1 = math.radians(station_lat)
        lat2 = math.radians(eq_lat)
        dlon = math.radians(eq_lon - station_lon)
        
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        azimuth = math.degrees(math.atan2(x, y))
        return (azimuth + 360) % 360
    
    def classify_azimuth(self, azimuth):
        """Classify azimuth to 8 classes"""
        azimuth = azimuth % 360
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
        if magnitude < 4.0:
            return 'Small'
        elif magnitude < 5.0:
            return 'Moderate'
        elif magnitude < 6.0:
            return 'Medium'
        elif magnitude < 7.0:
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
    
    def generate_spectrogram_image(self, data, station, date_str, hour, azimuth_class, magnitude_class):
        """Generate and save spectrogram image"""
        # Extract hour data
        start_idx = hour * 3600
        end_idx = start_idx + 3600
        
        h_hour = data['Hcomp'][start_idx:end_idx]
        d_hour = data['Dcomp'][start_idx:end_idx]
        z_hour = data['Zcomp'][start_idx:end_idx]
        
        if np.sum(~np.isnan(h_hour)) < 100:
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
        
        # Copy to class folders
        import shutil
        az_dest = os.path.join(self.output_dir, 'spectrograms', 'by_azimuth', azimuth_class, filename)
        mag_dest = os.path.join(self.output_dir, 'spectrograms', 'by_magnitude', magnitude_class, filename)
        shutil.copy2(filepath, az_dest)
        shutil.copy2(filepath, mag_dest)
        
        self.stats['spectrograms_generated'] += 1
        
        return filepath

    def scan_available_files(self):
        """Scan all available files in mdata2/ folder"""
        available_files = {}
        
        for station_dir in os.listdir(self.mdata_dir):
            station_path = os.path.join(self.mdata_dir, station_dir)
            
            if not os.path.isdir(station_path):
                continue
            
            if station_dir not in self.STATION_COORDS:
                continue
            
            files = []
            for f in os.listdir(station_path):
                if f.endswith('.gz') and f.startswith('S'):
                    # Parse filename: S{yymmdd}.{station}.gz
                    try:
                        parts = f.replace('.gz', '').split('.')
                        date_part = parts[0][1:]  # Remove 'S'
                        yy = int(date_part[:2])
                        mm = int(date_part[2:4])
                        dd = int(date_part[4:6])
                        
                        # Convert to full year
                        year = 2000 + yy if yy < 50 else 1900 + yy
                        
                        date = datetime(year, mm, dd)
                        files.append({
                            'filename': f,
                            'date': date,
                            'path': os.path.join(station_path, f)
                        })
                    except Exception as e:
                        logger.warning(f"Could not parse filename {f}: {e}")
            
            if files:
                available_files[station_dir] = sorted(files, key=lambda x: x['date'])
                logger.info(f"Station {station_dir}: {len(files)} files")
        
        return available_files
    
    def find_precursor_windows(self, earthquakes_df, available_files):
        """Find precursor windows that match available local data"""
        precursor_events = []
        
        for idx, eq in earthquakes_df.iterrows():
            eq_date = pd.to_datetime(eq['datetime'])
            eq_lat = eq['Latitude']
            eq_lon = eq['Longitude']
            eq_mag = eq['Magnitude']
            
            # Precursor window: 4-20 days before earthquake
            window_start = eq_date - timedelta(days=self.precursor_max_days)
            window_end = eq_date - timedelta(days=self.precursor_min_days)
            
            # Check each station
            for station, files in available_files.items():
                station_lat, station_lon = self.STATION_COORDS[station]
                
                # Find files in precursor window
                for file_info in files:
                    file_date = file_info['date']
                    
                    if window_start <= file_date <= window_end:
                        # Calculate azimuth
                        azimuth = self.calculate_azimuth(
                            station_lat, station_lon, eq_lat, eq_lon
                        )
                        
                        precursor_events.append({
                            'earthquake_id': eq.get('event_id', idx),
                            'earthquake_date': eq_date,
                            'earthquake_lat': eq_lat,
                            'earthquake_lon': eq_lon,
                            'earthquake_mag': eq_mag,
                            'station': station,
                            'precursor_date': file_date,
                            'days_before': (eq_date - file_date).days,
                            'azimuth': azimuth,
                            'file_path': file_info['path'],
                            'filename': file_info['filename']
                        })
        
        logger.info(f"Found {len(precursor_events)} potential precursor events")
        return precursor_events
    
    def process_precursor_event(self, event):
        """Process a single precursor event"""
        try:
            station = event['station']
            file_path = event['file_path']
            precursor_date = event['precursor_date']
            azimuth = event['azimuth']
            magnitude = event['earthquake_mag']
            
            # Read binary data
            data = self.read_binary_gz(file_path, station)
            
            if data is None:
                self.stats['files_failed'] += 1
                return None
            
            self.stats['files_processed'] += 1
            
            # Calculate Z/H ratio per hour
            zh_hourly = self.calculate_zh_ratio_hourly(data)
            
            # Find hour with maximum Z/H ratio (potential precursor)
            valid_hours = [h for h in zh_hourly if h['valid'] and not np.isnan(h['zh_max'])]
            
            if not valid_hours:
                return None
            
            # Get hour with max Z/H
            max_zh_hour = max(valid_hours, key=lambda x: x['zh_max'])
            
            # Check if Z/H ratio indicates potential precursor (threshold ~0.9-1.1)
            if max_zh_hour['zh_max'] < 0.85:
                return None
            
            self.stats['precursors_found'] += 1
            
            # Classify
            azimuth_class = self.classify_azimuth(azimuth)
            magnitude_class = self.classify_magnitude(magnitude)
            
            # Generate spectrogram
            date_str = precursor_date.strftime('%Y-%m-%d')
            hour = max_zh_hour['hour']
            
            image_path = self.generate_spectrogram_image(
                data, station, date_str, hour, azimuth_class, magnitude_class
            )
            
            if image_path is None:
                return None
            
            # Return metadata
            return {
                'No': self.stats['precursors_found'],
                'Stasiun': station,
                'Tanggal': date_str,
                'Jam': hour,
                'Azm': round(azimuth, 2),
                'Mag': magnitude,
                'azimuth_class': azimuth_class,
                'magnitude_class': magnitude_class,
                'zh_max': round(max_zh_hour['zh_max'], 4),
                'earthquake_date': event['earthquake_date'].strftime('%Y-%m-%d'),
                'days_before': event['days_before'],
                'spectrogram_file': os.path.basename(image_path),
                'source': 'local_mdata2'
            }
            
        except Exception as e:
            logger.error(f"Error processing event: {e}")
            self.stats['files_failed'] += 1
            return None
    
    def process_all(self, earthquake_catalog_path, min_magnitude=6.0, max_events=None):
        """Process all available local data"""
        logger.info("="*80)
        logger.info("LOCAL MDATA PROCESSOR - Starting")
        logger.info("="*80)
        
        # Load earthquake catalog
        logger.info(f"Loading earthquake catalog: {earthquake_catalog_path}")
        eq_df = pd.read_csv(earthquake_catalog_path)
        eq_df['datetime'] = pd.to_datetime(eq_df['datetime'])
        
        # Filter by magnitude
        eq_df = eq_df[eq_df['Magnitude'] >= min_magnitude]
        logger.info(f"Earthquakes with M >= {min_magnitude}: {len(eq_df)}")
        
        # Scan available files
        logger.info("Scanning available local files...")
        available_files = self.scan_available_files()
        
        total_files = sum(len(files) for files in available_files.values())
        logger.info(f"Total available files: {total_files}")
        
        # Find precursor windows
        logger.info("Finding precursor windows...")
        precursor_events = self.find_precursor_windows(eq_df, available_files)
        
        if max_events:
            precursor_events = precursor_events[:max_events]
        
        # Process events
        logger.info(f"Processing {len(precursor_events)} precursor events...")
        
        metadata_list = []
        
        for i, event in enumerate(precursor_events):
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i+1}/{len(precursor_events)}")
            
            metadata = self.process_precursor_event(event)
            
            if metadata:
                metadata_list.append(metadata)
        
        # Save results
        if metadata_list:
            # Save event list (format compatible with event_list.xlsx)
            event_df = pd.DataFrame(metadata_list)
            event_csv_path = os.path.join(self.output_dir, 'metadata', 'local_event_list.csv')
            event_df.to_csv(event_csv_path, index=False)
            
            # Also save as Excel
            event_xlsx_path = os.path.join(self.output_dir, 'metadata', 'local_event_list.xlsx')
            event_df.to_excel(event_xlsx_path, index=False)
            
            logger.info(f"Event list saved to {event_csv_path}")
            logger.info(f"Event list saved to {event_xlsx_path}")
        
        # Print summary
        logger.info("="*80)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files failed: {self.stats['files_failed']}")
        logger.info(f"Precursors found: {self.stats['precursors_found']}")
        logger.info(f"Spectrograms generated: {self.stats['spectrograms_generated']}")
        logger.info(f"Output directory: {self.output_dir}")
        
        return metadata_list


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process local geomagnetic data from mdata2/')
    parser.add_argument('--catalog', default='earthquake_catalog_2018_2025_merged.csv',
                        help='Path to earthquake catalog CSV')
    parser.add_argument('--min-mag', type=float, default=6.0,
                        help='Minimum earthquake magnitude (default: 6.0)')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Maximum events to process (default: all)')
    parser.add_argument('--output', default='dataset_local_mdata',
                        help='Output directory')
    
    args = parser.parse_args()
    
    processor = LocalMdataProcessor(
        mdata_dir='mdata2',
        output_dir=args.output
    )
    
    results = processor.process_all(
        earthquake_catalog_path=args.catalog,
        min_magnitude=args.min_mag,
        max_events=args.max_events
    )
    
    print(f"\nTotal precursor events found: {len(results)}")
    print(f"Output saved to: {args.output}")


if __name__ == '__main__':
    main()
