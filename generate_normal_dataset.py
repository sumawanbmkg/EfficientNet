#!/usr/bin/env python3
"""
Normal Data Generator - Generate spectrograms for quiet/normal days
Menggunakan data dari quiet_days.csv untuk membuat dataset "Normal" (tanpa gempa)

Fitur:
- Fetch data dari SSH server (sama seperti v22)
- Dual format support (.STN dan .gz)
- Label: "Normal" untuk magnitude dan azimuth
- Output format: 224×224 RGB, no axis/labels
- Auto-merge dengan dataset_unified
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
import logging
from pathlib import Path
import shutil

# Add intial to path
sys.path.insert(0, 'intial')
from geomagnetic_fetcher import GeomagneticDataFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NormalDataGenerator:
    """Generate dataset spectrogram untuk hari-hari tenang (normal/tanpa gempa)"""
    
    def __init__(self, output_dir='dataset_normal', prefer_compressed=True):
        """
        Initialize generator
        
        Args:
            output_dir: Directory untuk output files
            prefer_compressed: True = prioritas .gz files (faster)
        """
        self.output_dir = output_dir
        self.prefer_compressed = prefer_compressed
        
        # PC3 frequency range (10-45 mHz = 0.01-0.045 Hz)
        self.pc3_low = 0.01
        self.pc3_high = 0.045
        self.sampling_rate = 1.0  # 1 Hz
        
        # CNN input size (standard)
        self.image_size = 224  # 224x224 pixels
        
        # Statistics tracking
        self.session_stats = {
            'total_events_attempted': 0,
            'total_events_processed': 0,
            'total_events_skipped': 0,
            'total_events_failed': 0,
            'format_usage': {},
            'total_bytes_downloaded': 0
        }
        
        # Create output directories
        self._create_directories()
        
        # Load existing metadata untuk skip checking
        self.existing_metadata = self._load_existing_metadata()
        
        logger.info(f"🌟 NormalDataGenerator initialized")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"🎯 PC3 range: {self.pc3_low*1000:.1f}-{self.pc3_high*1000:.1f} mHz")
        logger.info(f"🖼️  Image size: {self.image_size}x{self.image_size} pixels")
        logger.info(f"🔄 Dual format support: ENABLED")
        logger.info(f"📦 Format preference: {'Compressed (.gz)' if prefer_compressed else 'Uncompressed (.STN)'}")
        
        if len(self.existing_metadata) > 0:
            logger.info(f"✅ Found {len(self.existing_metadata)} existing processed events")
            logger.info(f"⏭️  These events will be SKIPPED!")
        else:
            logger.info(f"🆕 No existing metadata - will process all events")

    
    def _load_existing_metadata(self):
        """Load existing metadata untuk skip checking"""
        metadata_path = Path(self.output_dir) / 'metadata' / 'dataset_metadata.csv'
        if metadata_path.exists():
            try:
                df = pd.read_csv(metadata_path)
                # Create set of processed events (station_date_hour)
                processed = set()
                for _, row in df.iterrows():
                    key = f"{row['station']}_{row['date']}_{int(row['hour'])}"
                    processed.add(key)
                return processed
            except Exception as e:
                logger.warning(f"Could not load existing metadata: {e}")
                return set()
        return set()
    
    
    def _create_directories(self):
        """Create output directories"""
        dirs = [
            self.output_dir,
            f'{self.output_dir}/spectrograms',
            f'{self.output_dir}/metadata',
            f'{self.output_dir}/logs'
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    
    def _pc3_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        """Apply PC3 bandpass filter"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    
    def _denoise_signal(self, data, window_size=60):
        """Simple denoising dengan moving average"""
        if len(data) < window_size:
            return data
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='same')
    
    
    def _generate_spectrogram(self, h_data, d_data, z_data, station, date, hour, output_path):
        """
        Generate 3-component spectrogram (CNN format: 224x224, no axis/labels)
        
        Args:
            h_data, d_data, z_data: Component data
            station: Station code
            date: Date string
            hour: Hour
            output_path: Output file path
        """
        try:
            # Apply PC3 filter
            h_filtered = self._pc3_bandpass_filter(h_data, self.pc3_low, self.pc3_high, self.sampling_rate)
            d_filtered = self._pc3_bandpass_filter(d_data, self.pc3_low, self.pc3_high, self.sampling_rate)
            z_filtered = self._pc3_bandpass_filter(z_data, self.pc3_low, self.pc3_high, self.sampling_rate)
            
            # Denoise
            h_denoised = self._denoise_signal(h_filtered)
            d_denoised = self._denoise_signal(d_filtered)
            z_denoised = self._denoise_signal(z_filtered)
            
            # Create figure dengan 3 subplots (CNN format: no axis, no labels)
            fig, axes = plt.subplots(3, 1, figsize=(self.image_size/100, self.image_size/100), dpi=100)
            
            # STFT parameters
            nperseg = 256
            noverlap = 128
            
            # Generate spectrograms untuk setiap komponen
            for idx, (data, ax, title) in enumerate([
                (h_denoised, axes[0], 'H'),
                (d_denoised, axes[1], 'D'),
                (z_denoised, axes[2], 'Z')
            ]):
                f, t, Sxx = spectrogram(data, fs=self.sampling_rate, 
                                       nperseg=nperseg, noverlap=noverlap)
                
                # Plot spectrogram (NO axis, NO labels, NO text)
                ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                             shading='gouraud', cmap='viridis')
                ax.set_ylim([self.pc3_low, self.pc3_high])
                
                # Remove ALL axis elements
                ax.axis('off')
            
            # Remove spacing between subplots
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
            
            # Save dengan format CNN (224x224 RGB, no axis)
            plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Verify image size dan resize jika perlu
            from PIL import Image
            img = Image.open(output_path)
            if img.size != (self.image_size, self.image_size):
                img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                img.save(output_path)
            
            # Convert to RGB jika perlu
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img.save(output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating spectrogram: {e}")
            return False
    
    
    def process_quiet_days(self, quiet_days_csv='quiet_days.csv', max_events=None):
        """
        Process quiet days dari CSV file
        
        Args:
            quiet_days_csv: Path ke file quiet_days.csv
            max_events: Maximum number of events to process (None = all)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🌟 STARTING NORMAL DATA GENERATION")
        logger.info(f"{'='*80}\n")
        
        # Load quiet days data
        try:
            df = pd.read_csv(quiet_days_csv)
            logger.info(f"📊 Loaded {len(df)} quiet day records from {quiet_days_csv}")
        except Exception as e:
            logger.error(f"❌ Failed to load {quiet_days_csv}: {e}")
            return
        
        # Validate columns
        required_cols = ['kode_stasiun', 'tanggal', 'jam']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"❌ CSV must have columns: {required_cols}")
            return
        
        # Initialize fetcher
        fetcher = GeomagneticDataFetcher(prefer_compressed=self.prefer_compressed)
        
        # Connect to SSH server
        try:
            fetcher.connect()
            logger.info(f"✅ Connected to SSH server")
        except Exception as e:
            logger.error(f"❌ Failed to connect to SSH server: {e}")
            return
        
        # Metadata list
        metadata_list = []
        
        # Process each record
        total_records = len(df) if max_events is None else min(len(df), max_events)
        
        for idx, row in df.iterrows():
            if max_events and idx >= max_events:
                break
            
            self.session_stats['total_events_attempted'] += 1
            
            station = row['kode_stasiun']
            date_str = row['tanggal']  # Format: YYYY-MM-DD
            hour = int(row['jam'])
            
            # Check if already processed
            event_key = f"{station}_{date_str}_{hour}"
            if event_key in self.existing_metadata:
                logger.info(f"⏭️  [{idx+1}/{total_records}] SKIP: {station} {date_str} H{hour:02d} (already processed)")
                self.session_stats['total_events_skipped'] += 1
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"📍 [{idx+1}/{total_records}] Processing: {station} {date_str} H{hour:02d}")
            logger.info(f"{'='*80}")
            
            try:
                # Parse date
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                year = date_obj.year
                month = date_obj.month
                day = date_obj.day
                
                # Fetch data (full day)
                logger.info(f"📥 Fetching data from SSH server...")
                data = fetcher.fetch_data(date_obj, station)
                
                if data is None:
                    logger.warning(f"⚠️  No data available for {station} {date_str}")
                    self.session_stats['total_events_failed'] += 1
                    continue
                
                # Extract hourly data (3600 samples per hour)
                start_idx = hour * 3600
                end_idx = start_idx + 3600
                
                # Extract components for the specific hour
                h_full = np.array(data['Hcomp'])
                d_full = np.array(data['Dcomp'])
                z_full = np.array(data['Zcomp'])
                
                # Check if we have enough data
                if len(h_full) < end_idx or len(d_full) < end_idx or len(z_full) < end_idx:
                    logger.warning(f"⚠️  Insufficient data for hour {hour}: need {end_idx} samples, have {len(h_full)}")
                    self.session_stats['total_events_failed'] += 1
                    continue
                
                h_data = h_full[start_idx:end_idx]
                d_data = d_full[start_idx:end_idx]
                z_data = z_full[start_idx:end_idx]
                
                # Validate data
                if len(h_data) < 3600 or len(d_data) < 3600 or len(z_data) < 3600:
                    logger.warning(f"⚠️  Insufficient data: H={len(h_data)}, D={len(d_data)}, Z={len(z_data)}")
                    self.session_stats['total_events_failed'] += 1
                    continue
                
                # Generate spectrogram filename
                spec_filename = f"{station}_{date_str.replace('-', '')}_H{hour:02d}_normal_3comp_spec.png"
                spec_path = os.path.join(self.output_dir, 'spectrograms', spec_filename)
                
                # Generate spectrogram
                logger.info(f"🎨 Generating spectrogram...")
                success = self._generate_spectrogram(
                    h_data, d_data, z_data,
                    station, date_str, hour,
                    spec_path
                )
                
                if not success:
                    logger.warning(f"⚠️  Failed to generate spectrogram")
                    self.session_stats['total_events_failed'] += 1
                    continue
                
                # Calculate statistics
                h_mean = np.mean(h_data)
                h_std = np.std(h_data)
                d_mean = np.mean(d_data)
                d_std = np.std(d_data)
                z_mean = np.mean(z_data)
                z_std = np.std(z_data)
                
                # PC3 filtered statistics
                h_pc3 = self._pc3_bandpass_filter(h_data, self.pc3_low, self.pc3_high, self.sampling_rate)
                d_pc3 = self._pc3_bandpass_filter(d_data, self.pc3_low, self.pc3_high, self.sampling_rate)
                z_pc3 = self._pc3_bandpass_filter(z_data, self.pc3_low, self.pc3_high, self.sampling_rate)
                
                h_pc3_std = np.std(h_pc3)
                d_pc3_std = np.std(d_pc3)
                z_pc3_std = np.std(z_pc3)
                
                # Add to metadata
                metadata_list.append({
                    'station': station,
                    'date': date_str,
                    'hour': hour,
                    'azimuth': 'Normal',  # Label untuk data normal
                    'magnitude': 'Normal',  # Label untuk data normal
                    'azimuth_class': 'Normal',
                    'magnitude_class': 'Normal',
                    'h_mean': h_mean,
                    'h_std': h_std,
                    'd_mean': d_mean,
                    'd_std': d_std,
                    'z_mean': z_mean,
                    'z_std': z_std,
                    'h_pc3_std': h_pc3_std,
                    'd_pc3_std': d_pc3_std,
                    'z_pc3_std': z_pc3_std,
                    'samples': len(h_data),
                    'spectrogram_file': spec_filename,
                    'image_size': f'{self.image_size}x{self.image_size}',
                    'data_source': 'SSH Server (Normal Days)',
                    'dataset_source': 'quiet_days.csv'
                })
                
                # Update statistics
                self.session_stats['total_events_processed'] += 1
                
                # Track format usage
                if hasattr(fetcher, 'session_stats'):
                    for fmt, count in fetcher.session_stats.get('format_usage', {}).items():
                        self.session_stats['format_usage'][fmt] = \
                            self.session_stats['format_usage'].get(fmt, 0) + count
                    self.session_stats['total_bytes_downloaded'] += \
                        fetcher.session_stats.get('total_bytes_downloaded', 0)
                
                logger.info(f"✅ Successfully processed {station} {date_str} H{hour:02d}")
                
                # Save metadata periodically (every 10 events)
                if len(metadata_list) % 10 == 0:
                    self._save_metadata(metadata_list)
                    logger.info(f"💾 Metadata saved ({len(metadata_list)} events)")
                
            except Exception as e:
                logger.error(f"❌ Error processing {station} {date_str} H{hour:02d}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.session_stats['total_events_failed'] += 1
                continue
        
        # Save final metadata
        if metadata_list:
            self._save_metadata(metadata_list)
            logger.info(f"💾 Final metadata saved ({len(metadata_list)} events)")
        
        # Print summary
        self._print_summary()
        
        # Merge dengan dataset_unified
        self._merge_with_unified()
    
    
    def _save_metadata(self, metadata_list):
        """Save metadata to CSV"""
        if not metadata_list:
            return
        
        df = pd.DataFrame(metadata_list)
        metadata_path = os.path.join(self.output_dir, 'metadata', 'dataset_metadata.csv')
        df.to_csv(metadata_path, index=False)
    
    
    def _print_summary(self):
        """Print processing summary"""
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 PROCESSING SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total events attempted: {self.session_stats['total_events_attempted']}")
        logger.info(f"✅ Successfully processed: {self.session_stats['total_events_processed']}")
        logger.info(f"⏭️  Skipped (already exist): {self.session_stats['total_events_skipped']}")
        logger.info(f"❌ Failed: {self.session_stats['total_events_failed']}")
        
        if self.session_stats['total_events_attempted'] > 0:
            success_rate = (self.session_stats['total_events_processed'] / 
                          self.session_stats['total_events_attempted'] * 100)
            logger.info(f"📈 Success rate: {success_rate:.1f}%")
        
        # Format usage
        if self.session_stats['format_usage']:
            logger.info(f"\n📦 Format Usage:")
            for fmt, count in self.session_stats['format_usage'].items():
                logger.info(f"  {fmt}: {count} files")
        
        # Bandwidth
        if self.session_stats['total_bytes_downloaded'] > 0:
            mb_downloaded = self.session_stats['total_bytes_downloaded'] / (1024 * 1024)
            logger.info(f"\n📊 Bandwidth:")
            logger.info(f"  Total downloaded: {mb_downloaded:.2f} MB")
        
        logger.info(f"{'='*80}\n")
    
    
    def _merge_with_unified(self):
        """Merge normal data dengan dataset_unified"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 MERGING WITH UNIFIED DATASET")
        logger.info(f"{'='*80}")
        
        try:
            # Load normal metadata
            normal_metadata_path = Path(self.output_dir) / 'metadata' / 'dataset_metadata.csv'
            if not normal_metadata_path.exists():
                logger.warning(f"⚠️  No normal metadata found to merge")
                return
            
            normal_df = pd.read_csv(normal_metadata_path)
            logger.info(f"📊 Loaded {len(normal_df)} normal records")
            
            # Load unified metadata
            unified_metadata_path = Path('dataset_unified') / 'metadata' / 'unified_metadata.csv'
            if unified_metadata_path.exists():
                unified_df = pd.read_csv(unified_metadata_path)
                logger.info(f"📊 Loaded {len(unified_df)} existing unified records")
            else:
                unified_df = pd.DataFrame()
                logger.info(f"🆕 Creating new unified dataset")
            
            # Copy spectrograms to unified
            unified_spec_dir = Path('dataset_unified') / 'spectrograms' / 'normal'
            unified_spec_dir.mkdir(parents=True, exist_ok=True)
            
            copied_count = 0
            for _, row in normal_df.iterrows():
                src_path = Path(self.output_dir) / 'spectrograms' / row['spectrogram_file']
                dst_path = unified_spec_dir / row['spectrogram_file']
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
            
            logger.info(f"📁 Copied {copied_count} spectrograms to unified/spectrograms/normal/")
            
            # Add unified_path column
            normal_df['unified_path'] = normal_df['spectrogram_file'].apply(
                lambda x: f'spectrograms/normal/{x}'
            )
            normal_df['image_type'] = 'normal'
            normal_df['augmentation_type'] = ''
            normal_df['augmentation_id'] = ''
            
            # Merge dataframes
            merged_df = pd.concat([unified_df, normal_df], ignore_index=True)
            
            # Remove duplicates based on spectrogram_file
            merged_df = merged_df.drop_duplicates(subset=['spectrogram_file'], keep='last')
            
            # Save merged metadata
            merged_df.to_csv(unified_metadata_path, index=False)
            logger.info(f"💾 Saved merged metadata: {len(merged_df)} total records")
            logger.info(f"   - Previous: {len(unified_df)} records")
            logger.info(f"   - Added: {len(normal_df)} normal records")
            logger.info(f"   - Total: {len(merged_df)} records")
            
            # Update summary
            summary_path = Path('dataset_unified') / 'metadata' / 'dataset_summary.txt'
            with open(summary_path, 'w') as f:
                f.write(f"Dataset Unified Summary\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"Total Records: {len(merged_df)}\n")
                f.write(f"Normal Records: {len(normal_df)}\n")
                f.write(f"Earthquake Records: {len(unified_df)}\n\n")
                
                # Class distribution
                f.write(f"Magnitude Class Distribution:\n")
                mag_dist = merged_df['magnitude_class'].value_counts()
                for cls, count in mag_dist.items():
                    f.write(f"  {cls}: {count}\n")
                
                f.write(f"\nAzimuth Class Distribution:\n")
                az_dist = merged_df['azimuth_class'].value_counts()
                for cls, count in az_dist.items():
                    f.write(f"  {cls}: {count}\n")
            
            logger.info(f"✅ Merge completed successfully!")
            logger.info(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"❌ Error merging with unified dataset: {e}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate normal/quiet day dataset')
    parser.add_argument('--input', type=str, default='quiet_days.csv',
                       help='Input CSV file with quiet days (default: quiet_days.csv)')
    parser.add_argument('--output', type=str, default='dataset_normal',
                       help='Output directory (default: dataset_normal)')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maximum number of events to process (default: all)')
    parser.add_argument('--prefer-stn', action='store_true',
                       help='Prefer .STN files over .gz (default: prefer .gz)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = NormalDataGenerator(
        output_dir=args.output,
        prefer_compressed=not args.prefer_stn
    )
    
    # Process quiet days
    generator.process_quiet_days(
        quiet_days_csv=args.input,
        max_events=args.max_events
    )


if __name__ == '__main__':
    main()
