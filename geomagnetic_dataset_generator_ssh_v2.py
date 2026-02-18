"""
Geomagnetic Dataset Generator - SSH Version 2.0
FITUR BARU: Skip proses jika data sudah berhasil di-generate sebelumnya

Generate spectrogram dataset dengan fetch data langsung dari server via SSH
Menggunakan intial/geomagnetic_fetcher.py untuk akses server
Proses 1 jam data sesuai waktu kejadian gempa

VERSION 2.0 FEATURES:
- ✅ Skip events yang sudah berhasil diproses (cek dari metadata CSV)
- ✅ Resume processing dari event terakhir
- ✅ Tidak perlu proses ulang dari awal
- ✅ Hemat waktu dan bandwidth
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

# Add intial to path
sys.path.insert(0, 'intial')
from geomagnetic_fetcher import GeomagneticDataFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeomagneticDatasetGeneratorSSH_V2:
    """Generate dataset spectrogram dengan fetch dari server SSH (1 jam data) - Version 2.0"""
    
    def __init__(self, output_dir='dataset_spectrogram_ssh'):
        """
        Initialize generator
        
        Args:
            output_dir: Directory untuk output files
        """
        self.output_dir = output_dir
        
        # PC3 frequency range (10-45 mHz = 0.01-0.045 Hz)
        self.pc3_low = 0.01
        self.pc3_high = 0.045
        self.sampling_rate = 1.0  # 1 Hz
        
        # Create output directories
        self._create_directories()
        
        # Load existing metadata untuk skip checking
        self.existing_metadata = self._load_existing_metadata()
        
        logger.info(f"GeomagneticDatasetGeneratorSSH_V2 initialized")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"PC3 range: {self.pc3_low*1000:.1f}-{self.pc3_high*1000:.1f} mHz")
        logger.info(f"Will fetch data from SSH server: 202.90.198.224:4343")
        
        if len(self.existing_metadata) > 0:
            logger.info(f"✅ Found {len(self.existing_metadata)} existing processed events")
            logger.info(f"   These events will be SKIPPED to save time!")
        else:
            logger.info(f"No existing metadata found - will process all events")
    
    def _load_existing_metadata(self):
        """Load existing metadata CSV untuk skip checking"""
        metadata_path = os.path.join(self.output_dir, 'metadata', 'dataset_metadata.csv')
        
        if os.path.exists(metadata_path):
            try:
                df = pd.read_csv(metadata_path)
                logger.info(f"Loaded existing metadata: {len(df)} events")
                return df
            except Exception as e:
                logger.warning(f"Failed to load existing metadata: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    
    def _is_already_processed(self, station, date, hour):
        """
        Check if event sudah diproses sebelumnya
        
        Args:
            station: Kode stasiun
            date: Date object atau string 'YYYY-MM-DD'
            hour: Jam (int)
            
        Returns:
            bool: True jika sudah diproses, False jika belum
        """
        if len(self.existing_metadata) == 0:
            return False
        
        # Convert date to string format
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        elif isinstance(date, pd.Timestamp):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        
        # Check if exists in metadata
        mask = (
            (self.existing_metadata['station'] == station) &
            (self.existing_metadata['date'] == date_str) &
            (self.existing_metadata['hour'] == hour)
        )
        
        exists = mask.any()
        
        if exists:
            logger.info(f"⏭️  SKIP: {station} - {date_str} Hour {hour:02d} (already processed)")
        
        return exists

    
    def _create_directories(self):
        """Create output directory structure"""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/spectrograms",
            f"{self.output_dir}/spectrograms/by_azimuth",
            f"{self.output_dir}/spectrograms/by_magnitude",
            f"{self.output_dir}/metadata",
            f"{self.output_dir}/logs"
        ]
        
        # Azimuth classes
        azimuth_classes = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        for az_class in azimuth_classes:
            dirs.append(f"{self.output_dir}/spectrograms/by_azimuth/{az_class}")
        
        # Magnitude classes
        mag_classes = ['Small', 'Moderate', 'Medium', 'Large', 'Major']
        for mag_class in mag_classes:
            dirs.append(f"{self.output_dir}/spectrograms/by_magnitude/{mag_class}")
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def fetch_hour_data(self, fetcher, year, month, day, hour, station):
        """
        Fetch data 1 jam dari server SSH
        
        Args:
            fetcher: GeomagneticDataFetcher instance (already connected)
            year: Tahun (int)
            month: Bulan (int)
            day: Hari (int)
            hour: Jam (int, 0-23)
            station: Kode stasiun (str)
            
        Returns:
            dict dengan H, D, Z components untuk 1 jam (3600 samples)
            atau None jika gagal
        """
        try:
            # Fetch full day data from server
            date = datetime(year, month, day)
            data = fetcher.fetch_data(date, station)
            
            if data is None:
                logger.error(f"Failed to fetch data from server")
                return None
            
            # Extract 1 hour data
            start_idx = hour * 3600
            end_idx = start_idx + 3600
            
            # Use Hcomp, Dcomp, Zcomp from fetcher
            h_full = data['Hcomp']
            d_full = data['Dcomp']
            z_full = data['Zcomp']
            
            if end_idx > len(h_full):
                logger.warning(f"Not enough data for hour {hour}, using available data")
                end_idx = min(start_idx + 3600, len(h_full))
            
            if start_idx >= len(h_full):
                logger.error(f"Start index {start_idx} exceeds data length {len(h_full)}")
                return None
            
            hour_data = {
                'H': h_full[start_idx:end_idx],
                'D': d_full[start_idx:end_idx],
                'Z': z_full[start_idx:end_idx],
                'Time': np.arange(len(h_full[start_idx:end_idx])) / 3600.0
            }
            
            logger.info(f"Successfully fetched {len(hour_data['H'])} samples for hour {hour}")
            return hour_data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            import traceback
            traceback.print_exc()
            return None

    
    def apply_pc3_filter(self, data):
        """
        Apply PC3 bandpass filter (10-45 mHz)
        
        Args:
            data: 1D array
            
        Returns:
            Filtered data
        """
        if len(data) < 100:
            logger.warning(f"Data too short for filtering: {len(data)} samples")
            return data
        
        # Remove NaN
        data_clean = np.nan_to_num(data, nan=np.nanmean(data))
        
        # Butterworth bandpass filter
        nyquist = self.sampling_rate / 2
        low = self.pc3_low / nyquist
        high = self.pc3_high / nyquist
        
        # Ensure frequencies are in valid range
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        
        if low >= high:
            logger.warning(f"Invalid filter range: {low} >= {high}")
            return data_clean
        
        try:
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, data_clean)
            return filtered
        except Exception as e:
            logger.error(f"Error in filtering: {e}")
            return data_clean
    
    def generate_spectrogram(self, data, component_name):
        """
        Generate spectrogram dari time series data
        
        Args:
            data: 1D array (ideally 3600 samples untuk 1 jam)
            component_name: Nama komponen (H, D, Z)
            
        Returns:
            tuple (frequencies, times, Sxx)
        """
        # STFT parameters
        nperseg = min(256, len(data) // 4)
        noverlap = nperseg // 2
        
        # Generate spectrogram
        f, t, Sxx = spectrogram(
            data,
            fs=self.sampling_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density'
        )
        
        return f, t, Sxx
    
    def classify_azimuth(self, azimuth):
        """Classify azimuth ke 8 kelas"""
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
        """Classify magnitude ke 5 kelas"""
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

    
    def save_spectrogram_image(self, f, t, Sxx_h, Sxx_d, Sxx_z, 
                               station, date, hour, component='3comp'):
        """Save spectrogram sebagai image"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        fig.patch.set_facecolor('black')
        
        date_str = date.replace('-', '')
        
        # H component
        im1 = axes[0].pcolormesh(t, f, 10 * np.log10(Sxx_h + 1e-10), 
                                 shading='gouraud', cmap='jet')
        axes[0].set_ylabel('Freq (Hz)', color='white')
        axes[0].set_title(f'{station} - {date} Hour {hour:02d} - PC3 Filtered (SSH v2.0)', 
                         color='white', fontsize=12)
        axes[0].set_ylim([self.pc3_low, self.pc3_high])
        axes[0].set_facecolor('black')
        axes[0].tick_params(colors='white')
        for spine in axes[0].spines.values():
            spine.set_color('white')
        plt.colorbar(im1, ax=axes[0], label='Power (dB)')
        
        # D component
        im2 = axes[1].pcolormesh(t, f, 10 * np.log10(Sxx_d + 1e-10),
                                 shading='gouraud', cmap='jet')
        axes[1].set_ylabel('Freq (Hz)', color='white')
        axes[1].set_ylim([self.pc3_low, self.pc3_high])
        axes[1].set_facecolor('black')
        axes[1].tick_params(colors='white')
        for spine in axes[1].spines.values():
            spine.set_color('white')
        plt.colorbar(im2, ax=axes[1], label='Power (dB)')
        
        # Z component
        im3 = axes[2].pcolormesh(t, f, 10 * np.log10(Sxx_z + 1e-10),
                                 shading='gouraud', cmap='jet')
        axes[2].set_ylabel('Freq (Hz)', color='white')
        axes[2].set_xlabel('Time (s)', color='white')
        axes[2].set_ylim([self.pc3_low, self.pc3_high])
        axes[2].set_facecolor('black')
        axes[2].tick_params(colors='white')
        for spine in axes[2].spines.values():
            spine.set_color('white')
        plt.colorbar(im3, ax=axes[2], label='Power (dB)')
        
        plt.tight_layout()
        
        # Save
        filename = f"{station}_{date_str}_H{hour:02d}_{component}_spec.png"
        filepath = os.path.join(self.output_dir, 'spectrograms', filename)
        plt.savefig(filepath, dpi=100, facecolor='black')
        plt.close()
        
        return filepath

    
    def process_event(self, fetcher, event_row):
        """
        Process single event
        
        Args:
            fetcher: GeomagneticDataFetcher instance (already connected)
            event_row: pandas Series dengan event info
            
        Returns:
            dict dengan processing results atau None jika gagal
        """
        try:
            station = event_row['Stasiun']
            date = pd.to_datetime(event_row['Tanggal'])
            hour = int(event_row['Jam'])
            azimuth = float(event_row['Azm'])
            magnitude = float(event_row['Mag'])
            
            # ✅ VERSION 2.0: Check if already processed
            if self._is_already_processed(station, date, hour):
                return 'SKIPPED'  # Special return value untuk skip
            
            logger.info("="*80)
            logger.info(f"Processing: {station} - {date.strftime('%Y-%m-%d')} Hour {hour:02d}")
            logger.info("="*80)
            
            # Fetch 1 hour data from SSH server
            data = self.fetch_hour_data(
                fetcher, date.year, date.month, date.day, hour, station
            )
            
            if data is None:
                logger.error(f"Failed to fetch data")
                return None
            
            # Check data length
            if len(data['H']) < 100:
                logger.error(f"Insufficient data: only {len(data['H'])} samples")
                return None
            
            # Apply PC3 filter
            h_pc3 = self.apply_pc3_filter(data['H'])
            d_pc3 = self.apply_pc3_filter(data['D'])
            z_pc3 = self.apply_pc3_filter(data['Z'])
            
            # Generate spectrograms
            f_h, t_h, Sxx_h = self.generate_spectrogram(h_pc3, 'H')
            f_d, t_d, Sxx_d = self.generate_spectrogram(d_pc3, 'D')
            f_z, t_z, Sxx_z = self.generate_spectrogram(z_pc3, 'Z')
            
            # Save spectrogram image
            date_str = date.strftime('%Y-%m-%d')
            image_path = self.save_spectrogram_image(
                f_h, t_h, Sxx_h, Sxx_d, Sxx_z,
                station, date_str, hour, '3comp'
            )
            
            # Classify
            azimuth_class = self.classify_azimuth(azimuth)
            magnitude_class = self.classify_magnitude(magnitude)
            
            # Copy to class folders
            self._copy_to_class_folders(image_path, azimuth_class, magnitude_class)
            
            # Prepare metadata
            metadata = {
                'station': station,
                'date': date_str,
                'hour': hour,
                'azimuth': azimuth,
                'magnitude': magnitude,
                'azimuth_class': azimuth_class,
                'magnitude_class': magnitude_class,
                'h_mean': float(np.nanmean(data['H'])),
                'h_std': float(np.nanstd(data['H'])),
                'd_mean': float(np.nanmean(data['D'])),
                'd_std': float(np.nanstd(data['D'])),
                'z_mean': float(np.nanmean(data['Z'])),
                'z_std': float(np.nanstd(data['Z'])),
                'h_pc3_std': float(np.std(h_pc3)),
                'd_pc3_std': float(np.std(d_pc3)),
                'z_pc3_std': float(np.std(z_pc3)),
                'samples': len(data['H']),
                'spectrogram_file': os.path.basename(image_path),
                'data_source': 'SSH Server v2.0'
            }
            
            logger.info(f"✅ Successfully processed: {station} - {date_str} Hour {hour:02d}")
            logger.info(f"   Azimuth: {azimuth}° → {azimuth_class}")
            logger.info(f"   Magnitude: {magnitude} → {magnitude_class}")
            logger.info(f"   Samples: {len(data['H'])}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing event: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _copy_to_class_folders(self, image_path, azimuth_class, magnitude_class):
        """Copy spectrogram ke folder klasifikasi"""
        import shutil
        
        filename = os.path.basename(image_path)
        
        # Copy ke azimuth folder
        az_dest = os.path.join(
            self.output_dir, 'spectrograms', 'by_azimuth', 
            azimuth_class, filename
        )
        shutil.copy2(image_path, az_dest)
        
        # Copy ke magnitude folder
        mag_dest = os.path.join(
            self.output_dir, 'spectrograms', 'by_magnitude',
            magnitude_class, filename
        )
        shutil.copy2(image_path, mag_dest)

    
    def process_event_list(self, event_file, max_events=None):
        """
        Process semua events dari Excel file dengan SSH connection
        VERSION 2.0: Skip events yang sudah diproses
        
        Args:
            event_file: Path ke event_list.xlsx
            max_events: Maximum number of events to process (None = all)
            
        Returns:
            tuple (metadata_df, success_list, failure_list, skipped_list)
        """
        # Read event list
        logger.info(f"Reading event list from: {event_file}")
        df = pd.read_excel(event_file)
        
        total_events = len(df) if max_events is None else min(max_events, len(df))
        logger.info(f"Total events to process: {total_events}")
        
        # Process events with single SSH connection
        metadata_list = []
        success_list = []
        failure_list = []
        skipped_list = []  # NEW: Track skipped events
        
        logger.info("="*80)
        logger.info("CONNECTING TO SSH SERVER...")
        logger.info("="*80)
        
        with GeomagneticDataFetcher() as fetcher:
            logger.info("✅ SSH Connection established!")
            logger.info("="*80)
            
            for idx, row in df.iterrows():
                if max_events and idx >= max_events:
                    break
                
                metadata = self.process_event(fetcher, row)
                
                event_info = {
                    'No': row['No'],
                    'Station': row['Stasiun'],
                    'Date': pd.to_datetime(row['Tanggal']).strftime('%Y-%m-%d'),
                    'Hour': int(row['Jam']),
                    'Azimuth': row['Azm'],
                    'Magnitude': row['Mag']
                }
                
                if metadata == 'SKIPPED':
                    # Event sudah diproses sebelumnya
                    skipped_list.append(event_info)
                elif metadata:
                    # Event berhasil diproses (baru)
                    metadata_list.append(metadata)
                    success_list.append(event_info)
                else:
                    # Event gagal
                    failure_list.append(event_info)
        
        logger.info("="*80)
        logger.info("SSH CONNECTION CLOSED")
        logger.info("="*80)
        
        # Combine new metadata with existing metadata
        if len(metadata_list) > 0:
            new_metadata_df = pd.DataFrame(metadata_list)
            
            # Append to existing metadata
            if len(self.existing_metadata) > 0:
                combined_metadata = pd.concat([self.existing_metadata, new_metadata_df], ignore_index=True)
            else:
                combined_metadata = new_metadata_df
            
            # Save combined metadata
            metadata_path = os.path.join(self.output_dir, 'metadata', 'dataset_metadata.csv')
            combined_metadata.to_csv(metadata_path, index=False)
            logger.info(f"Metadata saved to {metadata_path}")
            logger.info(f"Total metadata records: {len(combined_metadata)}")
            
            # Generate summary with combined data
            self._generate_summary(combined_metadata, len(success_list), len(failure_list), len(skipped_list))
        else:
            # No new data, but still generate summary
            if len(self.existing_metadata) > 0:
                self._generate_summary(self.existing_metadata, len(success_list), len(failure_list), len(skipped_list))
        
        # Save success/failure/skipped lists
        self._save_reports(success_list, failure_list, skipped_list)
        
        logger.info("="*80)
        logger.info("DATASET GENERATION COMPLETED!")
        logger.info("="*80)
        logger.info(f"✅ Newly processed: {len(success_list)}")
        logger.info(f"⏭️  Skipped (already exists): {len(skipped_list)}")
        logger.info(f"❌ Failed: {len(failure_list)}")
        total_attempted = len(success_list) + len(failure_list) + len(skipped_list)
        if total_attempted > 0:
            logger.info(f"Success rate (new): {len(success_list)/(len(success_list)+len(failure_list))*100:.1f}%")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Return combined metadata if available
        if len(metadata_list) > 0 and len(self.existing_metadata) > 0:
            return combined_metadata, success_list, failure_list, skipped_list
        elif len(metadata_list) > 0:
            return new_metadata_df, success_list, failure_list, skipped_list
        else:
            return self.existing_metadata, success_list, failure_list, skipped_list

    
    def _generate_summary(self, metadata_df, new_success_count, failure_count, skipped_count):
        """Generate summary report"""
        summary_path = os.path.join(self.output_dir, 'metadata', 'dataset_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("GEOMAGNETIC DATASET SUMMARY (SSH Server v2.0 - 1 Hour Data)\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Data Source: SSH Server (202.90.198.224:4343)\n")
            f.write(f"Generator Version: 2.0 (with skip feature)\n\n")
            
            f.write(f"Total Events in Dataset: {len(metadata_df)}\n")
            f.write(f"Newly Processed: {new_success_count}\n")
            f.write(f"Skipped (already exists): {skipped_count}\n")
            f.write(f"Failed: {failure_count}\n")
            
            total_attempted = new_success_count + failure_count + skipped_count
            if total_attempted > 0:
                f.write(f"Success Rate (new only): {new_success_count/(new_success_count+failure_count)*100:.1f}%\n\n")
            
            if len(metadata_df) > 0:
                f.write(f"Date Range: {metadata_df['date'].min()} to {metadata_df['date'].max()}\n")
                f.write(f"Stations: {metadata_df['station'].unique().tolist()}\n\n")
                
                f.write("Azimuth Distribution:\n")
                f.write("-"*40 + "\n")
                az_dist = metadata_df['azimuth_class'].value_counts()
                for az_class, count in az_dist.items():
                    f.write(f"  {az_class}: {count} ({count/len(metadata_df)*100:.1f}%)\n")
                
                f.write("\nMagnitude Distribution:\n")
                f.write("-"*40 + "\n")
                mag_dist = metadata_df['magnitude_class'].value_counts()
                for mag_class, count in mag_dist.items():
                    f.write(f"  {mag_class}: {count} ({count/len(metadata_df)*100:.1f}%)\n")
        
        logger.info(f"Dataset summary saved to {summary_path}")
    
    def _save_reports(self, success_list, failure_list, skipped_list):
        """Save success/failure/skipped reports to Excel"""
        report_path = os.path.join(self.output_dir, 'metadata', 'processing_report.xlsx')
        
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            # Summary
            summary_data = {
                'Metric': [
                    'Data Source',
                    'Generator Version',
                    'Total Events Attempted',
                    'Newly Processed',
                    'Skipped (already exists)',
                    'Failed',
                    'Success Rate (new only) (%)'
                ],
                'Value': [
                    'SSH Server (202.90.198.224:4343)',
                    '2.0 (with skip feature)',
                    len(success_list) + len(failure_list) + len(skipped_list),
                    len(success_list),
                    len(skipped_list),
                    len(failure_list),
                    f"{len(success_list)/(len(success_list)+len(failure_list))*100:.2f}%" if (len(success_list) + len(failure_list)) > 0 else "N/A"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Success list (newly processed)
            if success_list:
                pd.DataFrame(success_list).to_excel(writer, sheet_name='Berhasil (Baru)', index=False)
            
            # Skipped list
            if skipped_list:
                pd.DataFrame(skipped_list).to_excel(writer, sheet_name='Dilewati (Sudah Ada)', index=False)
            
            # Failure list
            if failure_list:
                pd.DataFrame(failure_list).to_excel(writer, sheet_name='Gagal', index=False)
        
        logger.info(f"Processing report saved to {report_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate geomagnetic spectrogram dataset from SSH server (1 hour data) - Version 2.0',
        epilog='VERSION 2.0: Automatically skips events that have been processed before!'
    )
    parser.add_argument('--event-file', default='intial/event_list.xlsx',
                       help='Path to event list Excel file')
    parser.add_argument('--output-dir', default='dataset_spectrogram_ssh',
                       help='Output directory')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maximum number of events to process (None = all 694 events)')
    
    args = parser.parse_args()
    
    # Check if paramiko is installed
    try:
        import paramiko
    except ImportError:
        logger.error("paramiko not installed. Please install: pip install paramiko")
        return
    
    # Create generator
    generator = GeomagneticDatasetGeneratorSSH_V2(output_dir=args.output_dir)
    
    # Process events
    metadata_df, success_list, failure_list, skipped_list = generator.process_event_list(
        event_file=args.event_file,
        max_events=args.max_events
    )
    
    print(f"\n✅ Dataset generation completed!")
    print(f"   Newly processed: {len(success_list)}")
    print(f"   Skipped (already exists): {len(skipped_list)}")
    print(f"   Failed: {len(failure_list)}")
    if len(success_list) + len(failure_list) > 0:
        print(f"   Success rate (new): {len(success_list)/(len(success_list)+len(failure_list))*100:.1f}%")
    print(f"   Total in dataset: {len(metadata_df)}")
    print(f"   Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
