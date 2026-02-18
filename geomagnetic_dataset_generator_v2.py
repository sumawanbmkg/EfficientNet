"""
Geomagnetic Dataset Generator v2.0
Generate spectrogram dataset dengan fetch data langsung dari server BMKG
Proses hanya 1 jam data sesuai waktu kejadian gempa
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, spectrogram
from datetime import datetime, timedelta
import logging
from pathlib import Path
import gzip
import struct
import urllib.request
import urllib.error

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RemoteGeomagneticFetcher:
    """Fetch geomagnetic data dari server BMKG"""
    
    def __init__(self):
        """Initialize fetcher dengan URL server BMKG"""
        # URL server BMKG untuk data geomagnetic
        self.base_url = "http://202.90.198.102/magnet/data"
        logger.info("RemoteGeomagneticFetcher initialized")
    
    def fetch_hour_data(self, year, month, day, hour, station):
        """
        Fetch data 1 jam dari server
        
        Args:
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
            # Format: S{YY}{MM}{DD}.{STATION}.gz
            date_str = f"{year:04d}{month:02d}{day:02d}"
            yy = year % 100
            filename = f"S{yy:02d}{month:02d}{day:02d}.{station}.gz"
            
            # Construct URL
            url = f"{self.base_url}/{year}/{month:02d}/{station}/{filename}"
            
            logger.info(f"Fetching from: {url}")
            
            # Download file
            response = urllib.request.urlopen(url, timeout=30)
            compressed_data = response.read()
            
            # Decompress
            decompressed_data = gzip.decompress(compressed_data)
            
            # Parse binary data (FRG604RC format)
            data = self._parse_frg604rc(decompressed_data)
            
            # Extract 1 hour data
            start_idx = hour * 3600
            end_idx = start_idx + 3600
            
            if end_idx > len(data['H']):
                logger.warning(f"Not enough data for hour {hour}, using available data")
                end_idx = len(data['H'])
            
            hour_data = {
                'H': data['H'][start_idx:end_idx],
                'D': data['D'][start_idx:end_idx],
                'Z': data['Z'][start_idx:end_idx],
                'Time': np.arange(3600) / 3600.0  # 0-1 hour
            }
            
            logger.info(f"Successfully fetched {len(hour_data['H'])} samples for hour {hour}")
            return hour_data
            
        except urllib.error.URLError as e:
            logger.error(f"URL error fetching data: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def _parse_frg604rc(self, data):
        """
        Parse FRG604RC binary format
        
        Format: 1-second data, 86400 samples per day
        Each sample: H, D, Z (float32)
        """
        # FRG604RC format: 12 bytes per sample (3 x float32)
        n_samples = len(data) // 12
        
        H = np.zeros(n_samples)
        D = np.zeros(n_samples)
        Z = np.zeros(n_samples)
        
        for i in range(n_samples):
            offset = i * 12
            H[i] = struct.unpack('<f', data[offset:offset+4])[0]
            D[i] = struct.unpack('<f', data[offset+4:offset+8])[0]
            Z[i] = struct.unpack('<f', data[offset+8:offset+12])[0]
        
        return {'H': H, 'D': D, 'Z': Z}


class GeomagneticDatasetGeneratorV2:
    """Generate dataset spectrogram dari data geomagnetic (1 jam data)"""
    
    def __init__(self, output_dir='dataset_spectrogram_v2'):
        """
        Initialize generator
        
        Args:
            output_dir: Directory untuk output files
        """
        self.output_dir = output_dir
        self.fetcher = RemoteGeomagneticFetcher()
        
        # PC3 frequency range (10-45 mHz = 0.01-0.045 Hz)
        self.pc3_low = 0.01
        self.pc3_high = 0.045
        self.sampling_rate = 1.0  # 1 Hz
        
        # Create output directories
        self._create_directories()
        
        logger.info(f"GeomagneticDatasetGeneratorV2 initialized")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"PC3 range: {self.pc3_low*1000:.1f}-{self.pc3_high*1000:.1f} mHz")
    
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
    
    def apply_pc3_filter(self, data):
        """
        Apply PC3 bandpass filter (10-45 mHz)
        
        Args:
            data: 1D array
            
        Returns:
            Filtered data
        """
        # Butterworth bandpass filter
        nyquist = self.sampling_rate / 2
        low = self.pc3_low / nyquist
        high = self.pc3_high / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, data)
        
        return filtered
    
    def generate_spectrogram(self, data, component_name):
        """
        Generate spectrogram dari time series data
        
        Args:
            data: 1D array (3600 samples untuk 1 jam)
            component_name: Nama komponen (H, D, Z)
            
        Returns:
            tuple (frequencies, times, Sxx)
        """
        # STFT parameters
        nperseg = 256
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
        """
        Classify azimuth ke 8 kelas
        
        Args:
            azimuth: Azimuth dalam derajat (0-360)
            
        Returns:
            str: Kelas azimuth (N, NE, E, SE, S, SW, W, NW)
        """
        # Normalize azimuth
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
        else:  # 292.5 <= azimuth < 337.5
            return 'NW'
    
    def classify_magnitude(self, magnitude):
        """
        Classify magnitude ke 5 kelas
        
        Args:
            magnitude: Magnitude gempa
            
        Returns:
            str: Kelas magnitude
        """
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
        """
        Save spectrogram sebagai image
        
        Args:
            f: Frequencies
            t: Times
            Sxx_h, Sxx_d, Sxx_z: Spectrograms untuk H, D, Z
            station: Station code
            date: Date string
            hour: Hour (0-23)
            component: Component name
            
        Returns:
            Path ke saved image
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        fig.patch.set_facecolor('black')
        
        date_str = date.replace('-', '')
        
        # H component
        im1 = axes[0].pcolormesh(t, f, 10 * np.log10(Sxx_h + 1e-10), 
                                 shading='gouraud', cmap='jet')
        axes[0].set_ylabel('Freq (Hz)', color='white')
        axes[0].set_title(f'{station} - {date} Hour {hour:02d} - PC3 Filtered', 
                         color='white', fontsize=12)
        axes[0].set_ylim([self.pc3_low, self.pc3_high])
        axes[0].set_facecolor('black')
        axes[0].tick_params(colors='white')
        plt.colorbar(im1, ax=axes[0], label='Power (dB)')
        
        # D component
        im2 = axes[1].pcolormesh(t, f, 10 * np.log10(Sxx_d + 1e-10),
                                 shading='gouraud', cmap='jet')
        axes[1].set_ylabel('Freq (Hz)', color='white')
        axes[1].set_ylim([self.pc3_low, self.pc3_high])
        axes[1].set_facecolor('black')
        axes[1].tick_params(colors='white')
        plt.colorbar(im2, ax=axes[1], label='Power (dB)')
        
        # Z component
        im3 = axes[2].pcolormesh(t, f, 10 * np.log10(Sxx_z + 1e-10),
                                 shading='gouraud', cmap='jet')
        axes[2].set_ylabel('Freq (Hz)', color='white')
        axes[2].set_xlabel('Time (s)', color='white')
        axes[2].set_ylim([self.pc3_low, self.pc3_high])
        axes[2].set_facecolor('black')
        axes[2].tick_params(colors='white')
        plt.colorbar(im3, ax=axes[2], label='Power (dB)')
        
        plt.tight_layout()
        
        # Save
        filename = f"{station}_{date_str}_H{hour:02d}_{component}_spec.png"
        filepath = os.path.join(self.output_dir, 'spectrograms', filename)
        plt.savefig(filepath, dpi=100, facecolor='black')
        plt.close()
        
        return filepath
    
    def process_event(self, event_row):
        """
        Process single event
        
        Args:
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
            
            logger.info("="*80)
            logger.info(f"Processing: {station} - {date.strftime('%Y-%m-%d')} Hour {hour:02d}")
            logger.info("="*80)
            
            # Fetch 1 hour data dari server
            data = self.fetcher.fetch_hour_data(
                date.year, date.month, date.day, hour, station
            )
            
            if data is None:
                logger.error(f"Failed to fetch data for {station} on {date.strftime('%Y-%m-%d')} hour {hour}")
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
                'h_mean': float(np.mean(data['H'])),
                'h_std': float(np.std(data['H'])),
                'd_mean': float(np.mean(data['D'])),
                'd_std': float(np.std(data['D'])),
                'z_mean': float(np.mean(data['Z'])),
                'z_std': float(np.std(data['Z'])),
                'h_pc3_std': float(np.std(h_pc3)),
                'd_pc3_std': float(np.std(d_pc3)),
                'z_pc3_std': float(np.std(z_pc3)),
                'spectrogram_file': os.path.basename(image_path)
            }
            
            logger.info(f"✅ Successfully processed: {station} - {date_str} Hour {hour:02d}")
            logger.info(f"   Azimuth: {azimuth}° → {azimuth_class}")
            logger.info(f"   Magnitude: {magnitude} → {magnitude_class}")
            
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
        Process semua events dari Excel file
        
        Args:
            event_file: Path ke event_list.xlsx
            max_events: Maximum number of events to process (None = all)
            
        Returns:
            DataFrame dengan metadata
        """
        # Read event list
        logger.info(f"Reading event list from: {event_file}")
        df = pd.read_excel(event_file)
        
        total_events = len(df) if max_events is None else min(max_events, len(df))
        logger.info(f"Total events to process: {total_events}")
        
        # Process events
        metadata_list = []
        success_count = 0
        failure_count = 0
        
        for idx, row in df.iterrows():
            if max_events and idx >= max_events:
                break
            
            metadata = self.process_event(row)
            
            if metadata:
                metadata_list.append(metadata)
                success_count += 1
            else:
                failure_count += 1
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata_list)
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, 'metadata', 'dataset_metadata.csv')
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"Metadata saved to {metadata_path}")
        
        # Generate summary
        self._generate_summary(metadata_df, success_count, failure_count)
        
        logger.info("="*80)
        logger.info("DATASET GENERATION COMPLETED!")
        logger.info("="*80)
        logger.info(f"Total events processed: {success_count}")
        logger.info(f"Total events failed: {failure_count}")
        logger.info(f"Success rate: {success_count/(success_count+failure_count)*100:.1f}%")
        logger.info(f"Output directory: {self.output_dir}")
        
        return metadata_df
    
    def _generate_summary(self, metadata_df, success_count, failure_count):
        """Generate summary report"""
        summary_path = os.path.join(self.output_dir, 'metadata', 'dataset_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("GEOMAGNETIC DATASET SUMMARY (1 Hour Data)\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total Events Processed: {success_count}\n")
            f.write(f"Total Events Failed: {failure_count}\n")
            f.write(f"Success Rate: {success_count/(success_count+failure_count)*100:.1f}%\n\n")
            
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


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate geomagnetic spectrogram dataset (1 hour data)')
    parser.add_argument('--event-file', default='intial/event_list.xlsx',
                       help='Path to event list Excel file')
    parser.add_argument('--output-dir', default='dataset_spectrogram_v2',
                       help='Output directory')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maximum number of events to process')
    
    args = parser.parse_args()
    
    # Create generator
    generator = GeomagneticDatasetGeneratorV2(output_dir=args.output_dir)
    
    # Process events
    metadata_df = generator.process_event_list(
        event_file=args.event_file,
        max_events=args.max_events
    )
    
    print(f"\n✅ Dataset generation completed!")
    print(f"   Total samples: {len(metadata_df)}")
    print(f"   Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
