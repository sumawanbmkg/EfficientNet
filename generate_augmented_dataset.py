"""
Generate Augmented Dataset dengan Physics-Preserving Augmentation
Langsung dari data asli (SSH server)

Fitur:
- Fetch data dari SSH server
- Binary reading yang benar (32-byte header, 17-byte records, baseline)
- Spektrogram 224×224 RGB (no axis, no text)
- Physics-preserving augmentation (SpecAugment style)
- Multiple augmented versions per sample
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
from PIL import Image

# Add intial to path
sys.path.insert(0, 'intial')
from geomagnetic_fetcher import GeomagneticDataFetcher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhysicsPreservingAugmentor:
    """
    Physics-preserving augmentation untuk spektrogram ULF
    Mengikuti prinsip SpecAugment (Park et al., 2019)
    """
    
    def __init__(self, 
                 time_mask_param=30,
                 freq_mask_param=15,
                 noise_std=0.05,
                 time_shift_max=10,
                 gain_range=0.15):
        """
        Initialize augmentor
        
        Args:
            time_mask_param: Max width untuk time masking (pixels)
            freq_mask_param: Max height untuk frequency masking (pixels)
            noise_std: Standard deviation untuk Gaussian noise
            time_shift_max: Max shift untuk time shifting (samples)
            gain_range: Range untuk gain scaling (±)
        """
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.gain_range = gain_range
        
        logger.info("PhysicsPreservingAugmentor initialized")
        logger.info(f"  Time mask: {time_mask_param} pixels")
        logger.info(f"  Freq mask: {freq_mask_param} pixels")
        logger.info(f"  Noise std: {noise_std}")
        logger.info(f"  Time shift: ±{time_shift_max} samples")
        logger.info(f"  Gain range: ±{gain_range*100}%")
    
    def add_gaussian_noise(self, data):
        """Tambah Gaussian noise (simulasi noise instrumen)"""
        noise = np.random.randn(*data.shape) * self.noise_std * np.std(data)
        return data + noise
    
    def time_shift(self, data):
        """Shift data dalam waktu (simulasi variasi timing)"""
        shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
        return np.roll(data, shift)
    
    def gain_scaling(self, data):
        """Scale gain (simulasi variasi energi)"""
        gain = 1.0 + np.random.uniform(-self.gain_range, self.gain_range)
        return data * gain
    
    def time_masking_spectrogram(self, Sxx):
        """
        Time masking pada spektrogram (SpecAugment)
        Mask kolom vertikal (time axis)
        """
        Sxx_masked = Sxx.copy()
        num_time_bins = Sxx.shape[1]
        
        # Random mask width
        mask_width = np.random.randint(1, min(self.time_mask_param, num_time_bins // 4))
        mask_start = np.random.randint(0, max(1, num_time_bins - mask_width))
        
        # Apply mask (set to minimum value)
        Sxx_masked[:, mask_start:mask_start + mask_width] = np.min(Sxx)
        
        return Sxx_masked
    
    def frequency_masking_spectrogram(self, Sxx):
        """
        Frequency masking pada spektrogram (SpecAugment)
        Mask baris horizontal (frequency axis)
        """
        Sxx_masked = Sxx.copy()
        num_freq_bins = Sxx.shape[0]
        
        # Random mask height
        mask_height = np.random.randint(1, min(self.freq_mask_param, num_freq_bins // 4))
        mask_start = np.random.randint(0, max(1, num_freq_bins - mask_height))
        
        # Apply mask (set to minimum value)
        Sxx_masked[mask_start:mask_start + mask_height, :] = np.min(Sxx)
        
        return Sxx_masked
    
    def augment_time_series(self, h_data, d_data, z_data, num_augmentations=3):
        """
        Generate multiple augmented versions dari time series
        
        Returns:
            List of (h_aug, d_aug, z_aug) tuples
        """
        augmented_list = []
        
        for i in range(num_augmentations):
            # Apply augmentation ke time series
            h_aug = h_data.copy()
            d_aug = d_data.copy()
            z_aug = z_data.copy()
            
            # 1. Gaussian noise (30% probability)
            if np.random.rand() < 0.3:
                h_aug = self.add_gaussian_noise(h_aug)
                d_aug = self.add_gaussian_noise(d_aug)
                z_aug = self.add_gaussian_noise(z_aug)
            
            # 2. Gain scaling (30% probability)
            if np.random.rand() < 0.3:
                h_aug = self.gain_scaling(h_aug)
                d_aug = self.gain_scaling(d_aug)
                z_aug = self.gain_scaling(z_aug)
            
            # 3. Time shift (30% probability)
            if np.random.rand() < 0.3:
                shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
                h_aug = np.roll(h_aug, shift)
                d_aug = np.roll(d_aug, shift)
                z_aug = np.roll(z_aug, shift)
            
            augmented_list.append((h_aug, d_aug, z_aug))
        
        return augmented_list
    
    def augment_spectrogram(self, Sxx_h, Sxx_d, Sxx_z, num_augmentations=3):
        """
        Generate multiple augmented versions dari spektrogram
        
        Returns:
            List of (Sxx_h_aug, Sxx_d_aug, Sxx_z_aug) tuples
        """
        augmented_list = []
        
        for i in range(num_augmentations):
            # Apply augmentation ke spektrogram
            Sxx_h_aug = Sxx_h.copy()
            Sxx_d_aug = Sxx_d.copy()
            Sxx_z_aug = Sxx_z.copy()
            
            # 1. Time masking (50% probability)
            if np.random.rand() < 0.5:
                Sxx_h_aug = self.time_masking_spectrogram(Sxx_h_aug)
                Sxx_d_aug = self.time_masking_spectrogram(Sxx_d_aug)
                Sxx_z_aug = self.time_masking_spectrogram(Sxx_z_aug)
            
            # 2. Frequency masking (50% probability)
            if np.random.rand() < 0.5:
                Sxx_h_aug = self.frequency_masking_spectrogram(Sxx_h_aug)
                Sxx_d_aug = self.frequency_masking_spectrogram(Sxx_d_aug)
                Sxx_z_aug = self.frequency_masking_spectrogram(Sxx_z_aug)
            
            augmented_list.append((Sxx_h_aug, Sxx_d_aug, Sxx_z_aug))
        
        return augmented_list


class AugmentedDatasetGenerator:
    """Generate dataset dengan augmentasi dari SSH server"""
    
    def __init__(self, output_dir='dataset_augmented', 
                 num_augmentations_per_sample=3,
                 augmentation_level='moderate'):
        """
        Initialize generator
        
        Args:
            output_dir: Output directory
            num_augmentations_per_sample: Jumlah augmented versions per sample
            augmentation_level: 'light', 'moderate', 'aggressive'
        """
        self.output_dir = output_dir
        self.num_augmentations = num_augmentations_per_sample
        
        # PC3 frequency range
        self.pc3_low = 0.01
        self.pc3_high = 0.045
        self.sampling_rate = 1.0
        self.image_size = 224
        
        # Setup augmentor
        if augmentation_level == 'light':
            aug_params = {
                'time_mask_param': 20,
                'freq_mask_param': 10,
                'noise_std': 0.03,
                'time_shift_max': 5,
                'gain_range': 0.1
            }
        elif augmentation_level == 'moderate':
            aug_params = {
                'time_mask_param': 30,
                'freq_mask_param': 15,
                'noise_std': 0.05,
                'time_shift_max': 10,
                'gain_range': 0.15
            }
        else:  # aggressive
            aug_params = {
                'time_mask_param': 40,
                'freq_mask_param': 20,
                'noise_std': 0.08,
                'time_shift_max': 15,
                'gain_range': 0.2
            }
        
        self.augmentor = PhysicsPreservingAugmentor(**aug_params)
        
        # Create directories
        self._create_directories()
        
        logger.info(f"AugmentedDatasetGenerator initialized")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Augmentations per sample: {num_augmentations_per_sample}")
        logger.info(f"Augmentation level: {augmentation_level}")
    
    def _create_directories(self):
        """Create output directories"""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/spectrograms",
            f"{self.output_dir}/spectrograms/by_azimuth",
            f"{self.output_dir}/spectrograms/by_magnitude",
            f"{self.output_dir}/metadata",
            f"{self.output_dir}/logs"
        ]
        
        # Azimuth classes
        for az_class in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
            dirs.append(f"{self.output_dir}/spectrograms/by_azimuth/{az_class}")
        
        # Magnitude classes
        for mag_class in ['Small', 'Moderate', 'Medium', 'Large', 'Major']:
            dirs.append(f"{self.output_dir}/spectrograms/by_magnitude/{mag_class}")
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def fetch_hour_data(self, fetcher, year, month, day, hour, station):
        """Fetch 1 hour data dari SSH server"""
        try:
            date = datetime(year, month, day)
            data = fetcher.fetch_data(date, station)
            
            if data is None:
                return None
            
            # Extract 1 hour
            start_idx = hour * 3600
            end_idx = start_idx + 3600
            
            h_full = data['Hcomp']
            d_full = data['Dcomp']
            z_full = data['Zcomp']
            
            if end_idx > len(h_full):
                end_idx = min(start_idx + 3600, len(h_full))
            
            if start_idx >= len(h_full):
                return None
            
            hour_data = {
                'H': h_full[start_idx:end_idx],
                'D': d_full[start_idx:end_idx],
                'Z': z_full[start_idx:end_idx]
            }
            
            return hour_data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def apply_pc3_filter(self, data):
        """Apply PC3 bandpass filter"""
        if len(data) < 100:
            return data
        
        data_clean = np.nan_to_num(data, nan=np.nanmean(data))
        
        nyquist = self.sampling_rate / 2
        low = max(0.001, min(self.pc3_low / nyquist, 0.999))
        high = max(0.001, min(self.pc3_high / nyquist, 0.999))
        
        if low >= high:
            return data_clean
        
        try:
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, data_clean)
            return filtered
        except:
            return data_clean
    
    def generate_spectrogram(self, data):
        """Generate spectrogram"""
        nperseg = min(256, len(data) // 4)
        noverlap = nperseg // 2
        
        f, t, Sxx = spectrogram(
            data, fs=self.sampling_rate,
            window='hann', nperseg=nperseg,
            noverlap=noverlap, scaling='density'
        )
        
        return f, t, Sxx
    
    def save_spectrogram_cnn(self, f, t, Sxx_h, Sxx_d, Sxx_z, filename):
        """
        Save spectrogram 224×224 RGB (NO axis, NO text)
        """
        fig_size = self.image_size / 100.0
        fig, axes = plt.subplots(3, 1, figsize=(fig_size, fig_size))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        
        # Limit to PC3 range
        freq_mask = (f >= self.pc3_low) & (f <= self.pc3_high)
        f_pc3 = f[freq_mask]
        Sxx_h_pc3 = Sxx_h[freq_mask, :]
        Sxx_d_pc3 = Sxx_d[freq_mask, :]
        Sxx_z_pc3 = Sxx_z[freq_mask, :]
        
        # Convert to dB
        Sxx_h_db = 10 * np.log10(Sxx_h_pc3 + 1e-10)
        Sxx_d_db = 10 * np.log10(Sxx_d_pc3 + 1e-10)
        Sxx_z_db = 10 * np.log10(Sxx_z_pc3 + 1e-10)
        
        # Plot (NO axis, NO text)
        axes[0].pcolormesh(t, f_pc3, Sxx_h_db, shading='gouraud', cmap='jet')
        axes[0].axis('off')
        
        axes[1].pcolormesh(t, f_pc3, Sxx_d_db, shading='gouraud', cmap='jet')
        axes[1].axis('off')
        
        axes[2].pcolormesh(t, f_pc3, Sxx_z_db, shading='gouraud', cmap='jet')
        axes[2].axis('off')
        
        # Save
        filepath = os.path.join(self.output_dir, 'spectrograms', filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Resize to exactly 224×224 dan convert to RGB
        img = Image.open(filepath)
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(filepath)
        
        return filepath
    
    def classify_azimuth(self, azimuth):
        """Classify azimuth"""
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
        """Classify magnitude"""
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
    
    def _copy_to_class_folders(self, image_path, azimuth_class, magnitude_class):
        """Copy to classification folders"""
        import shutil
        filename = os.path.basename(image_path)
        
        # Azimuth folder
        az_dest = os.path.join(
            self.output_dir, 'spectrograms', 'by_azimuth',
            azimuth_class, filename
        )
        shutil.copy2(image_path, az_dest)
        
        # Magnitude folder
        mag_dest = os.path.join(
            self.output_dir, 'spectrograms', 'by_magnitude',
            magnitude_class, filename
        )
        shutil.copy2(image_path, mag_dest)
    
    def process_event(self, fetcher, event_row):
        """
        Process event dengan augmentasi
        Generate: 1 original + N augmented versions
        """
        try:
            station = event_row['Stasiun']
            date = pd.to_datetime(event_row['Tanggal'])
            hour = int(event_row['Jam'])
            azimuth = float(event_row['Azm'])
            magnitude = float(event_row['Mag'])
            
            logger.info("="*80)
            logger.info(f"Processing: {station} - {date.strftime('%Y-%m-%d')} Hour {hour:02d}")
            
            # Fetch data
            data = self.fetch_hour_data(fetcher, date.year, date.month, date.day, hour, station)
            if data is None or len(data['H']) < 100:
                logger.error("Insufficient data")
                return []
            
            # Apply PC3 filter
            h_pc3 = self.apply_pc3_filter(data['H'])
            d_pc3 = self.apply_pc3_filter(data['D'])
            z_pc3 = self.apply_pc3_filter(data['Z'])
            
            # Classify
            azimuth_class = self.classify_azimuth(azimuth)
            magnitude_class = self.classify_magnitude(magnitude)
            
            metadata_list = []
            date_str = date.strftime('%Y%m%d')
            
            # 1. ORIGINAL (tanpa augmentasi)
            logger.info("  [1/{}] Generating ORIGINAL...".format(self.num_augmentations + 1))
            
            f_h, t_h, Sxx_h = self.generate_spectrogram(h_pc3)
            f_d, t_d, Sxx_d = self.generate_spectrogram(d_pc3)
            f_z, t_z, Sxx_z = self.generate_spectrogram(z_pc3)
            
            filename_orig = f"{station}_{date_str}_H{hour:02d}_orig_3comp_spec.png"
            image_path = self.save_spectrogram_cnn(f_h, t_h, Sxx_h, Sxx_d, Sxx_z, filename_orig)
            self._copy_to_class_folders(image_path, azimuth_class, magnitude_class)
            
            metadata_list.append({
                'station': station,
                'date': date.strftime('%Y-%m-%d'),
                'hour': hour,
                'azimuth': azimuth,
                'magnitude': magnitude,
                'azimuth_class': azimuth_class,
                'magnitude_class': magnitude_class,
                'spectrogram_file': filename_orig,
                'augmentation_type': 'original',
                'augmentation_id': 0
            })
            
            # 2. AUGMENTED VERSIONS
            # Augment time series
            augmented_ts_list = self.augmentor.augment_time_series(
                h_pc3, d_pc3, z_pc3, 
                num_augmentations=self.num_augmentations
            )
            
            for aug_id, (h_aug, d_aug, z_aug) in enumerate(augmented_ts_list, 1):
                logger.info(f"  [{aug_id+1}/{self.num_augmentations + 1}] Generating AUGMENTED #{aug_id}...")
                
                # Generate spectrograms
                f_h, t_h, Sxx_h = self.generate_spectrogram(h_aug)
                f_d, t_d, Sxx_d = self.generate_spectrogram(d_aug)
                f_z, t_z, Sxx_z = self.generate_spectrogram(z_aug)
                
                # Augment spectrograms (SpecAugment)
                aug_spec_list = self.augmentor.augment_spectrogram(
                    Sxx_h, Sxx_d, Sxx_z, num_augmentations=1
                )
                Sxx_h_aug, Sxx_d_aug, Sxx_z_aug = aug_spec_list[0]
                
                # Save
                filename_aug = f"{station}_{date_str}_H{hour:02d}_aug{aug_id}_3comp_spec.png"
                image_path = self.save_spectrogram_cnn(
                    f_h, t_h, Sxx_h_aug, Sxx_d_aug, Sxx_z_aug, filename_aug
                )
                self._copy_to_class_folders(image_path, azimuth_class, magnitude_class)
                
                metadata_list.append({
                    'station': station,
                    'date': date.strftime('%Y-%m-%d'),
                    'hour': hour,
                    'azimuth': azimuth,
                    'magnitude': magnitude,
                    'azimuth_class': azimuth_class,
                    'magnitude_class': magnitude_class,
                    'spectrogram_file': filename_aug,
                    'augmentation_type': 'augmented',
                    'augmentation_id': aug_id
                })
            
            logger.info(f"[OK] Generated {len(metadata_list)} images (1 orig + {self.num_augmentations} aug)")
            return metadata_list
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_event_list(self, event_file, max_events=None):
        """Process all events dengan augmentasi"""
        df = pd.read_excel(event_file)
        total_events = len(df) if max_events is None else min(max_events, len(df))
        
        logger.info(f"Total events to process: {total_events}")
        logger.info(f"Will generate {self.num_augmentations + 1} images per event")
        logger.info(f"Expected total images: {total_events * (self.num_augmentations + 1)}")
        
        metadata_list = []
        success_count = 0
        failure_count = 0
        
        with GeomagneticDataFetcher() as fetcher:
            logger.info("[OK] SSH Connection established!")
            
            for idx, row in df.iterrows():
                if max_events and idx >= max_events:
                    break
                
                event_metadata = self.process_event(fetcher, row)
                
                if event_metadata:
                    metadata_list.extend(event_metadata)
                    success_count += 1
                else:
                    failure_count += 1
        
        # Save metadata
        if metadata_list:
            metadata_df = pd.DataFrame(metadata_list)
            metadata_path = os.path.join(self.output_dir, 'metadata', 'dataset_metadata.csv')
            metadata_df.to_csv(metadata_path, index=False)
            
            # Summary
            summary_path = os.path.join(self.output_dir, 'metadata', 'dataset_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("AUGMENTED DATASET SUMMARY\n")
                f.write("="*60 + "\n\n")
                f.write(f"Total Events Processed: {success_count}\n")
                f.write(f"Failed: {failure_count}\n")
                f.write(f"Total Images Generated: {len(metadata_list)}\n")
                f.write(f"  - Original: {len(metadata_df[metadata_df['augmentation_type']=='original'])}\n")
                f.write(f"  - Augmented: {len(metadata_df[metadata_df['augmentation_type']=='augmented'])}\n\n")
                
                f.write("Magnitude Distribution:\n")
                for mag_class, count in metadata_df['magnitude_class'].value_counts().items():
                    f.write(f"  {mag_class}: {count}\n")
                
                f.write("\nAzimuth Distribution:\n")
                for az_class, count in metadata_df['azimuth_class'].value_counts().items():
                    f.write(f"  {az_class}: {count}\n")
        
        logger.info("="*80)
        logger.info("DATASET GENERATION COMPLETED!")
        logger.info("="*80)
        logger.info(f"Events processed: {success_count}")
        logger.info(f"Failed: {failure_count}")
        logger.info(f"Total images: {len(metadata_list)}")
        logger.info(f"Output: {self.output_dir}")
        
        return metadata_df


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate augmented dataset dengan physics-preserving augmentation'
    )
    parser.add_argument('--event-file', default='intial/event_list.xlsx',
                       help='Event list Excel file')
    parser.add_argument('--output-dir', default='dataset_augmented',
                       help='Output directory')
    parser.add_argument('--num-augmentations', type=int, default=3,
                       help='Number of augmented versions per sample')
    parser.add_argument('--augmentation-level', default='moderate',
                       choices=['light', 'moderate', 'aggressive'],
                       help='Augmentation level')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Max events to process (None = all)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = AugmentedDatasetGenerator(
        output_dir=args.output_dir,
        num_augmentations_per_sample=args.num_augmentations,
        augmentation_level=args.augmentation_level
    )
    
    # Process
    metadata_df = generator.process_event_list(
        event_file=args.event_file,
        max_events=args.max_events
    )
    
    print(f"\n[OK] Dataset generation completed!")
    print(f"Total images: {len(metadata_df)}")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
