"""
Geomagnetic Dataset Generator untuk CNN Deep Learning
Generate spectrogram dari data ULF geomagnetic dengan preprocessing lengkap:
- Remote data access dari server
- Denoising (magnetic storms & local noise)
- Bandpass filter PC3 (0.01-0.045 Hz)
- Spectrogram generation (STFT)
- Auto-labeling berdasarkan event_list.xlsx
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import signal
from scipy.signal import butter, filtfilt
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GeomagneticDatasetGenerator:
    """Generate dataset spectrogram untuk klasifikasi CNN"""
    
    def __init__(self, event_file='intial/event_list.xlsx', 
                 station_file='intial/lokasi_stasiun.csv',
                 output_dir='dataset_spectrogram'):
        """
        Initialize generator
        
        Args:
            event_file: Path ke file Excel event gempa
            station_file: Path ke file CSV lokasi stasiun
            output_dir: Directory output untuk dataset
        """
        self.event_file = event_file
        self.station_file = station_file
        self.output_dir = output_dir
        
        # Sampling rate (1 Hz untuk data 1-detik)
        self.fs = 1.0
        
        # PC3 frequency range (10-45 mHz = 0.01-0.045 Hz)
        self.pc3_low = 0.01
        self.pc3_high = 0.045
        
        # Magnetic storm threshold (Kp index)
        self.kp_threshold = 4.0  # Kp >= 4 dianggap storm
        
        # Load event list dan station info
        self.load_event_list()
        self.load_station_info()
        
        # Create output directories
        self.create_output_structure()
        
        logger.info(f"Dataset generator initialized")
        logger.info(f"Total events: {len(self.events)}")
        logger.info(f"Total stations: {len(self.stations)}")
    
    def load_event_list(self):
        """Load daftar event gempa dari Excel"""
        try:
            self.events = pd.read_excel(self.event_file)
            logger.info(f"Loaded {len(self.events)} events from {self.event_file}")
            logger.info(f"Columns: {self.events.columns.tolist()}")
            
            # Pastikan kolom yang diperlukan ada
            required_cols = ['Azm', 'Mag']
            for col in required_cols:
                if col not in self.events.columns:
                    logger.warning(f"Column '{col}' not found in event file")
            
        except Exception as e:
            logger.error(f"Failed to load event file: {e}")
            self.events = pd.DataFrame()
    
    def load_station_info(self):
        """Load informasi stasiun dari CSV"""
        try:
            self.stations = pd.read_csv(self.station_file, sep=';')
            # Clean whitespace
            self.stations.columns = self.stations.columns.str.strip()
            self.stations['Kode Stasiun'] = self.stations['Kode Stasiun'].str.strip()
            
            logger.info(f"Loaded {len(self.stations)} stations")
            logger.info(f"Stations: {self.stations['Kode Stasiun'].tolist()}")
            
        except Exception as e:
            logger.error(f"Failed to load station file: {e}")
            self.stations = pd.DataFrame()

    
    def create_output_structure(self):
        """Buat struktur folder output"""
        # Main output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Subdirectories untuk berbagai output
        self.dirs = {
            'spectrograms': Path(self.output_dir) / 'spectrograms',
            'raw_plots': Path(self.output_dir) / 'raw_plots',
            'filtered_plots': Path(self.output_dir) / 'filtered_plots',
            'metadata': Path(self.output_dir) / 'metadata',
            'logs': Path(self.output_dir) / 'logs'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories untuk setiap kelas Azimuth
        azimuth_classes = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'Unknown']
        self.azimuth_dirs = {}
        for azm_class in azimuth_classes:
            azm_dir = self.dirs['spectrograms'] / 'by_azimuth' / azm_class
            azm_dir.mkdir(parents=True, exist_ok=True)
            self.azimuth_dirs[azm_class] = azm_dir
        
        # Create subdirectories untuk setiap kelas Magnitude
        magnitude_classes = ['Small', 'Moderate', 'Medium', 'Large', 'Major', 'Unknown']
        self.magnitude_dirs = {}
        for mag_class in magnitude_classes:
            mag_dir = self.dirs['spectrograms'] / 'by_magnitude' / mag_class
            mag_dir.mkdir(parents=True, exist_ok=True)
            self.magnitude_dirs[mag_class] = mag_dir
        
        logger.info(f"Output structure created at {self.output_dir}")
        logger.info(f"Created {len(azimuth_classes)} azimuth class folders")
        logger.info(f"Created {len(magnitude_classes)} magnitude class folders")
    
    def fetch_remote_data(self, date, station, server_path='mdata'):
        """
        Fetch data dari server remote
        
        Args:
            date: datetime object atau string 'YYYY-MM-DD'
            station: Kode stasiun (e.g., 'GTO')
            server_path: Path ke data di server
            
        Returns:
            dict dengan komponen H, D, Z atau None jika gagal
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        try:
            # Import read_mdata module
            from intial.read_mdata import read_604rcsv_new_python
            
            # Fetch data
            data = read_604rcsv_new_python(
                date.year, date.month, date.day, 
                station, server_path
            )
            
            logger.info(f"Fetched data for {station} on {date.date()}")
            
            return {
                'H': data['H'],
                'D': data['D'],
                'Z': data['Z'],
                'X': data['X'],
                'Y': data['Y'],
                'Time': data['Time'],
                'station': station,
                'date': date
            }
            
        except FileNotFoundError:
            logger.warning(f"Data file not found for {station} on {date.date()}")
            return None
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def detect_magnetic_storm(self, data, method='threshold'):
        """
        Deteksi badai magnetik
        
        Args:
            data: dict dengan komponen H, D, Z
            method: 'threshold' atau 'kp_index'
            
        Returns:
            bool: True jika ada badai magnetik
        """
        if method == 'threshold':
            # Metode sederhana: deteksi variasi ekstrem
            h_std = np.nanstd(data['H'])
            d_std = np.nanstd(data['D'])
            z_std = np.nanstd(data['Z'])
            
            # Threshold: jika std > 3x median std (heuristik)
            # Untuk data normal, std biasanya < 50 nT
            storm_threshold_h = 150  # nT
            storm_threshold_d = 100  # nT
            storm_threshold_z = 150  # nT
            
            is_storm = (h_std > storm_threshold_h or 
                       d_std > storm_threshold_d or 
                       z_std > storm_threshold_z)
            
            if is_storm:
                logger.warning(f"Magnetic storm detected: H_std={h_std:.1f}, D_std={d_std:.1f}, Z_std={z_std:.1f}")
            
            return is_storm
        
        elif method == 'kp_index':
            # TODO: Implementasi fetch Kp index dari NOAA/GFZ
            # Untuk saat ini return False
            return False
        
        return False
    
    def remove_local_noise(self, data, window_size=60):
        """
        Hapus noise lokal menggunakan median filter
        
        Args:
            data: array 1D
            window_size: ukuran window untuk median filter (detik)
            
        Returns:
            array yang sudah dibersihkan
        """
        # Handle NaN
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return data
        
        # Interpolate NaN
        data_clean = np.array(data, dtype=float)
        if np.any(~valid_mask):
            x = np.arange(len(data))
            data_clean[~valid_mask] = np.interp(
                x[~valid_mask], x[valid_mask], data[valid_mask]
            )
        
        # Apply median filter untuk remove spikes
        from scipy.ndimage import median_filter
        data_denoised = median_filter(data_clean, size=window_size)
        
        return data_denoised
    
    def bandpass_filter_pc3(self, data, order=4):
        """
        Apply bandpass filter pada frekuensi PC3
        
        Args:
            data: array 1D
            order: order filter Butterworth
            
        Returns:
            array yang sudah difilter
        """
        # Design Butterworth bandpass filter
        nyquist = self.fs / 2.0
        low = self.pc3_low / nyquist
        high = self.pc3_high / nyquist
        
        # Ensure valid range
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))
        
        try:
            b, a = butter(order, [low, high], btype='band')
            
            # Apply zero-phase filter
            filtered = filtfilt(b, a, data)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Filter error: {e}")
            return data

    
    def generate_spectrogram(self, data, component_name='H', 
                            nperseg=256, noverlap=None, 
                            save_path=None, show_colorbar=True):
        """
        Generate spectrogram menggunakan STFT
        
        Args:
            data: array 1D (sudah difilter PC3)
            component_name: nama komponen ('H', 'D', 'Z')
            nperseg: panjang segment untuk STFT
            noverlap: overlap antar segment
            save_path: path untuk save gambar
            show_colorbar: tampilkan colorbar atau tidak
            
        Returns:
            dict dengan frequencies, times, Sxx (spectrogram)
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Compute STFT
        frequencies, times, Sxx = signal.spectrogram(
            data, 
            fs=self.fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann',
            scaling='density'
        )
        
        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Plot spectrogram
        if save_path:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot dengan colormap yang bagus untuk CNN
            im = ax.pcolormesh(
                times / 3600,  # Convert to hours
                frequencies * 1000,  # Convert to mHz
                Sxx_db,
                shading='gouraud',
                cmap='viridis'
            )
            
            ax.set_ylabel('Frequency (mHz)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
            ax.set_title(f'Spectrogram - {component_name} Component (PC3 Filtered)', 
                        fontsize=14, fontweight='bold')
            
            # Highlight PC3 range
            ax.axhline(y=self.pc3_low * 1000, color='red', 
                      linestyle='--', linewidth=1.5, alpha=0.7, label='PC3 Range')
            ax.axhline(y=self.pc3_high * 1000, color='red', 
                      linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Set y-axis limit to focus on PC3 range
            ax.set_ylim(0, 100)  # 0-100 mHz
            
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Power Spectral Density (dB)', fontsize=11)
            
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Spectrogram saved to {save_path}")
        
        return {
            'frequencies': frequencies,
            'times': times,
            'Sxx': Sxx,
            'Sxx_db': Sxx_db
        }
    
    def generate_3component_spectrogram(self, h_data, d_data, z_data,
                                       save_path=None, title=None):
        """
        Generate spectrogram 3 komponen dalam 1 gambar untuk CNN
        
        Args:
            h_data: H component (sudah difilter PC3)
            d_data: D component (sudah difilter PC3)
            z_data: Z component (sudah difilter PC3)
            save_path: path untuk save gambar
            title: judul gambar
            
        Returns:
            dict dengan spectrogram data untuk setiap komponen
        """
        nperseg = 256
        noverlap = nperseg // 2
        
        # Compute spectrograms
        f_h, t_h, Sxx_h = signal.spectrogram(h_data, fs=self.fs, 
                                             nperseg=nperseg, noverlap=noverlap)
        f_d, t_d, Sxx_d = signal.spectrogram(d_data, fs=self.fs, 
                                             nperseg=nperseg, noverlap=noverlap)
        f_z, t_z, Sxx_z = signal.spectrogram(z_data, fs=self.fs, 
                                             nperseg=nperseg, noverlap=noverlap)
        
        # Convert to dB
        Sxx_h_db = 10 * np.log10(Sxx_h + 1e-10)
        Sxx_d_db = 10 * np.log10(Sxx_d + 1e-10)
        Sxx_z_db = 10 * np.log10(Sxx_z + 1e-10)
        
        if save_path:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            components = [
                (t_h, f_h, Sxx_h_db, 'H Component', axes[0]),
                (t_d, f_d, Sxx_d_db, 'D Component', axes[1]),
                (t_z, f_z, Sxx_z_db, 'Z Component', axes[2])
            ]
            
            for times, freqs, Sxx_db, label, ax in components:
                im = ax.pcolormesh(
                    times / 3600,
                    freqs * 1000,
                    Sxx_db,
                    shading='gouraud',
                    cmap='viridis'
                )
                
                ax.set_ylabel(f'{label}\nFreq (mHz)', fontsize=11, fontweight='bold')
                ax.set_ylim(0, 100)
                
                # Highlight PC3 range
                ax.axhline(y=self.pc3_low * 1000, color='red', 
                          linestyle='--', linewidth=1, alpha=0.5)
                ax.axhline(y=self.pc3_high * 1000, color='red', 
                          linestyle='--', linewidth=1, alpha=0.5)
                
                ax.grid(True, alpha=0.3)
                
                # Colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('PSD (dB)', fontsize=9)
            
            axes[-1].set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
            
            if title:
                fig.suptitle(title, fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"3-component spectrogram saved to {save_path}")
        
        return {
            'H': {'f': f_h, 't': t_h, 'Sxx': Sxx_h, 'Sxx_db': Sxx_h_db},
            'D': {'f': f_d, 't': t_d, 'Sxx': Sxx_d, 'Sxx_db': Sxx_d_db},
            'Z': {'f': f_z, 't': t_z, 'Sxx': Sxx_z, 'Sxx_db': Sxx_z_db}
        }

    
    def get_event_label(self, date, station):
        """
        Dapatkan label event berdasarkan tanggal dan stasiun
        
        Args:
            date: datetime object
            station: kode stasiun
            
        Returns:
            dict dengan azimuth dan magnitude, atau None
        """
        # Filter events untuk tanggal yang sesuai
        # Asumsi: event_list.xlsx punya kolom tanggal
        # Untuk demo, kita return None dulu
        # TODO: Implementasi matching dengan event_list.xlsx
        
        return None
    
    def classify_by_azimuth(self, azimuth):
        """
        Klasifikasi berdasarkan azimuth
        
        Args:
            azimuth: azimuth dalam derajat (0-360)
            
        Returns:
            string: kategori arah ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
        """
        if azimuth is None or np.isnan(azimuth):
            return 'Unknown'
        
        # 8 arah mata angin
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
        elif 292.5 <= azimuth < 337.5:
            return 'NW'
        else:
            return 'Unknown'
    
    def classify_by_magnitude(self, magnitude):
        """
        Klasifikasi berdasarkan magnitude
        
        Args:
            magnitude: magnitude gempa
            
        Returns:
            string: kategori magnitude
        """
        if magnitude is None or np.isnan(magnitude):
            return 'Unknown'
        
        if magnitude < 4.0:
            return 'Small'  # M < 4
        elif 4.0 <= magnitude < 5.0:
            return 'Moderate'  # 4 <= M < 5
        elif 5.0 <= magnitude < 6.0:
            return 'Medium'  # 5 <= M < 6
        elif 6.0 <= magnitude < 7.0:
            return 'Large'  # 6 <= M < 7
        else:
            return 'Major'  # M >= 7
    
    def process_single_event(self, date, station, event_info=None, 
                            server_path='mdata'):
        """
        Process single event: fetch, denoise, filter, generate spectrogram
        
        Args:
            date: datetime atau string 'YYYY-MM-DD'
            station: kode stasiun
            event_info: dict dengan 'Azm' dan 'Mag' (optional)
            server_path: path ke data server
            
        Returns:
            dict dengan hasil processing dan metadata
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        date_str = date.strftime('%Y-%m-%d')
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {station} - {date_str}")
        logger.info(f"{'='*60}")
        
        # 1. Fetch remote data
        data = self.fetch_remote_data(date, station, server_path)
        if data is None:
            logger.error(f"Failed to fetch data for {station} on {date_str}")
            return None
        
        # 2. Check for magnetic storm
        is_storm = self.detect_magnetic_storm(data)
        if is_storm:
            logger.warning(f"Magnetic storm detected - data may be contaminated")
            # Bisa skip atau tetap process dengan flag
        
        # 3. Remove local noise
        h_denoised = self.remove_local_noise(data['H'])
        d_denoised = self.remove_local_noise(data['D'])
        z_denoised = self.remove_local_noise(data['Z'])
        
        # 4. Apply PC3 bandpass filter
        h_pc3 = self.bandpass_filter_pc3(h_denoised)
        d_pc3 = self.bandpass_filter_pc3(d_denoised)
        z_pc3 = self.bandpass_filter_pc3(z_denoised)
        
        # 5. Determine classification labels
        azimuth_class = 'Unknown'
        magnitude_class = 'Unknown'
        
        if event_info:
            azimuth_class = self.classify_by_azimuth(event_info.get('Azm'))
            magnitude_class = self.classify_by_magnitude(event_info.get('Mag'))
            logger.info(f"Labels: Azimuth={azimuth_class}, Magnitude={magnitude_class}")
        
        # 6. Generate spectrograms dan simpan ke folder yang sesuai
        filename_base = f"{station}_{date.strftime('%Y%m%d')}"
        
        # Save ke folder utama (all data)
        spec_path_main = self.dirs['spectrograms'] / f"{filename_base}_3comp_spec.png"
        
        # Save ke folder azimuth class
        spec_path_azm = self.azimuth_dirs[azimuth_class] / f"{filename_base}_3comp_spec.png"
        
        # Save ke folder magnitude class
        spec_path_mag = self.magnitude_dirs[magnitude_class] / f"{filename_base}_3comp_spec.png"
        
        # Individual spectrograms (di folder utama)
        spec_h = self.generate_spectrogram(
            h_pc3, 'H',
            save_path=self.dirs['spectrograms'] / f"{filename_base}_H_spec.png"
        )
        
        spec_d = self.generate_spectrogram(
            d_pc3, 'D',
            save_path=self.dirs['spectrograms'] / f"{filename_base}_D_spec.png"
        )
        
        spec_z = self.generate_spectrogram(
            z_pc3, 'Z',
            save_path=self.dirs['spectrograms'] / f"{filename_base}_Z_spec.png"
        )
        
        # 3-component spectrogram (untuk CNN) - simpan di 3 lokasi
        title = f"{station} - {date_str}"
        if event_info:
            title += f" | Azm: {event_info.get('Azm', 'N/A')}° ({azimuth_class}) | Mag: {event_info.get('Mag', 'N/A')} ({magnitude_class})"
        
        # Generate dan save ke folder utama
        spec_3comp = self.generate_3component_spectrogram(
            h_pc3, d_pc3, z_pc3,
            save_path=spec_path_main,
            title=title
        )
        
        # Copy ke folder azimuth class
        import shutil
        shutil.copy2(spec_path_main, spec_path_azm)
        logger.info(f"Copied to azimuth folder: {azimuth_class}/")
        
        # Copy ke folder magnitude class
        shutil.copy2(spec_path_main, spec_path_mag)
        logger.info(f"Copied to magnitude folder: {magnitude_class}/")
        
        # 7. Prepare metadata
        metadata = {
            'station': station,
            'date': date_str,
            'is_magnetic_storm': is_storm,
            'h_mean': float(np.nanmean(data['H'])),
            'h_std': float(np.nanstd(data['H'])),
            'd_mean': float(np.nanmean(data['D'])),
            'd_std': float(np.nanstd(data['D'])),
            'z_mean': float(np.nanmean(data['Z'])),
            'z_std': float(np.nanstd(data['Z'])),
            'h_pc3_std': float(np.std(h_pc3)),
            'd_pc3_std': float(np.std(d_pc3)),
            'z_pc3_std': float(np.std(z_pc3)),
            'spectrogram_files': {
                'H': f"{filename_base}_H_spec.png",
                'D': f"{filename_base}_D_spec.png",
                'Z': f"{filename_base}_Z_spec.png",
                '3comp': f"{filename_base}_3comp_spec.png"
            },
            'spectrogram_paths': {
                'main': str(spec_path_main),
                'azimuth_folder': str(spec_path_azm),
                'magnitude_folder': str(spec_path_mag)
            }
        }
        
        # Add event info if available
        if event_info:
            metadata['azimuth'] = event_info.get('Azm')
            metadata['magnitude'] = event_info.get('Mag')
            metadata['azimuth_class'] = azimuth_class
            metadata['magnitude_class'] = magnitude_class
        else:
            metadata['azimuth'] = None
            metadata['magnitude'] = None
            metadata['azimuth_class'] = 'Unknown'
            metadata['magnitude_class'] = 'Unknown'
        
        logger.info(f"Processing completed for {station} - {date_str}")
        
        return metadata

    
    def generate_dataset_from_events(self, server_path='mdata', 
                                    max_events=None):
        """
        Generate dataset dari semua events di event_list.xlsx
        
        Args:
            server_path: path ke data server
            max_events: maksimal jumlah events yang diprocess (None = semua)
            
        Returns:
            DataFrame dengan metadata semua events
        """
        if self.events.empty:
            logger.error("No events loaded from event file")
            return None
        
        all_metadata = []
        
        # Limit events jika diperlukan
        events_to_process = self.events.head(max_events) if max_events else self.events
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"Starting dataset generation for {len(events_to_process)} events")
        logger.info(f"{'#'*60}\n")
        
        for idx, event in events_to_process.iterrows():
            try:
                # Extract event info
                # Kolom di event_list.xlsx: Stasiun, Tanggal, Azm, Mag
                
                event_info = {
                    'Azm': event.get('Azm'),
                    'Mag': event.get('Mag')
                }
                
                # Get date and station from event_list
                if 'Tanggal' in event and 'Stasiun' in event:
                    date = event['Tanggal']
                    station = event['Stasiun']
                    
                    # Skip if station is NaN
                    if pd.isna(station):
                        logger.warning(f"Event {idx}: Station is NaN, skipping")
                        continue
                else:
                    logger.warning(f"Event {idx}: No Tanggal/Stasiun columns, skipping")
                    continue
                
                # Process event
                metadata = self.process_single_event(
                    date, station, event_info, server_path
                )
                
                if metadata:
                    all_metadata.append(metadata)
                
            except Exception as e:
                logger.error(f"Error processing event {idx}: {e}")
                continue
        
        # Save metadata to CSV
        if all_metadata:
            metadata_df = pd.DataFrame(all_metadata)
            metadata_file = self.dirs['metadata'] / 'dataset_metadata.csv'
            metadata_df.to_csv(metadata_file, index=False)
            logger.info(f"\nMetadata saved to {metadata_file}")
            logger.info(f"Total processed: {len(metadata_df)} events")
            
            # Generate summary statistics
            self.generate_dataset_summary(metadata_df)
            
            return metadata_df
        else:
            logger.warning("No events were successfully processed")
            return None
    
    def generate_dataset_summary(self, metadata_df):
        """
        Generate summary statistics dari dataset
        
        Args:
            metadata_df: DataFrame dengan metadata
        """
        summary_file = self.dirs['metadata'] / 'dataset_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("GEOMAGNETIC DATASET SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total Events: {len(metadata_df)}\n")
            f.write(f"Date Range: {metadata_df['date'].min()} to {metadata_df['date'].max()}\n")
            f.write(f"Stations: {metadata_df['station'].unique().tolist()}\n\n")
            
            # Magnetic storm statistics
            if 'is_magnetic_storm' in metadata_df.columns:
                n_storms = metadata_df['is_magnetic_storm'].sum()
                f.write(f"Magnetic Storms Detected: {n_storms} ({n_storms/len(metadata_df)*100:.1f}%)\n\n")
            
            # Azimuth distribution
            if 'azimuth_class' in metadata_df.columns:
                f.write("Azimuth Distribution:\n")
                azm_counts = metadata_df['azimuth_class'].value_counts()
                for azm, count in azm_counts.items():
                    f.write(f"  {azm}: {count} ({count/len(metadata_df)*100:.1f}%)\n")
                f.write("\n")
            
            # Magnitude distribution
            if 'magnitude_class' in metadata_df.columns:
                f.write("Magnitude Distribution:\n")
                mag_counts = metadata_df['magnitude_class'].value_counts()
                for mag, count in mag_counts.items():
                    f.write(f"  {mag}: {count} ({count/len(metadata_df)*100:.1f}%)\n")
                f.write("\n")
            
            # Signal statistics
            f.write("Signal Statistics (PC3 Filtered):\n")
            f.write(f"  H Component Std: {metadata_df['h_pc3_std'].mean():.3f} ± {metadata_df['h_pc3_std'].std():.3f} nT\n")
            f.write(f"  D Component Std: {metadata_df['d_pc3_std'].mean():.3f} ± {metadata_df['d_pc3_std'].std():.3f} nT\n")
            f.write(f"  Z Component Std: {metadata_df['z_pc3_std'].mean():.3f} ± {metadata_df['z_pc3_std'].std():.3f} nT\n")
        
        logger.info(f"Dataset summary saved to {summary_file}")


def main():
    """Main function untuk demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Geomagnetic Dataset untuk CNN')
    parser.add_argument('--event-file', default='intial/event_list.xlsx',
                       help='Path ke file event list')
    parser.add_argument('--station-file', default='intial/lokasi_stasiun.csv',
                       help='Path ke file lokasi stasiun')
    parser.add_argument('--server-path', default='mdata',
                       help='Path ke data server')
    parser.add_argument('--output-dir', default='dataset_spectrogram',
                       help='Output directory')
    parser.add_argument('--max-events', type=int, default=None,
                       help='Maksimal jumlah events (None = semua)')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo dengan single event')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = GeomagneticDatasetGenerator(
        event_file=args.event_file,
        station_file=args.station_file,
        output_dir=args.output_dir
    )
    
    if args.demo:
        # Demo: process single event
        logger.info("Running DEMO mode - processing single event")
        
        # Example: process data untuk stasiun GTO tanggal 2024-01-15
        demo_date = '2024-01-15'
        demo_station = 'GTO'
        demo_event_info = {'Azm': 135.5, 'Mag': 5.2}
        
        metadata = generator.process_single_event(
            demo_date, demo_station, demo_event_info, args.server_path
        )
        
        if metadata:
            logger.info("\nDemo completed successfully!")
            logger.info(f"Metadata: {metadata}")
        else:
            logger.error("Demo failed")
    
    else:
        # Generate full dataset
        metadata_df = generator.generate_dataset_from_events(
            server_path=args.server_path,
            max_events=args.max_events
        )
        
        if metadata_df is not None:
            logger.info("\n" + "="*60)
            logger.info("DATASET GENERATION COMPLETED!")
            logger.info("="*60)
            logger.info(f"Total events processed: {len(metadata_df)}")
            logger.info(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
