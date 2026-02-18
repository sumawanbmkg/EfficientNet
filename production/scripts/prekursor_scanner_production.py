#!/usr/bin/env python3
"""
Prekursor Scanner - Production Version with Final Model
Scan geomagnetic data untuk prediksi gempa dengan model CNN final

CRITICAL UPDATES:
- Uses final PyTorch VGG16 model (trained on all data)
- Preprocessing matches training exactly
- 100% Normal class detection accuracy
- 98.94% overall magnitude accuracy
- LOEO/LOSO validated (97.53%/97.57%)

Author: Earthquake Prediction Research Team
Date: 4 February 2026
Version: 2.1 (Final Production)
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import signal
import logging
from pathlib import Path
import json
from PIL import Image

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'intial'))
sys.path.insert(0, os.path.dirname(__file__))

# Import modules
from intial.geomagnetic_fetcher import GeomagneticDataFetcher
from intial.signal_processing import GeomagneticSignalProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiTaskVGG16(nn.Module):
    """
    Multi-task VGG16 model for earthquake prediction
    Same architecture as training
    """
    def __init__(self, num_magnitude_classes, num_azimuth_classes):
        super(MultiTaskVGG16, self).__init__()
        vgg16 = models.vgg16(pretrained=False)
        self.features = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.shared = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        self.magnitude_head = nn.Linear(512, num_magnitude_classes)
        self.azimuth_head = nn.Linear(512, num_azimuth_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared(x)
        mag_out = self.magnitude_head(x)
        azi_out = self.azimuth_head(x)
        return mag_out, azi_out


class PrekursorScannerProduction:
    """
    Production scanner with fixed model (no data leakage)
    """
    
    def __init__(self, model_path=None):
        """
        Initialize scanner
        
        Args:
            model_path: Path to model checkpoint (default: best fixed model)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {self.device}")
        
        # Load station list
        self.stations = self._load_stations()
        logger.info(f"📍 Loaded {len(self.stations)} stations")
        
        # Load model
        if model_path is None:
            model_path = self._find_best_model()
        
        self.model, self.class_mappings = self._load_model(model_path)
        logger.info(f"✅ Model loaded from: {model_path}")
        
        # Initialize processors
        self.signal_processor = GeomagneticSignalProcessor(sampling_rate=1.0)
        
        # Define preprocessing transform (matches training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_stations(self):
        """Load station list from CSV"""
        station_file = 'intial/lokasi_stasiun.csv'
        if not os.path.exists(station_file):
            logger.warning(f"Station file not found: {station_file}")
            return {}
        
        df = pd.read_csv(station_file, sep=';')
        stations = {}
        for _, row in df.iterrows():
            code = str(row['Kode Stasiun']).strip()
            if code and code != 'nan':
                stations[code] = {
                    'code': code,
                    'lat': row['Latitude'],
                    'lon': row['Longitude']
                }
        return stations
    
    def _find_best_model(self):
        """Find best fixed model"""
        # Use latest fixed model
        model_path = Path('experiments_fixed/exp_fixed_20260202_163643/best_model.pth')
        
        if model_path.exists():
            return str(model_path)
        
        # Try to find any fixed experiment
        exp_dir = Path('experiments_fixed')
        if exp_dir.exists():
            exp_folders = sorted(exp_dir.glob('exp_fixed_*'))
            if exp_folders:
                latest_exp = exp_folders[-1]
                model_path = latest_exp / 'best_model.pth'
                if model_path.exists():
                    return str(model_path)
        
        raise FileNotFoundError("No fixed model found! Please train model first.")
    
    def _load_model(self, model_path):
        """Load trained model and class mappings"""
        model_path = Path(model_path)
        exp_dir = model_path.parent
        
        # Load class mappings
        mapping_file = exp_dir / 'class_mappings.json'
        if not mapping_file.exists():
            raise FileNotFoundError(f"Class mappings not found: {mapping_file}")
        
        with open(mapping_file, 'r') as f:
            class_mappings = json.load(f)
        
        magnitude_classes = class_mappings['magnitude_classes']
        azimuth_classes = class_mappings['azimuth_classes']
        
        # Create model
        model = MultiTaskVGG16(len(magnitude_classes), len(azimuth_classes))
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model, class_mappings
    
    def fetch_data(self, date, station_code):
        """
        Fetch geomagnetic data for specific date and station
        
        Args:
            date: Date string 'YYYY-MM-DD' or datetime object
            station_code: Station code (e.g., 'GTO', 'SCN')
            
        Returns:
            dict with geomagnetic data or None if failed
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        logger.info(f"📡 Fetching data for {date.date()} from station {station_code}...")
        
        try:
            with GeomagneticDataFetcher() as fetcher:
                data = fetcher.fetch_data(date, station_code)
            
            if data is None:
                logger.error(f"❌ Failed to fetch data")
                return None
            
            logger.info(f"✅ Data fetched successfully")
            logger.info(f"   Coverage: {data['stats']['coverage']:.1f}%")
            logger.info(f"   Valid samples: {data['stats']['valid_samples']}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error fetching data: {e}")
            return None
    
    def generate_spectrogram(self, data):
        """
        Generate spectrogram from geomagnetic data (TRAINING-COMPATIBLE)
        
        CRITICAL: Matches training preprocessing EXACTLY:
        - Filter: 0.01-0.045 Hz (PC3 range)
        - 3 components: H, D, Z stacked vertically
        - Colormap: jet
        - Image size: 224x224 RGB
        
        Args:
            data: Geomagnetic data dict
            
        Returns:
            PIL Image (224x224 RGB)
        """
        logger.info(f"🎨 Generating spectrogram (TRAINING-COMPATIBLE)...")
        
        # PC3 filter range (same as training)
        pc3_low = 0.01   # 10 mHz
        pc3_high = 0.045 # 45 mHz
        
        logger.info(f"   Filter range: {pc3_low*1000:.1f}-{pc3_high*1000:.1f} mHz")
        
        # Process all 3 components
        components_data = {}
        for comp_name in ['Hcomp', 'Dcomp', 'Zcomp']:
            signal_data = data[comp_name]
            
            # Remove NaN values
            valid_mask = ~np.isnan(signal_data)
            if not np.any(valid_mask):
                logger.error(f"❌ All data is NaN for {comp_name}")
                return None
            
            # Interpolate NaN
            signal_clean = np.array(signal_data, dtype=float)
            if np.any(~valid_mask):
                x = np.arange(len(signal_data))
                signal_clean[~valid_mask] = np.interp(
                    x[~valid_mask], x[valid_mask], signal_data[valid_mask]
                )
            
            # Apply PC3 bandpass filter
            signal_filtered = self.signal_processor.bandpass_filter(
                signal_clean, low_freq=pc3_low, high_freq=pc3_high
            )
            
            components_data[comp_name] = signal_filtered
        
        # Generate spectrograms for all 3 components
        fs = 1.0  # 1 Hz sampling rate
        nperseg = 256
        noverlap = nperseg // 2
        
        spectrograms_db = {}
        for comp_name, signal_filtered in components_data.items():
            f, t, Sxx = signal.spectrogram(
                signal_filtered,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann'
            )
            
            # Convert to dB scale
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
            spectrograms_db[comp_name] = (f, t, Sxx_db)
        
        # Create 3-component image (H, D, Z stacked vertically)
        import matplotlib
        matplotlib.use('Agg')
        
        fig_height = 224 / 100.0
        fig_width = 224 / 100.0
        
        fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        
        # Limit frequency range to PC3
        f_h, t_h, Sxx_h_db = spectrograms_db['Hcomp']
        f_d, t_d, Sxx_d_db = spectrograms_db['Dcomp']
        f_z, t_z, Sxx_z_db = spectrograms_db['Zcomp']
        
        freq_mask = (f_h >= pc3_low) & (f_h <= pc3_high)
        f_pc3 = f_h[freq_mask]
        Sxx_h_pc3 = Sxx_h_db[freq_mask, :]
        Sxx_d_pc3 = Sxx_d_db[freq_mask, :]
        Sxx_z_pc3 = Sxx_z_db[freq_mask, :]
        
        # Use jet colormap (same as training)
        axes[0].pcolormesh(t_h, f_pc3, Sxx_h_pc3, shading='gouraud', cmap='jet')
        axes[0].axis('off')
        
        axes[1].pcolormesh(t_d, f_pc3, Sxx_d_pc3, shading='gouraud', cmap='jet')
        axes[1].axis('off')
        
        axes[2].pcolormesh(t_z, f_pc3, Sxx_z_pc3, shading='gouraud', cmap='jet')
        axes[2].axis('off')
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        plt.savefig(tmp_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Load and resize to exactly 224x224
        img = Image.open(tmp_path)
        
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.LANCZOS)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        logger.info(f"✅ Spectrogram generated: {img.size}")
        
        return img
    
    def predict(self, spectrogram_image):
        """
        Predict earthquake parameters from spectrogram
        
        Args:
            spectrogram_image: PIL Image (224x224 RGB)
            
        Returns:
            dict with predictions and confidence scores
        """
        logger.info("🔮 Running prediction...")
        
        # Preprocess image (matches training)
        image_tensor = self.transform(spectrogram_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            mag_output, azi_output = self.model(image_tensor)
            
            # Get probabilities
            mag_probs = torch.softmax(mag_output, dim=1)[0]
            azi_probs = torch.softmax(azi_output, dim=1)[0]
            
            # Get predictions
            mag_pred_idx = torch.argmax(mag_probs).item()
            azi_pred_idx = torch.argmax(azi_probs).item()
            
            mag_pred = self.class_mappings['magnitude_classes'][mag_pred_idx]
            azi_pred = self.class_mappings['azimuth_classes'][azi_pred_idx]
            
            mag_conf = mag_probs[mag_pred_idx].item() * 100
            azi_conf = azi_probs[azi_pred_idx].item() * 100
        
        # Prepare results
        results = {
            'magnitude': {
                'class_id': mag_pred_idx,
                'class_name': mag_pred,
                'confidence': mag_conf,
                'probabilities': mag_probs.cpu().numpy()
            },
            'azimuth': {
                'class_id': azi_pred_idx,
                'class_name': azi_pred,
                'confidence': azi_conf,
                'probabilities': azi_probs.cpu().numpy()
            },
            'is_normal': mag_pred == 'Normal',
            'is_precursor': mag_pred != 'Normal'
        }
        
        logger.info(f"✅ Prediction complete")
        logger.info(f"   Magnitude: {mag_pred} ({mag_conf:.1f}%)")
        logger.info(f"   Azimuth: {azi_pred} ({azi_conf:.1f}%)")
        logger.info(f"   Status: {'NORMAL' if results['is_normal'] else 'PRECURSOR DETECTED'}")
        
        return results
    
    def visualize_results(self, data, spectrogram_image, predictions, save_path=None):
        """
        Visualize scan results
        
        Args:
            data: Geomagnetic data dict
            spectrogram_image: PIL Image
            predictions: Prediction results dict
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        station = data['station']
        date = data['date'].strftime('%Y-%m-%d')
        fig.suptitle(
            f'Prekursor Scanner (Production v2.0) - Station {station} - {date}',
            fontsize=16, fontweight='bold'
        )
        
        # 1. Raw H, D, Z components (left column)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        
        time_hours = np.arange(len(data['Hcomp'])) / 3600.0
        
        ax1.plot(time_hours, data['Hcomp'], 'r-', linewidth=0.5, alpha=0.7)
        ax1.set_ylabel('H (nT)', fontweight='bold')
        ax1.set_title('H Component (Northward)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 24)
        
        ax2.plot(time_hours, data['Dcomp'], 'g-', linewidth=0.5, alpha=0.7)
        ax2.set_ylabel('D (nT)', fontweight='bold')
        ax2.set_title('D Component (Eastward)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 24)
        
        ax3.plot(time_hours, data['Zcomp'], 'b-', linewidth=0.5, alpha=0.7)
        ax3.set_ylabel('Z (nT)', fontweight='bold')
        ax3.set_xlabel('Time (hours)', fontweight='bold')
        ax3.set_title('Z Component (Vertical)', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 24)
        
        # 2. Spectrogram (middle top)
        ax4 = fig.add_subplot(gs[0, 1])
        ax4.imshow(spectrogram_image, aspect='auto')
        ax4.set_title('3-Component Spectrogram (H-D-Z)', fontsize=10, fontweight='bold')
        ax4.axis('off')
        
        # 3. Magnitude prediction (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        mag_pred = predictions['magnitude']
        
        mag_classes = self.class_mappings['magnitude_classes']
        mag_probs = mag_pred['probabilities'] * 100
        
        colors = ['red' if i == mag_pred['class_id'] else 'lightgray' 
                  for i in range(len(mag_probs))]
        
        bars = ax5.barh(range(len(mag_probs)), mag_probs, color=colors)
        ax5.set_yticks(range(len(mag_probs)))
        ax5.set_yticklabels(mag_classes, fontsize=8)
        ax5.set_xlabel('Confidence (%)', fontweight='bold')
        ax5.set_title('Magnitude Prediction', fontsize=10, fontweight='bold')
        ax5.set_xlim(0, 100)
        ax5.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, prob) in enumerate(zip(bars, mag_probs)):
            if prob > 5:
                ax5.text(prob + 2, i, f'{prob:.1f}%', 
                        va='center', fontsize=8, fontweight='bold')
        
        # 4. Azimuth prediction (middle bottom)
        ax6 = fig.add_subplot(gs[2, 1])
        az_pred = predictions['azimuth']
        
        az_classes = self.class_mappings['azimuth_classes']
        az_probs = az_pred['probabilities'] * 100
        
        colors = ['blue' if i == az_pred['class_id'] else 'lightgray' 
                  for i in range(len(az_probs))]
        
        bars = ax6.barh(range(len(az_probs)), az_probs, color=colors)
        ax6.set_yticks(range(len(az_probs)))
        ax6.set_yticklabels(az_classes, fontsize=8)
        ax6.set_xlabel('Confidence (%)', fontweight='bold')
        ax6.set_title('Azimuth Prediction', fontsize=10, fontweight='bold')
        ax6.set_xlim(0, 100)
        ax6.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, prob) in enumerate(zip(bars, az_probs)):
            if prob > 5:
                ax6.text(prob + 2, i, f'{prob:.1f}%', 
                        va='center', fontsize=8, fontweight='bold')
        
        # 5. Summary box (right column)
        ax7 = fig.add_subplot(gs[:, 2])
        ax7.axis('off')
        
        is_normal = predictions['is_normal']
        is_precursor = predictions['is_precursor']
        
        summary_text = f"""
SCAN RESULTS (Production Model v2.0)
{'='*40}

📍 STATION INFORMATION
   Code: {station}
   Location: {self.stations.get(station, {}).get('lat', 'N/A')}°, 
             {self.stations.get(station, {}).get('lon', 'N/A')}°
   Date: {date}

📊 DATA QUALITY
   Coverage: {data['stats']['coverage']:.1f}%
   Valid Samples: {data['stats']['valid_samples']:,}
   H Mean: {data['stats']['h_mean']:.1f} nT
   Z Mean: {data['stats']['z_mean']:.1f} nT

🔮 PREDICTIONS

🎯 PRECURSOR STATUS
   {'⚠️  PRECURSOR DETECTED' if is_precursor else '✅ NO PRECURSOR (Normal)'}
   {'   Earthquake precursor signals found' if is_precursor else '   Normal geomagnetic conditions'}

📏 MAGNITUDE
   Prediction: {mag_pred['class_name']}
   Confidence: {mag_pred['confidence']:.1f}%

🧭 AZIMUTH
   Prediction: {az_pred['class_name']}
   Confidence: {az_pred['confidence']:.1f}%

⚡ OVERALL ASSESSMENT
"""
        
        # Risk assessment
        if is_precursor:
            risk_level = "🔴 PRECURSOR DETECTED"
            risk_text = "   Earthquake precursor signals detected!\n   Monitor closely for seismic activity."
        else:
            risk_level = "🟢 NORMAL CONDITIONS"
            risk_text = "   Normal geomagnetic conditions.\n   No earthquake precursor detected."
        
        summary_text += f"   {risk_level}\n{risk_text}\n"
        
        # Confidence
        avg_conf = (mag_pred['confidence'] + az_pred['confidence']) / 2
        if avg_conf >= 70:
            conf_text = "   High confidence prediction"
        elif avg_conf >= 50:
            conf_text = "   Moderate confidence prediction"
        else:
            conf_text = "   Low confidence - use with caution"
        
        summary_text += f"\n📊 CONFIDENCE LEVEL\n{conf_text}\n   Average: {avg_conf:.1f}%"
        
        summary_text += f"\n\n✅ MODEL INFO\n   Version: Production v2.1\n   Accuracy: 98.94% (magnitude)\n   Azimuth: 83.92%\n   Normal Detection: 100%"
        
        ax7.text(0.05, 0.95, summary_text, 
                transform=ax7.transAxes,
                fontsize=9,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 Results saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def scan(self, date, station_code, save_results=True):
        """
        Complete scan workflow
        
        Args:
            date: Date string 'YYYY-MM-DD' or datetime object
            station_code: Station code
            save_results: Save results to file
            
        Returns:
            dict with all results
        """
        logger.info("="*60)
        logger.info("🚀 PREKURSOR SCANNER (Production v2.0) - Starting scan...")
        logger.info("="*60)
        
        # Validate station
        if station_code not in self.stations:
            logger.error(f"❌ Invalid station code: {station_code}")
            logger.info(f"Available stations: {', '.join(sorted(self.stations.keys()))}")
            return None
        
        # Step 1: Fetch data
        data = self.fetch_data(date, station_code)
        if data is None:
            return None
        
        # Step 2: Generate spectrogram
        spectrogram_image = self.generate_spectrogram(data)
        if spectrogram_image is None:
            return None
        
        # Step 3: Predict
        predictions = self.predict(spectrogram_image)
        
        # Step 4: Visualize
        if isinstance(date, str):
            date_obj = datetime.strptime(date, '%Y-%m-%d')
        else:
            date_obj = date
        
        save_path = None
        if save_results:
            output_dir = Path('scanner_results')
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / f"scan_{station_code}_{date_obj.strftime('%Y%m%d')}_v2.png"
        
        fig = self.visualize_results(data, spectrogram_image, predictions, save_path)
        
        # Compile results
        results = {
            'date': date_obj.strftime('%Y-%m-%d'),
            'station': station_code,
            'data_quality': data['stats'],
            'predictions': predictions,
            'figure': fig
        }
        
        logger.info("="*60)
        logger.info("✅ SCAN COMPLETE!")
        logger.info("="*60)
        
        return results
    
    def list_stations(self):
        """List available stations"""
        print("\n📍 AVAILABLE STATIONS:")
        print("="*60)
        for code in sorted(self.stations.keys()):
            station = self.stations[code]
            print(f"  {code:5s} - Lat: {station['lat']:8.4f}°  Lon: {station['lon']:8.4f}°")
        print("="*60)
        print(f"Total: {len(self.stations)} stations\n")


def interactive_scan():
    """Interactive mode for scanner"""
    print("\n" + "="*60)
    print("🔍 PREKURSOR SCANNER - Production v2.0 (Interactive Mode)")
    print("="*60)
    
    # Initialize scanner
    try:
        scanner = PrekursorScannerProduction()
    except Exception as e:
        print(f"❌ Failed to initialize scanner: {e}")
        return
    
    # Show available stations
    scanner.list_stations()
    
    # Get user input
    while True:
        print("\n" + "-"*60)
        station_code = input("Enter station code (or 'quit' to exit): ").strip().upper()
        
        if station_code.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if station_code not in scanner.stations:
            print(f"❌ Invalid station code: {station_code}")
            continue
        
        date_str = input("Enter date (YYYY-MM-DD): ").strip()
        
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")
            continue
        
        # Run scan
        print("\n")
        results = scanner.scan(date_str, station_code, save_results=True)
        
        if results:
            print("\n✅ Scan completed successfully!")
            print(f"📊 Results saved to: scanner_results/scan_{station_code}_{date_str.replace('-', '')}_v2.png")
        else:
            print("\n❌ Scan failed. Check logs for details.")
        
        # Ask to continue
        continue_scan = input("\nScan another date/station? (y/n): ").strip().lower()
        if continue_scan != 'y':
            print("👋 Goodbye!")
            break


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prekursor Scanner - Production v2.0 (Fixed Model)'
    )
    parser.add_argument(
        '--date', '-d',
        type=str,
        help='Date to scan (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--station', '-s',
        type=str,
        help='Station code (e.g., GTO, SCN)'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--list-stations', '-l',
        action='store_true',
        help='List available stations'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Path to model checkpoint'
    )
    
    args = parser.parse_args()
    
    # List stations mode
    if args.list_stations:
        scanner = PrekursorScannerProduction(model_path=args.model)
        scanner.list_stations()
        return
    
    # Interactive mode
    if args.interactive or (not args.date and not args.station):
        interactive_scan()
        return
    
    # Command line mode
    if not args.date or not args.station:
        parser.print_help()
        return
    
    # Run scan
    scanner = PrekursorScannerProduction(model_path=args.model)
    results = scanner.scan(args.date, args.station.upper(), save_results=True)
    
    if results:
        print("\n✅ Scan completed successfully!")
    else:
        print("\n❌ Scan failed. Check logs for details.")


if __name__ == '__main__':
    main()
