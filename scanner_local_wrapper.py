#!/usr/bin/env python3
"""
Local Scanner Wrapper for Dashboard
Uses local data files instead of SSH connection

Author: Earthquake Prediction Research Team
Date: 3 February 2026
"""

import os
import sys
import gzip
import struct
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal
import logging
from pathlib import Path
import json
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'intial'))
from intial.signal_processing import GeomagneticSignalProcessor


class MultiTaskVGG16(nn.Module):
    """Multi-task VGG16 model"""
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


class MultiTaskConvNeXt(nn.Module):
    """Multi-task ConvNeXt model - matches trained architecture exactly"""
    def __init__(self, num_magnitude_classes=4, num_azimuth_classes=9):
        super(MultiTaskConvNeXt, self).__init__()
        self.backbone = models.convnext_tiny(pretrained=False)
        num_features = self.backbone.classifier[2].in_features  # 768
        self.backbone.classifier = nn.Identity()
        
        # Magnitude head (index: 0=LN, 1=Flatten, 2=Linear, 3=GELU, 4=Drop, 5=Linear)
        self.mag_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Flatten(start_dim=1),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_magnitude_classes)
        )
        
        # Azimuth head
        self.azi_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Flatten(start_dim=1),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_azimuth_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.mean([-2, -1])
        return self.mag_head(features), self.azi_head(features)


class MultiTaskEfficientNet(nn.Module):
    """Multi-task EfficientNet model - matches trained architecture"""
    def __init__(self, num_magnitude_classes=4, num_azimuth_classes=9, dropout_rate=0.4):
        super(MultiTaskEfficientNet, self).__init__()
        base_model = models.efficientnet_b0(pretrained=False)
        feature_dim = base_model.classifier[1].in_features  # 1280
        
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Shared classifier (matches training)
        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
        )
        
        self.magnitude_head = nn.Linear(512, num_magnitude_classes)
        self.azimuth_head = nn.Linear(512, num_azimuth_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared_fc(x)
        return self.magnitude_head(x), self.azimuth_head(x)


class LocalScannerWrapper:
    """Scanner that uses local data files - supports multiple model architectures"""
    
    def __init__(self, model_path=None, model_arch=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_arch = model_arch
        self.model, self.class_mappings = self._load_model(model_path)
        self.signal_processor = GeomagneticSignalProcessor(sampling_rate=1.0)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info(f"✅ Local scanner initialized with {self.model_arch.upper()} model")
    
    def _load_model(self, model_path=None):
        """Load model with architecture detection"""
        if model_path is None:
            model_path = Path('experiments_fixed/exp_fixed_20260202_163643/best_model.pth')
        else:
            model_path = Path(model_path)
        
        exp_dir = model_path.parent
        
        # Load class mappings
        mapping_file = exp_dir / 'class_mappings.json'
        default_mappings = {
            'magnitude_classes': ['Large', 'Medium', 'Moderate', 'Normal'],
            'azimuth_classes': ['E', 'N', 'NE', 'NW', 'S', 'SE', 'SW', 'W', 'Unknown']
        }
        
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                raw_mappings = json.load(f)
            
            # Handle different formats
            if 'magnitude_classes' in raw_mappings:
                class_mappings = raw_mappings
            elif 'magnitude' in raw_mappings:
                # ConvNeXt format: {"magnitude": {"0": "Large", ...}}
                mag_dict = raw_mappings['magnitude']
                azi_dict = raw_mappings['azimuth']
                class_mappings = {
                    'magnitude_classes': [mag_dict[str(i)] for i in range(len(mag_dict))],
                    'azimuth_classes': [azi_dict[str(i)] for i in range(len(azi_dict))]
                }
            else:
                class_mappings = default_mappings
        else:
            config_file = exp_dir / 'config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                class_mappings = {
                    'magnitude_classes': config.get('magnitude_classes', default_mappings['magnitude_classes']),
                    'azimuth_classes': config.get('azimuth_classes', default_mappings['azimuth_classes'])
                }
            else:
                logger.warning("Class mappings not found, using defaults")
                class_mappings = default_mappings
        
        # Detect architecture
        if self.model_arch is None:
            if 'convnext' in str(model_path).lower():
                self.model_arch = 'convnext'
            elif 'efficientnet' in str(model_path).lower():
                self.model_arch = 'efficientnet'
            else:
                self.model_arch = 'vgg16'
        
        # Create model
        num_mag = len(class_mappings['magnitude_classes'])
        num_azi = len(class_mappings['azimuth_classes'])
        
        if self.model_arch == 'convnext':
            model = MultiTaskConvNeXt(num_mag, num_azi)
        elif self.model_arch == 'efficientnet':
            model = MultiTaskEfficientNet(num_mag, num_azi)
        else:
            model = MultiTaskVGG16(num_mag, num_azi)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model, class_mappings
    
    def read_local_file(self, date_str, station):
        """Read local data file"""
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Format: S180117.SCN.gz
        filename = f"S{date_obj.strftime('%y%m%d')}.{station}.gz"
        filepath = Path(f"mdata2/{station}/{filename}")
        
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            return None
        
        logger.info(f"📂 Reading local file: {filepath}")
        
        try:
            with gzip.open(filepath, 'rb') as f:
                data_bytes = f.read()
            
            # Parse binary data (86400 samples, 3 components)
            num_samples = 86400
            h_data = []
            d_data = []
            z_data = []
            
            for i in range(num_samples):
                offset = i * 6
                if offset + 6 <= len(data_bytes):
                    h_val = struct.unpack('<h', data_bytes[offset:offset+2])[0]
                    d_val = struct.unpack('<h', data_bytes[offset+2:offset+4])[0]
                    z_val = struct.unpack('<h', data_bytes[offset+4:offset+6])[0]
                    
                    # Convert to nT (scale factor 0.01)
                    h_data.append(h_val * 0.01 if h_val != -32768 else np.nan)
                    d_data.append(d_val * 0.01 if d_val != -32768 else np.nan)
                    z_data.append(z_val * 0.01 if z_val != -32768 else np.nan)
            
            # Calculate stats
            h_valid = [x for x in h_data if not np.isnan(x)]
            z_valid = [x for x in z_data if not np.isnan(x)]
            
            data = {
                'Hcomp': np.array(h_data),
                'Dcomp': np.array(d_data),
                'Zcomp': np.array(z_data),
                'station': station,
                'date': date_obj,
                'stats': {
                    'coverage': (len(h_valid) / num_samples) * 100,
                    'valid_samples': len(h_valid),
                    'h_mean': np.mean(h_valid) if h_valid else 0,
                    'z_mean': np.mean(z_valid) if z_valid else 0
                }
            }
            
            logger.info(f"✅ Data loaded: {data['stats']['coverage']:.1f}% coverage")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error reading file: {e}")
            return None
    
    def generate_spectrogram(self, data):
        """Generate spectrogram (training-compatible)"""
        pc3_low = 0.01
        pc3_high = 0.045
        
        components_data = {}
        for comp_name in ['Hcomp', 'Dcomp', 'Zcomp']:
            signal_data = data[comp_name]
            valid_mask = ~np.isnan(signal_data)
            
            if not np.any(valid_mask):
                return None
            
            signal_clean = np.array(signal_data, dtype=float)
            if np.any(~valid_mask):
                x = np.arange(len(signal_data))
                signal_clean[~valid_mask] = np.interp(
                    x[~valid_mask], x[valid_mask], signal_data[valid_mask]
                )
            
            signal_filtered = self.signal_processor.bandpass_filter(
                signal_clean, low_freq=pc3_low, high_freq=pc3_high
            )
            components_data[comp_name] = signal_filtered
        
        # Generate spectrograms
        fs = 1.0
        nperseg = 256
        noverlap = nperseg // 2
        
        spectrograms_db = {}
        for comp_name, signal_filtered in components_data.items():
            f, t, Sxx = signal.spectrogram(
                signal_filtered, fs=fs, nperseg=nperseg,
                noverlap=noverlap, window='hann'
            )
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            spectrograms_db[comp_name] = (f, t, Sxx_db)
        
        # Create image
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(3, 1, figsize=(2.24, 2.24))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        
        f_h, t_h, Sxx_h_db = spectrograms_db['Hcomp']
        f_d, t_d, Sxx_d_db = spectrograms_db['Dcomp']
        f_z, t_z, Sxx_z_db = spectrograms_db['Zcomp']
        
        freq_mask = (f_h >= pc3_low) & (f_h <= pc3_high)
        f_pc3 = f_h[freq_mask]
        Sxx_h_pc3 = Sxx_h_db[freq_mask, :]
        Sxx_d_pc3 = Sxx_d_db[freq_mask, :]
        Sxx_z_pc3 = Sxx_z_db[freq_mask, :]
        
        axes[0].pcolormesh(t_h, f_pc3, Sxx_h_pc3, shading='gouraud', cmap='jet')
        axes[0].axis('off')
        
        axes[1].pcolormesh(t_d, f_pc3, Sxx_d_pc3, shading='gouraud', cmap='jet')
        axes[1].axis('off')
        
        axes[2].pcolormesh(t_z, f_pc3, Sxx_z_pc3, shading='gouraud', cmap='jet')
        axes[2].axis('off')
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        plt.savefig(tmp_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        img = Image.open(tmp_path)
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        os.unlink(tmp_path)
        return img
    
    def predict(self, spectrogram_image):
        """Predict"""
        image_tensor = self.transform(spectrogram_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mag_output, azi_output = self.model(image_tensor)
            mag_probs = torch.softmax(mag_output, dim=1)[0]
            azi_probs = torch.softmax(azi_output, dim=1)[0]
            
            mag_pred_idx = torch.argmax(mag_probs).item()
            azi_pred_idx = torch.argmax(azi_probs).item()
            
            mag_pred = self.class_mappings['magnitude_classes'][mag_pred_idx]
            azi_pred = self.class_mappings['azimuth_classes'][azi_pred_idx]
            
            mag_conf = mag_probs[mag_pred_idx].item() * 100
            azi_conf = azi_probs[azi_pred_idx].item() * 100
        
        return {
            'magnitude': {
                'class_id': mag_pred_idx,
                'class_name': mag_pred,
                'confidence': mag_conf
            },
            'azimuth': {
                'class_id': azi_pred_idx,
                'class_name': azi_pred,
                'confidence': azi_conf
            },
            'is_normal': mag_pred == 'Normal',
            'is_precursor': mag_pred != 'Normal'
        }
    
    def scan(self, date_str, station, save_path=None):
        """Complete scan"""
        logger.info(f"🔍 Scanning {station} on {date_str}...")
        
        # Read local data
        data = self.read_local_file(date_str, station)
        if data is None:
            return None
        
        # Generate spectrogram
        spectrogram = self.generate_spectrogram(data)
        if spectrogram is None:
            return None
        
        # Predict
        predictions = self.predict(spectrogram)
        
        # Save spectrogram if requested
        if save_path:
            spectrogram.save(save_path)
            logger.info(f"💾 Saved to: {save_path}")
        
        logger.info(f"✅ Scan complete: {predictions['magnitude']['class_name']} ({predictions['magnitude']['confidence']:.1f}%)")
        
        return {
            'date': date_str,
            'station': station,
            'predictions': predictions,
            'spectrogram': spectrogram
        }


if __name__ == '__main__':
    # Test
    scanner = LocalScannerWrapper()
    result = scanner.scan('2018-01-17', 'SCN', save_path='test_scan.png')
    if result:
        print(f"\n✅ Test successful!")
        print(f"Magnitude: {result['predictions']['magnitude']['class_name']}")
        print(f"Azimuth: {result['predictions']['azimuth']['class_name']}")
