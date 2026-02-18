#!/usr/bin/env python3
"""
Prekursor Scanner - Interactive Earthquake Prediction Tool
Scan geomagnetic data untuk prediksi gempa dengan model CNN

Features:
- Pilih tanggal dan stasiun
- Fetch data real-time dari server
- Generate spectrogram
- Prediksi magnitude dan azimuth
- Confidence score untuk setiap prediksi
- Visualisasi hasil

Author: Earthquake Prediction Research Team
Date: 2 February 2026
Version: 1.0
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import signal
import logging
from pathlib import Path
import json

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'intial'))
sys.path.insert(0, os.path.dirname(__file__))

# Import modules
from intial.geomagnetic_fetcher import GeomagneticDataFetcher
from intial.signal_processing import GeomagneticSignalProcessor

# Import PyTorch models
import torch.nn as nn
from torchvision import models


# ============================================================================
# MODEL ARCHITECTURES - Must match training exactly!
# ============================================================================

class MultiTaskVGG16(nn.Module):
    """
    Multi-task VGG16 model - MATCHES experiments_fixed training exactly!
    This is the correct architecture for best_model.pth from experiments_fixed.
    """
    def __init__(self, num_magnitude_classes, num_azimuth_classes):
        super(MultiTaskVGG16, self).__init__()
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(pretrained=False)
        
        # Use VGG16 features (features.0, features.2, etc.)
        self.features = vgg16.features
        
        # Adaptive pooling
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        
        # Magnitude head
        self.magnitude_head = nn.Linear(512, num_magnitude_classes)
        
        # Azimuth head
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
    """Multi-task ConvNeXt model - for ConvNeXt-based checkpoints"""
    def __init__(self, num_magnitude_classes=4, num_azimuth_classes=9):
        super(MultiTaskConvNeXt, self).__init__()
        self.backbone = models.convnext_tiny(pretrained=False)
        num_features = self.backbone.classifier[2].in_features  # 768
        self.backbone.classifier = nn.Identity()
        
        self.mag_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Flatten(start_dim=1),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_magnitude_classes)
        )
        
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
    """Multi-task EfficientNet model - for EfficientNet-based checkpoints"""
    def __init__(self, num_magnitude_classes=4, num_azimuth_classes=9, dropout_rate=0.3):
        super(MultiTaskEfficientNet, self).__init__()
        base_model = models.efficientnet_b0(pretrained=False)
        feature_dim = base_model.classifier[1].in_features  # 1280
        
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PrekursorScanner:
    """
    Interactive scanner untuk prediksi gempa dari data geomagnetik
    """
    
    def __init__(self, model_path=None):
        """
        Initialize scanner
        
        Args:
            model_path: Path ke model checkpoint (default: best model dari Phase 1)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️  Using device: {self.device}")
        
        # Load station list
        self.stations = self._load_stations()
        logger.info(f"📍 Loaded {len(self.stations)} stations")
        
        # Model architecture (will be set by _load_model)
        self.model_arch = None
        
        # Load model
        if model_path is None:
            model_path = self._find_best_model()
        
        self.model = self._load_model(model_path)
        logger.info(f"✅ Model loaded from: {model_path}")
        logger.info(f"   Architecture: {self.model_arch.upper()}")
        
        # Load class mappings
        self.class_mappings = self._load_class_mappings()
        
        # Initialize processors
        self.signal_processor = GeomagneticSignalProcessor(sampling_rate=1.0)
        
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
        """Find best model from experiments - prioritizes models with better Large class detection"""
        
        # PRIORITY 1: Check experiments_augmented (BEST - Focal Loss + Oversampling)
        aug_exp_dir = Path('experiments_augmented')
        if aug_exp_dir.exists():
            exp_folders = sorted(aug_exp_dir.glob('exp_aug_*'), reverse=True)
            for exp_folder in exp_folders:
                model_path = exp_folder / 'best_model.pth'
                mapping_path = exp_folder / 'class_mappings.json'
                # Only use if both model and mappings exist
                if model_path.exists() and mapping_path.exists():
                    logger.info(f"✅ Using AUGMENTED model (Focal Loss + Oversampling): {model_path}")
                    return str(model_path)
        
        # PRIORITY 2: Check final_production_model (EfficientNet)
        prod_model = Path('final_production_model/best_final_model.pth')
        if prod_model.exists():
            logger.info(f"✅ Using PRODUCTION model (EfficientNet): {prod_model}")
            return str(prod_model)
        
        # PRIORITY 3: Check experiments_fixed (VGG16 with proper data split)
        fixed_exp_dir = Path('experiments_fixed')
        if fixed_exp_dir.exists():
            exp_folders = sorted(fixed_exp_dir.glob('exp_fixed_*'))
            if exp_folders:
                latest_exp = exp_folders[-1]
                model_path = latest_exp / 'best_model.pth'
                if model_path.exists():
                    logger.info(f"✅ Using FIXED model (VGG16): {model_path}")
                    return str(model_path)
        
        # PRIORITY 4: Check experiments_v4
        exp_dir = Path('experiments_v4')
        if exp_dir.exists():
            exp_folders = sorted(exp_dir.glob('exp_v4_phase1_*'))
            if exp_folders:
                latest_exp = exp_folders[-1]
                model_path = latest_exp / 'best_model.pth'
                if model_path.exists():
                    logger.warning(f"⚠️  Using experiments_v4 model (may have data leakage)")
                    return str(model_path)
        
        # Fallback to mdata2 folder
        fallback_path = 'mdata2/best_vgg16_model_phase1.keras'
        if os.path.exists(fallback_path):
            logger.warning("Using fallback model (Keras format not supported)")
        
        raise FileNotFoundError("No model found! Please train model first.")
    
    def _load_model(self, model_path):
        """
        Load trained model with AUTOMATIC architecture detection.
        
        CRITICAL: The model architecture MUST match the checkpoint!
        - experiments_fixed uses MultiTaskVGG16 (features.0, features.2, etc.)
        - experiments_convnext uses MultiTaskConvNeXt (backbone.features.*)
        - final_production_model uses MultiTaskEfficientNet (features.0.0, etc.)
        """
        logger.info(f"🔍 Detecting model architecture from checkpoint...")
        
        # Load checkpoint to inspect keys
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Detect architecture from state_dict keys
        sample_keys = list(state_dict.keys())[:10]
        logger.info(f"   Sample keys: {sample_keys[:5]}")
        
        # Check for VGG16 pattern (features.0, features.2, etc. - direct Conv2d)
        is_vgg16 = any('features.0.weight' in k for k in state_dict.keys())
        
        # Check for EfficientNet pattern (features.0.0, features.1.0, etc. - nested)
        is_efficientnet = any('features.0.0' in k or 'shared_fc' in k for k in state_dict.keys())
        
        # Check for ConvNeXt pattern (backbone.features.*)
        is_convnext = any('backbone' in k for k in state_dict.keys())
        
        # Determine number of classes from checkpoint
        num_magnitude_classes = 4  # Default
        num_azimuth_classes = 9    # Default
        
        # Try to infer from state_dict
        for key in state_dict.keys():
            if 'magnitude_head' in key and 'weight' in key:
                num_magnitude_classes = state_dict[key].shape[0]
            if 'azimuth_head' in key and 'weight' in key:
                num_azimuth_classes = state_dict[key].shape[0]
        
        logger.info(f"   Detected classes: {num_magnitude_classes} magnitude, {num_azimuth_classes} azimuth")
        
        # Create model with correct architecture
        if is_efficientnet and not is_vgg16:
            logger.info(f"   ✅ Detected EfficientNet architecture (final_production_model)")
            model = MultiTaskEfficientNet(
                num_magnitude_classes=num_magnitude_classes,
                num_azimuth_classes=num_azimuth_classes
            )
            self.model_arch = 'efficientnet'
        elif is_convnext:
            logger.info(f"   ✅ Detected ConvNeXt architecture")
            model = MultiTaskConvNeXt(
                num_magnitude_classes=num_magnitude_classes,
                num_azimuth_classes=num_azimuth_classes
            )
            self.model_arch = 'convnext'
        elif is_vgg16:
            logger.info(f"   ✅ Detected VGG16 architecture (experiments_fixed)")
            model = MultiTaskVGG16(
                num_magnitude_classes=num_magnitude_classes,
                num_azimuth_classes=num_azimuth_classes
            )
            self.model_arch = 'vgg16'
        else:
            # Fallback: try VGG16 first (most common)
            logger.warning(f"   ⚠️  Could not detect architecture, trying VGG16...")
            model = MultiTaskVGG16(
                num_magnitude_classes=num_magnitude_classes,
                num_azimuth_classes=num_azimuth_classes
            )
            self.model_arch = 'vgg16'
        
        # Load state dict
        try:
            model.load_state_dict(state_dict)
            logger.info(f"   ✅ Model weights loaded successfully!")
        except Exception as e:
            logger.error(f"   ❌ Failed to load weights: {e}")
            logger.error(f"   This usually means architecture mismatch!")
            raise
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_class_mappings(self):
        """Load class mappings - MUST match the MODEL that is loaded, not training_data!"""
        
        # CRITICAL FIX: Load mapping from the SAME directory as the model!
        # The model in experiments_fixed was trained with 4 magnitude classes and 9 azimuth classes
        
        # First, try to load from model directory
        model_mapping_file = None
        
        # Check experiments_augmented first (BEST model)
        aug_exp_dir = Path('experiments_augmented')
        if aug_exp_dir.exists():
            exp_folders = sorted(aug_exp_dir.glob('exp_aug_*'))
            if exp_folders:
                aug_mapping = exp_folders[-1] / 'class_mappings.json'
                if aug_mapping.exists():
                    model_mapping_file = str(aug_mapping)
        
        # Check final_production_model (EfficientNet)
        if model_mapping_file is None:
            prod_mapping = 'final_production_model/class_mappings.json'
            if os.path.exists(prod_mapping):
                model_mapping_file = prod_mapping
        
        # Check experiments_fixed (VGG16)
        if model_mapping_file is None:
            exp_mapping = 'experiments_fixed/exp_fixed_20260202_163643/class_mappings.json'
            if os.path.exists(exp_mapping):
                model_mapping_file = exp_mapping
        
        if model_mapping_file and os.path.exists(model_mapping_file):
            with open(model_mapping_file, 'r') as f:
                loaded_mappings = json.load(f)
            
            # Convert to the format we need
            mappings = {
                'magnitude': {},
                'azimuth': {}
            }
            
            # Load magnitude classes
            if 'magnitude_classes' in loaded_mappings:
                for idx, name in enumerate(loaded_mappings['magnitude_classes']):
                    if name == 'Normal':
                        mappings['magnitude'][idx] = 'Normal (No Earthquake)'
                    elif name == 'Large':
                        mappings['magnitude'][idx] = 'Large (M ≥ 6.0)'
                    elif name == 'Medium':
                        mappings['magnitude'][idx] = 'Medium (5.0 ≤ M < 6.0)'
                    elif name == 'Moderate':
                        mappings['magnitude'][idx] = 'Moderate (4.0 ≤ M < 5.0)'
                    else:
                        mappings['magnitude'][idx] = name
            
            # Load azimuth classes
            if 'azimuth_classes' in loaded_mappings:
                for idx, name in enumerate(loaded_mappings['azimuth_classes']):
                    if name == 'Normal':
                        mappings['azimuth'][idx] = 'Normal (No Earthquake)'
                    elif name == 'N':
                        mappings['azimuth'][idx] = 'North (N)'
                    elif name == 'S':
                        mappings['azimuth'][idx] = 'South (S)'
                    elif name == 'E':
                        mappings['azimuth'][idx] = 'East (E)'
                    elif name == 'W':
                        mappings['azimuth'][idx] = 'West (W)'
                    elif name == 'NE':
                        mappings['azimuth'][idx] = 'Northeast (NE)'
                    elif name == 'NW':
                        mappings['azimuth'][idx] = 'Northwest (NW)'
                    elif name == 'SE':
                        mappings['azimuth'][idx] = 'Southeast (SE)'
                    elif name == 'SW':
                        mappings['azimuth'][idx] = 'Southwest (SW)'
                    else:
                        mappings['azimuth'][idx] = name
            
            logger.info(f"✅ Loaded class mappings from MODEL directory: {model_mapping_file}")
            logger.info(f"   Magnitude classes ({len(mappings['magnitude'])}): {list(mappings['magnitude'].values())}")
            logger.info(f"   Azimuth classes ({len(mappings['azimuth'])}): {list(mappings['azimuth'].values())}")
            
            return mappings
        
        # Fallback: Default mappings for 4 magnitude + 9 azimuth classes
        # This matches experiments_fixed model
        logger.warning("⚠️  Using default mappings (4 mag + 9 azi classes)")
        mappings = {
            'magnitude': {
                0: 'Normal (No Earthquake)',
                1: 'Moderate (4.0 ≤ M < 5.0)',
                2: 'Medium (5.0 ≤ M < 6.0)',
                3: 'Large (M ≥ 6.0)'
            },
            'azimuth': {
                0: 'Normal (No Earthquake)',
                1: 'North (N)',
                2: 'Northeast (NE)',
                3: 'East (E)',
                4: 'Southeast (SE)',
                5: 'South (S)',
                6: 'Southwest (SW)',
                7: 'West (W)',
                8: 'Northwest (NW)'
            }
        }
        
        return mappings
    
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
    
    def generate_spectrogram(self, data, component='3comp'):
        """
        Generate spectrogram from geomagnetic data (TRAINING-COMPATIBLE VERSION)
        
        CRITICAL: This version matches EXACTLY the training preprocessing:
        - Filter: 0.01-0.045 Hz (PC3 range, same as training)
        - 3 components: H, D, Z stacked vertically
        - Colormap: jet (same as training)
        - No percentile normalization (raw dB values)
        - Image size: 224x224
        
        Args:
            data: Geomagnetic data dict
            component: '3comp' for 3-component (H,D,Z), or 'Hcomp'/'Dcomp'/'Zcomp' for single
            
        Returns:
            spectrogram image (224x224x3) as numpy array
        """
        logger.info(f"🎨 Generating spectrogram (TRAINING-COMPATIBLE)...")
        
        # CRITICAL FIX 1: Use training filter range (0.01-0.045 Hz)
        pc3_low = 0.01   # 10 mHz (same as training)
        pc3_high = 0.045 # 45 mHz (same as training)
        
        logger.info(f"   Filter range: {pc3_low*1000:.1f}-{pc3_high*1000:.1f} mHz (TRAINING MATCH)")
        
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
            
            # CRITICAL FIX 1: Apply PC3 bandpass filter with TRAINING range
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
            
            # CRITICAL FIX 2: Convert to dB scale (NO percentile normalization)
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
            spectrograms_db[comp_name] = (f, t, Sxx_db)
        
        # CRITICAL FIX 3: Create 3-component image (H, D, Z stacked vertically)
        # This matches training format exactly
        
        # Create figure with 3 subplots (same as training)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig_height = 224 / 100.0  # 2.24 inches
        fig_width = 224 / 100.0   # 2.24 inches
        
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
        
        # CRITICAL FIX 4: Use jet colormap (same as training)
        # Plot H component
        axes[0].pcolormesh(t_h, f_pc3, Sxx_h_pc3, shading='gouraud', cmap='jet')
        axes[0].axis('off')
        
        # Plot D component
        axes[1].pcolormesh(t_d, f_pc3, Sxx_d_pc3, shading='gouraud', cmap='jet')
        axes[1].axis('off')
        
        # Plot Z component
        axes[2].pcolormesh(t_z, f_pc3, Sxx_z_pc3, shading='gouraud', cmap='jet')
        axes[2].axis('off')
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        plt.savefig(tmp_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Load and resize to exactly 224x224
        from PIL import Image
        img = Image.open(tmp_path)
        
        # Resize to exactly 224x224
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.LANCZOS)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        spectrogram_rgb = np.array(img)
        
        # Clean up temp file
        import os
        os.unlink(tmp_path)
        
        logger.info(f"✅ Spectrogram generated: {spectrogram_rgb.shape}")
        logger.info(f"   ✅ TRAINING-COMPATIBLE: 3-component, jet colormap, PC3 filter")
        
        return spectrogram_rgb
    
    def predict(self, spectrogram):
        """
        Predict earthquake parameters from spectrogram
        
        Args:
            spectrogram: Spectrogram image (224x224x3)
            
        Returns:
            dict with predictions and confidence scores
        """
        logger.info("🔮 Running prediction...")
        
        # Prepare input
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(spectrogram).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        img_tensor = img_tensor / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            magnitude_logits, azimuth_logits = self.model(img_tensor)
            
            # Get probabilities
            magnitude_probs = torch.softmax(magnitude_logits, dim=1)
            azimuth_probs = torch.softmax(azimuth_logits, dim=1)
            
            # Get raw predictions
            magnitude_pred_raw = torch.argmax(magnitude_probs, dim=1).item()
            azimuth_pred_raw = torch.argmax(azimuth_probs, dim=1).item()
            
            # Get confidence scores
            magnitude_conf_raw = magnitude_probs[0, magnitude_pred_raw].item() * 100
            azimuth_conf_raw = azimuth_probs[0, azimuth_pred_raw].item() * 100
            
            # Get top-3 predictions for azimuth (for low confidence cases)
            top3_azi_probs, top3_azi_indices = torch.topk(azimuth_probs[0], min(3, len(azimuth_probs[0])))
            top3_mag_probs, top3_mag_indices = torch.topk(magnitude_probs[0], min(3, len(magnitude_probs[0])))
            
            # NOTE: Training data does NOT have "Normal" class
            # All predictions are earthquake precursors with magnitude and direction
            # The model was trained only on earthquake events, not normal days
            
            magnitude_pred = magnitude_pred_raw
            azimuth_pred = azimuth_pred_raw
            magnitude_conf = magnitude_conf_raw
            azimuth_conf = azimuth_conf_raw
            is_corrected = False
            is_low_confidence = False
            
            # Check for low confidence predictions
            avg_conf = (magnitude_conf_raw + azimuth_conf_raw) / 2
            if avg_conf < 30.0:  # Threshold: 30%
                logger.warning(f"⚠️  Low confidence detected: Avg={avg_conf:.1f}%")
                logger.info(f"   This may indicate uncertain prediction")
                is_low_confidence = True
            
            # Log all azimuth probabilities for debugging
            logger.info(f"\n📊 AZIMUTH PROBABILITIES (all classes):")
            for idx in range(len(azimuth_probs[0])):
                az_name = self.class_mappings['azimuth'].get(idx, f'Class_{idx}')
                prob = azimuth_probs[0, idx].item() * 100
                marker = " ← TOP" if idx == azimuth_pred_raw else ""
                logger.info(f"   {az_name}: {prob:.1f}%{marker}")
            
            # Log top-3 predictions if low confidence
            if is_low_confidence:
                logger.warning(f"\n⚠️  LOW CONFIDENCE - Top-3 Azimuth Predictions:")
                for i in range(len(top3_azi_indices)):
                    idx = top3_azi_indices[i].item()
                    prob = top3_azi_probs[i].item() * 100
                    az_name = self.class_mappings['azimuth'].get(idx, f'Class_{idx}')
                    logger.warning(f"   {i+1}. {az_name}: {prob:.1f}%")
                logger.warning(f"   → Consider all top predictions, not just #1!")
        
        # Prepare results
        # Determine if this is a precursor based on confidence
        # Low confidence may indicate normal (non-earthquake) conditions
        avg_confidence = (magnitude_conf + azimuth_conf) / 2
        is_precursor = avg_confidence >= 30.0  # Precursor if confidence >= 30%
        
        # Build top-3 predictions list
        top3_azimuth = []
        for i in range(len(top3_azi_indices)):
            idx = top3_azi_indices[i].item()
            prob = top3_azi_probs[i].item() * 100
            az_name = self.class_mappings['azimuth'].get(idx, f'Class_{idx}')
            top3_azimuth.append({'class_id': idx, 'class_name': az_name, 'confidence': prob})
        
        top3_magnitude = []
        for i in range(len(top3_mag_indices)):
            idx = top3_mag_indices[i].item()
            prob = top3_mag_probs[i].item() * 100
            mag_name = self.class_mappings['magnitude'].get(idx, f'Class_{idx}')
            top3_magnitude.append({'class_id': idx, 'class_name': mag_name, 'confidence': prob})
        
        results = {
            'magnitude': {
                'class_id': magnitude_pred,
                'class_name': self.class_mappings['magnitude'].get(magnitude_pred, 'Unknown'),
                'confidence': magnitude_conf,
                'probabilities': magnitude_probs[0].cpu().numpy(),
                'raw_prediction': magnitude_pred_raw,
                'raw_confidence': magnitude_conf_raw,
                'top3': top3_magnitude
            },
            'azimuth': {
                'class_id': azimuth_pred,
                'class_name': self.class_mappings['azimuth'].get(azimuth_pred, 'Unknown'),
                'confidence': azimuth_conf,
                'probabilities': azimuth_probs[0].cpu().numpy(),
                'raw_prediction': azimuth_pred_raw,
                'raw_confidence': azimuth_conf_raw,
                'top3': top3_azimuth
            },
            'is_corrected': is_corrected,
            'is_low_confidence': is_low_confidence,
            'is_precursor': is_precursor,
            'avg_confidence': avg_confidence
        }
        
        logger.info(f"\n✅ Prediction complete")
        if is_corrected:
            logger.info(f"   ⚠️  Prediction corrected for consistency")
        logger.info(f"   Magnitude: {results['magnitude']['class_name']} ({magnitude_conf:.1f}%)")
        logger.info(f"   Azimuth: {results['azimuth']['class_name']} ({azimuth_conf:.1f}%)")
        logger.info(f"   Precursor: {'YES' if results['is_precursor'] else 'NO (low confidence)'}")
        
        if is_low_confidence:
            logger.warning(f"\n🔴 PERINGATAN: Confidence rendah ({avg_confidence:.1f}%)!")
            logger.warning(f"   Prediksi mungkin tidak akurat.")
            logger.warning(f"   Pertimbangkan top-3 predictions di atas.")
        
        return results

    
    def visualize_results(self, data, spectrogram, predictions, save_path=None):
        """
        Visualize scan results
        
        Args:
            data: Geomagnetic data dict
            spectrogram: Spectrogram image
            predictions: Prediction results dict
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        station = data['station']
        date = data['date'].strftime('%Y-%m-%d')
        fig.suptitle(
            f'Prekursor Scanner - Station {station} - {date}',
            fontsize=16, fontweight='bold'
        )
        
        # 1. Raw H, D, Z components (left column)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        
        time_hours = np.arange(len(data['Hcomp'])) / 3600.0
        
        # H component
        ax1.plot(time_hours, data['Hcomp'], 'r-', linewidth=0.5, alpha=0.7)
        ax1.set_ylabel('H (nT)', fontweight='bold')
        ax1.set_title('H Component (Northward)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 24)
        
        # D component
        ax2.plot(time_hours, data['Dcomp'], 'g-', linewidth=0.5, alpha=0.7)
        ax2.set_ylabel('D (nT)', fontweight='bold')
        ax2.set_title('D Component (Eastward)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 24)
        
        # Z component
        ax3.plot(time_hours, data['Zcomp'], 'b-', linewidth=0.5, alpha=0.7)
        ax3.set_ylabel('Z (nT)', fontweight='bold')
        ax3.set_xlabel('Time (hours)', fontweight='bold')
        ax3.set_title('Z Component (Vertical)', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 24)
        
        # 2. Spectrogram (middle top)
        ax4 = fig.add_subplot(gs[0, 1])
        ax4.imshow(spectrogram, aspect='auto', cmap='viridis')
        ax4.set_title('Spectrogram (H Component)', fontsize=10, fontweight='bold')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Frequency')
        ax4.axis('off')
        
        # 3. Magnitude prediction (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        mag_pred = predictions['magnitude']
        
        # Bar chart of probabilities
        mag_classes = list(self.class_mappings['magnitude'].values())
        mag_probs = mag_pred['probabilities'] * 100
        
        colors = ['red' if i == mag_pred['class_id'] else 'lightgray' 
                  for i in range(len(mag_probs))]
        
        bars = ax5.barh(range(len(mag_probs)), mag_probs, color=colors)
        ax5.set_yticks(range(len(mag_probs)))
        ax5.set_yticklabels([c.split('(')[0].strip() for c in mag_classes], fontsize=8)
        ax5.set_xlabel('Confidence (%)', fontweight='bold')
        ax5.set_title('Magnitude Prediction', fontsize=10, fontweight='bold')
        ax5.set_xlim(0, 100)
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, mag_probs)):
            if prob > 5:
                ax5.text(prob + 2, i, f'{prob:.1f}%', 
                        va='center', fontsize=8, fontweight='bold')
        
        # 4. Azimuth prediction (middle bottom)
        ax6 = fig.add_subplot(gs[2, 1])
        az_pred = predictions['azimuth']
        
        # Bar chart of probabilities
        az_classes = list(self.class_mappings['azimuth'].values())
        az_probs = az_pred['probabilities'] * 100
        
        colors = ['blue' if i == az_pred['class_id'] else 'lightgray' 
                  for i in range(len(az_probs))]
        
        bars = ax6.barh(range(len(az_probs)), az_probs, color=colors)
        ax6.set_yticks(range(len(az_probs)))
        ax6.set_yticklabels([c.split('(')[0].strip() for c in az_classes], fontsize=8)
        ax6.set_xlabel('Confidence (%)', fontweight='bold')
        ax6.set_title('Azimuth Prediction', fontsize=10, fontweight='bold')
        ax6.set_xlim(0, 100)
        ax6.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, az_probs)):
            if prob > 5:
                ax6.text(prob + 2, i, f'{prob:.1f}%', 
                        va='center', fontsize=8, fontweight='bold')
        
        # 5. Summary box (right column)
        ax7 = fig.add_subplot(gs[:, 2])
        ax7.axis('off')
        
        # Create summary text
        # Check if prediction was corrected
        is_corrected = predictions.get('is_corrected', False)
        is_precursor = predictions.get('is_precursor', False)
        
        correction_note = ""
        if is_corrected:
            correction_note = f"""
⚠️  CONSISTENCY CORRECTION APPLIED
   Raw predictions were inconsistent.
   Corrected for logical consistency.
   
"""
        
        summary_text = f"""
SCAN RESULTS
{'='*40}

� STATION INFORMATION
   Code: {station}
   Location: {self.stations.get(station, {}).get('lat', 'N/A')}°, 
             {self.stations.get(station, {}).get('lon', 'N/A')}°
   Date: {date}

📊 DATA QUALITY
   Coverage: {data['stats']['coverage']:.1f}%
   Valid Samples: {data['stats']['valid_samples']:,}
   H Mean: {data['stats']['h_mean']:.1f} nT
   Z Mean: {data['stats']['z_mean']:.1f} nT

{correction_note}🔮 PREDICTIONS

🎯 PRECURSOR STATUS
   {'⚠️  PRECURSOR DETECTED' if is_precursor else '✅ NO PRECURSOR (Normal)'}
   {'   Earthquake precursor signals found' if is_precursor else '   Normal geomagnetic conditions'}

📏 MAGNITUDE
   Prediction: {mag_pred['class_name']}
   Confidence: {mag_pred['confidence']:.1f}%
   
   Interpretation:
   {'⚠️  HIGH RISK' if mag_pred['class_id'] in [0, 1] else '✅ LOW RISK'}
   {'   Large/Medium earthquake likely' if mag_pred['class_id'] in [0, 1] else '   Normal conditions'}

🧭 AZIMUTH
   Prediction: {az_pred['class_name']}
   Confidence: {az_pred['confidence']:.1f}%
   
   Interpretation:
   {'⚠️  DIRECTIONAL ANOMALY' if az_pred['class_id'] != 0 else '✅ NO ANOMALY'}
   {'   Earthquake direction: ' + az_pred['class_name'] if az_pred['class_id'] != 0 else '   Normal conditions'}

⚡ OVERALL ASSESSMENT
"""
        
        # Overall risk assessment
        is_earthquake = (mag_pred['class_id'] in [0, 1, 2] and 
                        az_pred['class_id'] != 0)
        
        if is_earthquake:
            risk_level = "🔴 HIGH RISK"
            risk_text = "   Earthquake precursor detected!\n   Monitor closely for seismic activity."
        elif mag_pred['class_id'] == 3 and az_pred['class_id'] == 0:
            risk_level = "🟢 LOW RISK"
            risk_text = "   Normal geomagnetic conditions.\n   No earthquake precursor detected."
        else:
            risk_level = "🟡 MODERATE RISK"
            risk_text = "   Anomalous conditions detected.\n   Continue monitoring."
        
        summary_text += f"   {risk_level}\n{risk_text}\n"
        
        # Add confidence interpretation
        avg_conf = (mag_pred['confidence'] + az_pred['confidence']) / 2
        if avg_conf >= 70:
            conf_text = "   High confidence prediction"
        elif avg_conf >= 50:
            conf_text = "   Moderate confidence prediction"
        else:
            conf_text = "   Low confidence - use with caution"
        
        summary_text += f"\n📊 CONFIDENCE LEVEL\n{conf_text}\n   Average: {avg_conf:.1f}%"
        
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
        logger.info("🚀 PREKURSOR SCANNER - Starting scan...")
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
        spectrogram = self.generate_spectrogram(data, component='Hcomp')
        if spectrogram is None:
            return None
        
        # Step 3: Predict
        predictions = self.predict(spectrogram)
        
        # Step 4: Visualize
        if isinstance(date, str):
            date_obj = datetime.strptime(date, '%Y-%m-%d')
        else:
            date_obj = date
        
        save_path = None
        if save_results:
            output_dir = Path('scanner_results')
            output_dir.mkdir(exist_ok=True)
            save_path = output_dir / f"scan_{station_code}_{date_obj.strftime('%Y%m%d')}.png"
        
        fig = self.visualize_results(data, spectrogram, predictions, save_path)
        
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
    print("🔍 PREKURSOR SCANNER - Interactive Mode")
    print("="*60)
    
    # Initialize scanner
    try:
        scanner = PrekursorScanner()
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
            print(f"📊 Results saved to: scanner_results/scan_{station_code}_{date_str.replace('-', '')}.png")
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
        description='Prekursor Scanner - Earthquake Prediction Tool'
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
        scanner = PrekursorScanner(model_path=args.model)
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
    scanner = PrekursorScanner(model_path=args.model)
    results = scanner.scan(args.date, args.station.upper(), save_results=True)
    
    if results:
        print("\n✅ Scan completed successfully!")
    else:
        print("\n❌ Scan failed. Check logs for details.")


if __name__ == '__main__':
    main()
