#!/usr/bin/env python3
"""
Test Multiple Models - Compare predictions from different models
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from pathlib import Path
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'intial'))
from intial.geomagnetic_fetcher import GeomagneticDataFetcher
from intial.signal_processing import GeomagneticSignalProcessor
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("="*70)
print("TEST MULTIPLE MODELS")
print("Compare predictions from different model architectures")
print("="*70)

# ============================================================================
# Model Definitions
# ============================================================================

class MultiTaskVGG16(nn.Module):
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
        return self.magnitude_head(x), self.azimuth_head(x)


class MultiTaskEfficientNet(nn.Module):
    def __init__(self, num_magnitude_classes=4, num_azimuth_classes=9, dropout_rate=0.3):
        super(MultiTaskEfficientNet, self).__init__()
        base_model = models.efficientnet_b0(pretrained=False)
        feature_dim = base_model.classifier[1].in_features
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


# ============================================================================
# Helper Functions
# ============================================================================

def load_model(model_path, model_class, num_mag, num_azi):
    """Load a model from checkpoint"""
    model = model_class(num_mag, num_azi)
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def generate_spectrogram(data):
    """Generate spectrogram from geomagnetic data"""
    signal_processor = GeomagneticSignalProcessor(sampling_rate=1.0)
    pc3_low = 0.01
    pc3_high = 0.045
    
    components_data = {}
    for comp_name in ['Hcomp', 'Dcomp', 'Zcomp']:
        signal_data = data[comp_name]
        valid_mask = ~np.isnan(signal_data)
        signal_clean = np.array(signal_data, dtype=float)
        if np.any(~valid_mask):
            x = np.arange(len(signal_data))
            signal_clean[~valid_mask] = np.interp(
                x[~valid_mask], x[valid_mask], signal_data[valid_mask]
            )
        signal_filtered = signal_processor.bandpass_filter(
            signal_clean, low_freq=pc3_low, high_freq=pc3_high
        )
        components_data[comp_name] = signal_filtered
    
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
    
    fig, axes = plt.subplots(3, 1, figsize=(2.24, 2.24))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    
    f_h, t_h, Sxx_h_db = spectrograms_db['Hcomp']
    f_d, t_d, Sxx_d_db = spectrograms_db['Dcomp']
    f_z, t_z, Sxx_z_db = spectrograms_db['Zcomp']
    
    freq_mask = (f_h >= pc3_low) & (f_h <= pc3_high)
    f_pc3 = f_h[freq_mask]
    
    axes[0].pcolormesh(t_h, f_pc3, Sxx_h_db[freq_mask, :], shading='gouraud', cmap='jet')
    axes[0].axis('off')
    axes[1].pcolormesh(t_d, f_pc3, Sxx_d_db[freq_mask, :], shading='gouraud', cmap='jet')
    axes[1].axis('off')
    axes[2].pcolormesh(t_z, f_pc3, Sxx_z_db[freq_mask, :], shading='gouraud', cmap='jet')
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


def predict_with_model(model, spectrogram_img, class_mappings):
    """Run prediction with a model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(spectrogram_img).unsqueeze(0)
    
    with torch.no_grad():
        mag_output, azi_output = model(img_tensor)
        mag_probs = torch.softmax(mag_output, dim=1)[0]
        azi_probs = torch.softmax(azi_output, dim=1)[0]
        
        mag_pred_idx = torch.argmax(mag_probs).item()
        azi_pred_idx = torch.argmax(azi_probs).item()
        
        mag_classes = class_mappings.get('magnitude_classes', [])
        azi_classes = class_mappings.get('azimuth_classes', [])
        
        mag_pred = mag_classes[mag_pred_idx] if mag_pred_idx < len(mag_classes) else f'Class_{mag_pred_idx}'
        azi_pred = azi_classes[azi_pred_idx] if azi_pred_idx < len(azi_classes) else f'Class_{azi_pred_idx}'
        
        mag_conf = mag_probs[mag_pred_idx].item() * 100
        azi_conf = azi_probs[azi_pred_idx].item() * 100
    
    return {
        'magnitude': mag_pred,
        'magnitude_conf': mag_conf,
        'azimuth': azi_pred,
        'azimuth_conf': azi_conf,
        'mag_probs': mag_probs.numpy(),
        'azi_probs': azi_probs.numpy(),
        'mag_classes': mag_classes,
        'azi_classes': azi_classes
    }


# ============================================================================
# Main Test
# ============================================================================

print("\n📡 Fetching data for TRT 2026-02-02...")

try:
    with GeomagneticDataFetcher() as fetcher:
        data = fetcher.fetch_data('2026-02-02', 'TRT')
    
    if data is None:
        print("❌ Failed to fetch data")
        sys.exit(1)
    
    print(f"✅ Data fetched: {data['stats']['coverage']:.1f}% coverage")
    
    # Generate spectrogram
    print("\n🎨 Generating spectrogram...")
    spectrogram = generate_spectrogram(data)
    print(f"✅ Spectrogram generated")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test Model 1: experiments_fixed (VGG16)
# ============================================================================

print("\n" + "="*70)
print("MODEL 1: experiments_fixed (VGG16)")
print("="*70)

model1_path = Path('experiments_fixed/exp_fixed_20260202_163643/best_model.pth')
mapping1_path = Path('experiments_fixed/exp_fixed_20260202_163643/class_mappings.json')

if model1_path.exists() and mapping1_path.exists():
    with open(mapping1_path, 'r') as f:
        mappings1 = json.load(f)
    
    num_mag1 = len(mappings1['magnitude_classes'])
    num_azi1 = len(mappings1['azimuth_classes'])
    
    print(f"Loading model: {model1_path}")
    print(f"Classes: {num_mag1} magnitude, {num_azi1} azimuth")
    print(f"Magnitude classes: {mappings1['magnitude_classes']}")
    print(f"Azimuth classes: {mappings1['azimuth_classes']}")
    
    model1 = load_model(model1_path, MultiTaskVGG16, num_mag1, num_azi1)
    result1 = predict_with_model(model1, spectrogram, mappings1)
    
    print(f"\n🔮 PREDICTION:")
    print(f"   Magnitude: {result1['magnitude']} ({result1['magnitude_conf']:.1f}%)")
    print(f"   Azimuth: {result1['azimuth']} ({result1['azimuth_conf']:.1f}%)")
    
    print(f"\n📊 All magnitude probabilities:")
    for i, (cls, prob) in enumerate(zip(result1['mag_classes'], result1['mag_probs'])):
        marker = " ← PREDICTED" if cls == result1['magnitude'] else ""
        expected = " ← EXPECTED" if cls == 'Large' else ""
        print(f"   {cls}: {prob*100:.1f}%{marker}{expected}")
    
    print(f"\n📊 All azimuth probabilities:")
    for i, (cls, prob) in enumerate(zip(result1['azi_classes'], result1['azi_probs'])):
        marker = " ← PREDICTED" if cls == result1['azimuth'] else ""
        expected = " ← EXPECTED" if cls == 'SW' else ""
        print(f"   {cls}: {prob*100:.1f}%{marker}{expected}")
else:
    print("❌ Model 1 not found")

# ============================================================================
# Test Model 2: final_production_model (EfficientNet)
# ============================================================================

print("\n" + "="*70)
print("MODEL 2: final_production_model (EfficientNet)")
print("="*70)

model2_path = Path('final_production_model/best_final_model.pth')
mapping2_path = Path('final_production_model/class_mappings.json')

if model2_path.exists() and mapping2_path.exists():
    with open(mapping2_path, 'r') as f:
        mappings2 = json.load(f)
    
    num_mag2 = len(mappings2['magnitude_classes'])
    num_azi2 = len(mappings2['azimuth_classes'])
    
    print(f"Loading model: {model2_path}")
    print(f"Classes: {num_mag2} magnitude, {num_azi2} azimuth")
    print(f"Magnitude classes: {mappings2['magnitude_classes']}")
    print(f"Azimuth classes: {mappings2['azimuth_classes']}")
    
    try:
        model2 = load_model(model2_path, MultiTaskEfficientNet, num_mag2, num_azi2)
        result2 = predict_with_model(model2, spectrogram, mappings2)
        
        print(f"\n🔮 PREDICTION:")
        print(f"   Magnitude: {result2['magnitude']} ({result2['magnitude_conf']:.1f}%)")
        print(f"   Azimuth: {result2['azimuth']} ({result2['azimuth_conf']:.1f}%)")
        
        print(f"\n📊 All magnitude probabilities:")
        for i, (cls, prob) in enumerate(zip(result2['mag_classes'], result2['mag_probs'])):
            marker = " ← PREDICTED" if cls == result2['magnitude'] else ""
            expected = " ← EXPECTED" if cls == 'Large' else ""
            print(f"   {cls}: {prob*100:.1f}%{marker}{expected}")
        
        print(f"\n📊 All azimuth probabilities:")
        for i, (cls, prob) in enumerate(zip(result2['azi_classes'], result2['azi_probs'])):
            marker = " ← PREDICTED" if cls == result2['azimuth'] else ""
            expected = " ← EXPECTED" if cls == 'SW' else ""
            print(f"   {cls}: {prob*100:.1f}%{marker}{expected}")
    except Exception as e:
        print(f"❌ Error loading model 2: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ Model 2 not found")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("""
Expected result for TRT 2026-02-02:
- Magnitude: Large (M6.4 earthquake)
- Azimuth: SW (Southwest direction)

The low confidence and incorrect predictions indicate:
1. Severe class imbalance in training data
2. Model bias toward majority classes (Normal, Medium)
3. Insufficient Large earthquake samples (only 20 out of 1384)

RECOMMENDED SOLUTIONS:
1. Retrain with stronger class weights for Large class
2. Use focal loss to focus on hard examples
3. Augment Large class samples
4. Consider binary classification first (Earthquake vs Normal)
""")

print("="*70)
