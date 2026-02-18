#!/usr/bin/env python3
"""
Test Augmented Model - Verify improvement for Large class detection
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
print("TEST AUGMENTED MODEL")
print("Comparing OLD vs NEW model for TRT 2026-02-02 (M6.4 SW)")
print("="*70)

# Model definition
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


def generate_spectrogram(data):
    """Generate spectrogram"""
    signal_processor = GeomagneticSignalProcessor(sampling_rate=1.0)
    pc3_low, pc3_high = 0.01, 0.045
    
    components_data = {}
    for comp_name in ['Hcomp', 'Dcomp', 'Zcomp']:
        signal_data = data[comp_name]
        valid_mask = ~np.isnan(signal_data)
        signal_clean = np.array(signal_data, dtype=float)
        if np.any(~valid_mask):
            x = np.arange(len(signal_data))
            signal_clean[~valid_mask] = np.interp(x[~valid_mask], x[valid_mask], signal_data[valid_mask])
        signal_filtered = signal_processor.bandpass_filter(signal_clean, low_freq=pc3_low, high_freq=pc3_high)
        components_data[comp_name] = signal_filtered
    
    fs, nperseg, noverlap = 1.0, 256, 128
    spectrograms_db = {}
    for comp_name, sig in components_data.items():
        f, t, Sxx = signal.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
        spectrograms_db[comp_name] = (f, t, 10 * np.log10(Sxx + 1e-10))
    
    fig, axes = plt.subplots(3, 1, figsize=(2.24, 2.24))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
    
    f_h = spectrograms_db['Hcomp'][0]
    freq_mask = (f_h >= pc3_low) & (f_h <= pc3_high)
    
    for ax, comp in zip(axes, ['Hcomp', 'Dcomp', 'Zcomp']):
        f, t, Sxx_db = spectrograms_db[comp]
        ax.pcolormesh(t, f[freq_mask], Sxx_db[freq_mask, :], shading='gouraud', cmap='jet')
        ax.axis('off')
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    plt.savefig(tmp_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    img = Image.open(tmp_path).resize((224, 224), Image.LANCZOS).convert('RGB')
    os.unlink(tmp_path)
    return img


def predict(model, img, mappings):
    """Run prediction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        tensor = transform(img).unsqueeze(0)
        mag_out, azi_out = model(tensor)
        mag_probs = torch.softmax(mag_out, dim=1)[0].numpy()
        azi_probs = torch.softmax(azi_out, dim=1)[0].numpy()
    
    return mag_probs, azi_probs


# Fetch data
print("\n📡 Fetching TRT 2026-02-02 data...")
with GeomagneticDataFetcher() as fetcher:
    data = fetcher.fetch_data('2026-02-02', 'TRT')
print(f"✅ Data fetched: {data['stats']['coverage']:.1f}% coverage")

# Generate spectrogram
print("🎨 Generating spectrogram...")
spectrogram = generate_spectrogram(data)
print("✅ Spectrogram generated")

# ============================================================================
# TEST OLD MODEL (final_production_model)
# ============================================================================
print("\n" + "="*70)
print("OLD MODEL: final_production_model (EfficientNet)")
print("="*70)

old_model_path = Path('final_production_model/best_final_model.pth')
old_mapping_path = Path('final_production_model/class_mappings.json')

with open(old_mapping_path, 'r') as f:
    old_mappings = json.load(f)

old_model = MultiTaskEfficientNet(4, 9)
checkpoint = torch.load(old_model_path, map_location='cpu')
if 'model_state_dict' in checkpoint:
    old_model.load_state_dict(checkpoint['model_state_dict'])
else:
    old_model.load_state_dict(checkpoint)
old_model.eval()

old_mag_probs, old_azi_probs = predict(old_model, spectrogram, old_mappings)

print(f"\n📊 MAGNITUDE PROBABILITIES:")
for i, (cls, prob) in enumerate(zip(old_mappings['magnitude_classes'], old_mag_probs)):
    marker = " ← EXPECTED" if cls == 'Large' else ""
    print(f"   {cls}: {prob*100:.1f}%{marker}")

print(f"\n📊 AZIMUTH PROBABILITIES:")
for i, (cls, prob) in enumerate(zip(old_mappings['azimuth_classes'], old_azi_probs)):
    marker = " ← EXPECTED" if cls == 'SW' else ""
    print(f"   {cls}: {prob*100:.1f}%{marker}")

# ============================================================================
# TEST NEW MODEL (experiments_augmented)
# ============================================================================
print("\n" + "="*70)
print("NEW MODEL: experiments_augmented (with Focal Loss + Oversampling)")
print("="*70)

new_model_path = Path('experiments_augmented/exp_aug_20260210_161337/best_model.pth')
new_mapping_path = Path('experiments_augmented/exp_aug_20260210_161337/class_mappings.json')

if new_model_path.exists():
    with open(new_mapping_path, 'r') as f:
        new_mappings = json.load(f)
    
    new_model = MultiTaskEfficientNet(4, 9, dropout_rate=0.4)
    checkpoint = torch.load(new_model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        new_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        new_model.load_state_dict(checkpoint)
    new_model.eval()
    
    new_mag_probs, new_azi_probs = predict(new_model, spectrogram, new_mappings)
    
    print(f"\n📊 MAGNITUDE PROBABILITIES:")
    for i, (cls, prob) in enumerate(zip(new_mappings['magnitude_classes'], new_mag_probs)):
        marker = " ← EXPECTED" if cls == 'Large' else ""
        print(f"   {cls}: {prob*100:.1f}%{marker}")
    
    print(f"\n📊 AZIMUTH PROBABILITIES:")
    for i, (cls, prob) in enumerate(zip(new_mappings['azimuth_classes'], new_azi_probs)):
        marker = " ← EXPECTED" if cls == 'SW' else ""
        print(f"   {cls}: {prob*100:.1f}%{marker}")
    
    # ============================================================================
    # COMPARISON
    # ============================================================================
    print("\n" + "="*70)
    print("COMPARISON: OLD vs NEW")
    print("="*70)
    
    # Find Large index in each mapping
    old_large_idx = old_mappings['magnitude_classes'].index('Large')
    new_large_idx = new_mappings['magnitude_classes'].index('Large')
    
    old_sw_idx = old_mappings['azimuth_classes'].index('SW')
    new_sw_idx = new_mappings['azimuth_classes'].index('SW')
    
    old_large_prob = old_mag_probs[old_large_idx] * 100
    new_large_prob = new_mag_probs[new_large_idx] * 100
    
    old_sw_prob = old_azi_probs[old_sw_idx] * 100
    new_sw_prob = new_azi_probs[new_sw_idx] * 100
    
    print(f"\n🎯 LARGE CLASS (Expected for M6.4):")
    print(f"   OLD Model: {old_large_prob:.1f}%")
    print(f"   NEW Model: {new_large_prob:.1f}%")
    improvement_large = new_large_prob - old_large_prob
    if improvement_large > 0:
        print(f"   ✅ IMPROVEMENT: +{improvement_large:.1f}%")
    else:
        print(f"   ❌ REGRESSION: {improvement_large:.1f}%")
    
    print(f"\n🎯 SW CLASS (Expected direction):")
    print(f"   OLD Model: {old_sw_prob:.1f}%")
    print(f"   NEW Model: {new_sw_prob:.1f}%")
    improvement_sw = new_sw_prob - old_sw_prob
    if improvement_sw > 0:
        print(f"   ✅ IMPROVEMENT: +{improvement_sw:.1f}%")
    else:
        print(f"   ❌ REGRESSION: {improvement_sw:.1f}%")
    
    # Overall assessment
    print(f"\n📊 OVERALL ASSESSMENT:")
    if new_large_prob > old_large_prob and new_large_prob >= 30:
        print(f"   ✅ Large class detection IMPROVED significantly!")
    elif new_large_prob > old_large_prob:
        print(f"   ⚠️  Large class detection improved but still below 30%")
    else:
        print(f"   ❌ Large class detection did not improve")
else:
    print("❌ New model not found!")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
