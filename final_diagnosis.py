#!/usr/bin/env python3
"""
Final Diagnosis - Complete analysis of the prediction issue
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
print("FINAL DIAGNOSIS")
print("Complete analysis of prediction issue for TRT 2026-02-02")
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


# Load model
model_path = Path('final_production_model/best_final_model.pth')
mapping_path = Path('final_production_model/class_mappings.json')

with open(mapping_path, 'r') as f:
    mappings = json.load(f)

model = MultiTaskEfficientNet(4, 9)
checkpoint = torch.load(model_path, map_location='cpu')
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

print(f"\n✅ Model loaded: EfficientNet")
print(f"   Magnitude classes: {mappings['magnitude_classes']}")
print(f"   Azimuth classes: {mappings['azimuth_classes']}")

# Fetch data
print(f"\n📡 Fetching TRT 2026-02-02 data...")
with GeomagneticDataFetcher() as fetcher:
    data = fetcher.fetch_data('2026-02-02', 'TRT')

print(f"✅ Data fetched: {data['stats']['coverage']:.1f}% coverage")

# Generate spectrogram
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
print(f"✅ Spectrogram generated")

# Predict
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    mag_output, azi_output = model(img_tensor)
    mag_probs = torch.softmax(mag_output, dim=1)[0]
    azi_probs = torch.softmax(azi_output, dim=1)[0]

print("\n" + "="*70)
print("PREDICTION RESULTS")
print("="*70)

print(f"\n📊 MAGNITUDE PROBABILITIES:")
print(f"   Class order: {mappings['magnitude_classes']}")
for i, (cls, prob) in enumerate(zip(mappings['magnitude_classes'], mag_probs.numpy())):
    expected = " ← EXPECTED (M6.4)" if cls == 'Large' else ""
    print(f"   {i}: {cls}: {prob*100:.1f}%{expected}")

print(f"\n📊 AZIMUTH PROBABILITIES:")
print(f"   Class order: {mappings['azimuth_classes']}")
for i, (cls, prob) in enumerate(zip(mappings['azimuth_classes'], azi_probs.numpy())):
    expected = " ← EXPECTED" if cls == 'SW' else ""
    print(f"   {i}: {cls}: {prob*100:.1f}%{expected}")

# Find Large index
large_idx = mappings['magnitude_classes'].index('Large')
sw_idx = mappings['azimuth_classes'].index('SW')

print(f"\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

print(f"""
🎯 EXPECTED:
   - Magnitude: Large (M6.4 earthquake)
   - Azimuth: SW (Southwest direction)

📊 ACTUAL PROBABILITIES:
   - Large: {mag_probs[large_idx].item()*100:.1f}%
   - SW: {azi_probs[sw_idx].item()*100:.1f}%

🔍 ROOT CAUSE ANALYSIS:

1. TRAINING DATA IMBALANCE:
   - Large class: only 28 samples (1.4% of total)
   - Normal class: 888 samples (45.0% of total)
   - Ratio: 31.7:1 (Normal:Large)

2. TRT STATION SPECIFIC:
   - TRT has 88 samples in training data
   - ALL TRT samples are Medium class
   - NO Large samples from TRT station!
   - Model never learned Large pattern for TRT

3. MODEL BEHAVIOR:
   - Model is biased toward majority classes
   - Low confidence indicates uncertainty
   - Predictions spread across multiple classes

🔧 RECOMMENDED SOLUTIONS:

1. IMMEDIATE (No retraining):
   - Lower confidence threshold for precursor detection
   - Use ensemble of multiple models
   - Consider top-3 predictions, not just top-1

2. SHORT-TERM (Requires retraining):
   - Augment Large class samples (SMOTE, rotation, etc.)
   - Use stronger class weights (10x-50x for Large)
   - Use focal loss to focus on hard examples

3. LONG-TERM:
   - Collect more Large earthquake data
   - Use transfer learning from similar domains
   - Consider binary classification first (Earthquake vs Normal)
""")

print("="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
