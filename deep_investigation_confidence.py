#!/usr/bin/env python3
"""
INVESTIGASI MENDALAM: Mengapa Confidence Magnitude Hanya 29%?

Gempa M6.4 seharusnya terdeteksi dengan confidence TINGGI (>80%)
Tapi hasil: Large 29% - INI MASALAH SERIUS!

Investigasi ini akan:
1. Analisis arsitektur model yang digunakan
2. Cek apakah model yang di-load benar
3. Bandingkan preprocessing training vs scanner SECARA DETAIL
4. Test dengan spectrogram training langsung
5. Identifikasi EXACT root cause

Author: Deep Investigation Team
Date: 10 February 2026
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

print("=" * 80)
print("🔬 INVESTIGASI MENDALAM: CONFIDENCE MAGNITUDE 29%")
print("=" * 80)

# ============================================================================
# BAGIAN 1: IDENTIFIKASI MODEL YANG DIGUNAKAN
# ============================================================================
print("\n" + "=" * 80)
print("🤖 BAGIAN 1: IDENTIFIKASI MODEL")
print("=" * 80)

# Cari semua model yang tersedia
model_candidates = [
    'experiments_fixed/exp_fixed_20260202_163643/best_model.pth',
    'experiments_v4/exp_v4_phase1_*/best_model.pth',
    'final_production_model/best_model.pth',
    'convnext_production_model/best_model.pth',
    'loeo_validation_results/fold_*/best_model.pth',
    'loeo_convnext_results/fold_*/best_model.pth',
    'mdata2/best_vgg16_model_phase1.keras',
]

print("\n🔍 Mencari model yang tersedia...")

import glob
found_models = []
for pattern in model_candidates:
    matches = glob.glob(pattern)
    for m in matches:
        if os.path.exists(m):
            size_mb = os.path.getsize(m) / (1024 * 1024)
            found_models.append((m, size_mb))
            print(f"   ✅ {m} ({size_mb:.1f} MB)")

if not found_models:
    print("   ❌ Tidak ada model ditemukan!")

# ============================================================================
# BAGIAN 2: ANALISIS ARSITEKTUR MODEL
# ============================================================================
print("\n" + "=" * 80)
print("🏗️  BAGIAN 2: ANALISIS ARSITEKTUR MODEL")
print("=" * 80)

# Load model dan cek arsitekturnya
model_path = 'experiments_fixed/exp_fixed_20260202_163643/best_model.pth'
if os.path.exists(model_path):
    print(f"\n📂 Loading model: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Cek isi checkpoint
        print(f"\n📋 Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Analisis layer names
        print(f"\n🔧 Model layers (first 20):")
        layer_names = list(state_dict.keys())[:20]
        for name in layer_names:
            shape = state_dict[name].shape
            print(f"   {name}: {shape}")
        
        # Cek output layers
        print(f"\n🎯 Output layers:")
        for name in state_dict.keys():
            if 'fc' in name.lower() or 'classifier' in name.lower() or 'head' in name.lower():
                shape = state_dict[name].shape
                print(f"   {name}: {shape}")
        
        # Cek config jika ada
        if 'config' in checkpoint:
            print(f"\n⚙️  Config: {checkpoint['config']}")
        
        # Cek training info
        if 'epoch' in checkpoint:
            print(f"\n📊 Training info:")
            print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"   Best val acc: {checkpoint.get('best_val_acc', 'N/A')}")
            
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")

# ============================================================================
# BAGIAN 3: CEK CLASS MAPPING DI MODEL
# ============================================================================
print("\n" + "=" * 80)
print("🗺️  BAGIAN 3: CEK CLASS MAPPING")
print("=" * 80)

# Cek class mapping dari training
config_path = 'experiments_fixed/exp_fixed_20260202_163643/config.json'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"\n✅ Config dari training:")
    print(json.dumps(config, indent=2))

class_mapping_path = 'experiments_fixed/exp_fixed_20260202_163643/class_mappings.json'
if os.path.exists(class_mapping_path):
    with open(class_mapping_path, 'r') as f:
        class_mappings = json.load(f)
    print(f"\n✅ Class mappings dari training:")
    print(json.dumps(class_mappings, indent=2))

# ============================================================================
# BAGIAN 4: ANALISIS DISTRIBUSI MAGNITUDE DI TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("📊 BAGIAN 4: DISTRIBUSI MAGNITUDE TRAINING")
print("=" * 80)

metadata_path = 'training_data/training_metadata.csv'
if os.path.exists(metadata_path):
    import pandas as pd
    df = pd.read_csv(metadata_path)
    
    mag_col = 'magnitude_class' if 'magnitude_class' in df.columns else 'magnitude'
    
    print(f"\n📈 Distribusi Magnitude:")
    mag_counts = df[mag_col].value_counts()
    total = len(df)
    for mag, count in mag_counts.items():
        pct = count / total * 100
        print(f"   {mag}: {count} samples ({pct:.1f}%)")
    
    # Cek apakah ada class "Large"
    if 'Large' in mag_counts.index:
        large_count = mag_counts['Large']
        print(f"\n⚠️  KRITIS: Class 'Large' hanya {large_count} samples ({large_count/total*100:.1f}%)!")
        print(f"   Ini SANGAT SEDIKIT untuk training yang baik!")
    else:
        print(f"\n❌ Class 'Large' TIDAK ADA di training data!")

# ============================================================================
# BAGIAN 5: TEST DENGAN SPECTROGRAM TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("🧪 BAGIAN 5: TEST DENGAN SPECTROGRAM TRAINING")
print("=" * 80)

# Cari spectrogram Large dari training
training_spec_dir = 'training_data/spectrograms/original'
if os.path.exists(training_spec_dir):
    # Cari file Large
    large_files = []
    if os.path.exists(metadata_path):
        large_df = df[df[mag_col] == 'Large']
        if len(large_df) > 0:
            print(f"\n📂 Spectrogram 'Large' dari training:")
            for _, row in large_df.head(5).iterrows():
                spec_file = row['spectrogram_file']
                spec_path = os.path.join(training_spec_dir, spec_file)
                if os.path.exists(spec_path):
                    print(f"   ✅ {spec_file}")
                    large_files.append(spec_path)
                else:
                    # Coba path alternatif
                    alt_path = row.get('unified_path', '')
                    if alt_path:
                        full_alt = os.path.join('training_data', alt_path)
                        if os.path.exists(full_alt):
                            print(f"   ✅ {alt_path}")
                            large_files.append(full_alt)
                        else:
                            print(f"   ❌ {spec_file} (not found)")
                    else:
                        print(f"   ❌ {spec_file} (not found)")

# ============================================================================
# BAGIAN 6: BANDINGKAN PREPROCESSING DETAIL
# ============================================================================
print("\n" + "=" * 80)
print("🔬 BAGIAN 6: PREPROCESSING COMPARISON DETAIL")
print("=" * 80)

# Load sample training spectrogram
if large_files:
    sample_training = large_files[0]
    print(f"\n📷 Analisis spectrogram training: {sample_training}")
    
    img = Image.open(sample_training)
    img_array = np.array(img)
    
    print(f"   Size: {img.size}")
    print(f"   Mode: {img.mode}")
    print(f"   Array shape: {img_array.shape}")
    print(f"   Dtype: {img_array.dtype}")
    print(f"   Min value: {img_array.min()}")
    print(f"   Max value: {img_array.max()}")
    print(f"   Mean value: {img_array.mean():.2f}")
    print(f"   Std value: {img_array.std():.2f}")
    
    # Analisis per channel
    if len(img_array.shape) == 3:
        print(f"\n   Per-channel statistics:")
        for i, ch in enumerate(['R', 'G', 'B']):
            ch_data = img_array[:,:,i]
            print(f"   {ch}: min={ch_data.min()}, max={ch_data.max()}, mean={ch_data.mean():.2f}, std={ch_data.std():.2f}")

# ============================================================================
# BAGIAN 7: SIMULASI INFERENCE DENGAN MODEL
# ============================================================================
print("\n" + "=" * 80)
print("🔮 BAGIAN 7: SIMULASI INFERENCE")
print("=" * 80)

try:
    # Import model
    sys.path.insert(0, '.')
    from earthquake_cnn_v3 import EarthquakeCNNV3
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Cek jumlah kelas dari checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Cari output layer untuk menentukan jumlah kelas
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Cari magnitude dan azimuth head
        num_mag_classes = None
        num_azi_classes = None
        
        for name, param in state_dict.items():
            if 'magnitude' in name.lower() and ('weight' in name or 'bias' in name):
                if 'weight' in name:
                    num_mag_classes = param.shape[0]
                    print(f"   Magnitude classes: {num_mag_classes} (from {name})")
            if 'azimuth' in name.lower() and ('weight' in name or 'bias' in name):
                if 'weight' in name:
                    num_azi_classes = param.shape[0]
                    print(f"   Azimuth classes: {num_azi_classes} (from {name})")
        
        if num_mag_classes and num_azi_classes:
            print(f"\n✅ Model architecture:")
            print(f"   Magnitude classes: {num_mag_classes}")
            print(f"   Azimuth classes: {num_azi_classes}")
            
            # Create model
            model = EarthquakeCNNV3(
                num_magnitude_classes=num_mag_classes,
                num_azimuth_classes=num_azi_classes,
                dropout_rate=0.3
            )
            
            # Load weights
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            print(f"\n✅ Model loaded successfully!")
            
            # Test dengan spectrogram training
            if large_files:
                print(f"\n🧪 Testing dengan spectrogram training 'Large'...")
                
                for spec_path in large_files[:3]:
                    img = Image.open(spec_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    if img.size != (224, 224):
                        img = img.resize((224, 224), Image.LANCZOS)
                    
                    img_array = np.array(img)
                    img_tensor = torch.from_numpy(img_array).float()
                    img_tensor = img_tensor.permute(2, 0, 1) / 255.0
                    img_tensor = img_tensor.unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        mag_logits, azi_logits = model(img_tensor)
                        mag_probs = torch.softmax(mag_logits, dim=1)
                        azi_probs = torch.softmax(azi_logits, dim=1)
                        
                        mag_pred = torch.argmax(mag_probs, dim=1).item()
                        mag_conf = mag_probs[0, mag_pred].item() * 100
                        
                        azi_pred = torch.argmax(azi_probs, dim=1).item()
                        azi_conf = azi_probs[0, azi_pred].item() * 100
                    
                    filename = os.path.basename(spec_path)
                    print(f"\n   📷 {filename}")
                    print(f"      Magnitude: class {mag_pred} ({mag_conf:.1f}%)")
                    print(f"      Azimuth: class {azi_pred} ({azi_conf:.1f}%)")
                    
                    # Print all magnitude probabilities
                    print(f"      All magnitude probs: ", end="")
                    for i in range(num_mag_classes):
                        print(f"[{i}]={mag_probs[0,i].item()*100:.1f}% ", end="")
                    print()
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# BAGIAN 8: ANALISIS MASALAH POTENSIAL
# ============================================================================
print("\n" + "=" * 80)
print("🎯 BAGIAN 8: ANALISIS MASALAH POTENSIAL")
print("=" * 80)

print("""
📋 KEMUNGKINAN PENYEBAB CONFIDENCE RENDAH:

1. CLASS IMBALANCE EKSTREM
   - Jika 'Large' hanya <5% dari training data
   - Model bias ke kelas mayoritas (Medium)
   - Confidence untuk Large akan selalu rendah

2. MODEL ARCHITECTURE MISMATCH
   - Scanner menggunakan model berbeda dari training
   - Jumlah kelas tidak sesuai
   - Weight tidak compatible

3. PREPROCESSING MISMATCH
   - Normalization berbeda (0-1 vs 0-255)
   - Color space berbeda (RGB vs BGR)
   - Resize method berbeda

4. SOFTMAX TEMPERATURE
   - Model terlalu "confident" atau "uncertain"
   - Perlu temperature scaling

5. TRAINING ISSUE
   - Model tidak converge dengan baik
   - Overfitting ke kelas mayoritas
   - Learning rate terlalu tinggi/rendah

6. DATA QUALITY
   - Spectrogram training berbeda dari inference
   - Noise atau artifact berbeda
""")

# ============================================================================
# BAGIAN 9: REKOMENDASI SOLUSI
# ============================================================================
print("\n" + "=" * 80)
print("💡 BAGIAN 9: REKOMENDASI SOLUSI")
print("=" * 80)

print("""
🔧 SOLUSI YANG DIREKOMENDASIKAN:

1. IMMEDIATE: Verifikasi Model
   - Pastikan model yang di-load adalah model yang benar
   - Cek jumlah kelas sesuai dengan training
   - Test dengan spectrogram training langsung

2. SHORT-TERM: Fix Class Imbalance
   - Gunakan class weights yang lebih agresif
   - Oversample kelas 'Large'
   - Undersample kelas 'Medium'

3. MEDIUM-TERM: Temperature Scaling
   - Calibrate model dengan temperature scaling
   - Adjust softmax temperature untuk confidence yang lebih baik

4. LONG-TERM: Retrain dengan Data Seimbang
   - Kumpulkan lebih banyak data 'Large'
   - Gunakan focal loss dengan gamma tinggi
   - Implementasi mixup/cutmix augmentation
""")

print("\n" + "=" * 80)
print("✅ INVESTIGASI MENDALAM SELESAI")
print("=" * 80)
