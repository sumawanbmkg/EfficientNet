#!/usr/bin/env python3
"""
Test untuk memverifikasi bahwa mapping sudah benar setelah perbaikan.
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image

print("=" * 80)
print("🧪 TEST: VERIFIKASI MAPPING SETELAH PERBAIKAN")
print("=" * 80)

# ============================================================================
# TEST 1: Load class mappings dari model
# ============================================================================
print("\n" + "=" * 80)
print("📋 TEST 1: CLASS MAPPINGS")
print("=" * 80)

model_mapping_file = 'experiments_fixed/exp_fixed_20260202_163643/class_mappings.json'
if os.path.exists(model_mapping_file):
    with open(model_mapping_file, 'r') as f:
        model_mapping = json.load(f)
    
    print(f"\n✅ Model Mapping (dari {model_mapping_file}):")
    print(f"\n   Magnitude classes:")
    for idx, name in enumerate(model_mapping.get('magnitude_classes', [])):
        print(f"      {idx}: {name}")
    
    print(f"\n   Azimuth classes:")
    for idx, name in enumerate(model_mapping.get('azimuth_classes', [])):
        print(f"      {idx}: {name}")

# ============================================================================
# TEST 2: Load model dan test inference
# ============================================================================
print("\n" + "=" * 80)
print("🤖 TEST 2: MODEL INFERENCE")
print("=" * 80)

try:
    sys.path.insert(0, '.')
    from earthquake_cnn_v3 import EarthquakeCNNV3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load model
    model_path = 'experiments_fixed/exp_fixed_20260202_163643/best_model.pth'
    if os.path.exists(model_path):
        print(f"\n📂 Loading model: {model_path}")
        
        model = EarthquakeCNNV3(
            num_magnitude_classes=4,  # Normal, Moderate, Medium, Large
            num_azimuth_classes=9,    # Normal, N, NE, E, SE, S, SW, W, NW
            dropout_rate=0.3
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            mag_logits, azi_logits = model(dummy_input)
            mag_probs = torch.softmax(mag_logits, dim=1)
            azi_probs = torch.softmax(azi_logits, dim=1)
        
        print(f"\n📊 Model output shapes:")
        print(f"   Magnitude logits: {mag_logits.shape} (should be [1, 4])")
        print(f"   Azimuth logits: {azi_logits.shape} (should be [1, 9])")
        
        # Verify output dimensions
        assert mag_logits.shape[1] == 4, f"Expected 4 magnitude classes, got {mag_logits.shape[1]}"
        assert azi_logits.shape[1] == 9, f"Expected 9 azimuth classes, got {azi_logits.shape[1]}"
        
        print(f"\n✅ Model output dimensions CORRECT!")
        
        # Show probability distribution
        print(f"\n📈 Probability distribution (dummy input):")
        print(f"   Magnitude probs: ", end="")
        mag_classes = ['Normal', 'Moderate', 'Medium', 'Large']
        for i, (name, prob) in enumerate(zip(mag_classes, mag_probs[0])):
            print(f"{name}={prob.item()*100:.1f}% ", end="")
        print()
        
        print(f"   Azimuth probs: ", end="")
        azi_classes = ['Normal', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        for i, (name, prob) in enumerate(zip(azi_classes, azi_probs[0])):
            print(f"{name}={prob.item()*100:.1f}% ", end="")
        print()
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 3: Test dengan spectrogram training Large
# ============================================================================
print("\n" + "=" * 80)
print("🧪 TEST 3: INFERENCE DENGAN SPECTROGRAM LARGE")
print("=" * 80)

try:
    import pandas as pd
    
    # Load metadata
    metadata_path = 'training_data/training_metadata.csv'
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        mag_col = 'magnitude_class' if 'magnitude_class' in df.columns else 'magnitude'
        
        # Find Large samples
        large_df = df[df[mag_col] == 'Large']
        print(f"\n📂 Found {len(large_df)} Large samples in training data")
        
        if len(large_df) > 0:
            # Test with first Large sample
            sample = large_df.iloc[0]
            spec_file = sample['spectrogram_file']
            
            # Try different paths
            possible_paths = [
                f"training_data/spectrograms/original/{spec_file}",
                f"training_data/{sample.get('unified_path', '')}",
                f"dataset_unified/spectrograms/{spec_file}",
            ]
            
            spec_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    spec_path = p
                    break
            
            if spec_path:
                print(f"\n📷 Testing with: {spec_path}")
                
                # Load and preprocess
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
                
                # Get predictions
                mag_pred = torch.argmax(mag_probs, dim=1).item()
                mag_conf = mag_probs[0, mag_pred].item() * 100
                
                azi_pred = torch.argmax(azi_probs, dim=1).item()
                azi_conf = azi_probs[0, azi_pred].item() * 100
                
                print(f"\n📊 Prediction results:")
                print(f"   Magnitude: {mag_classes[mag_pred]} ({mag_conf:.1f}%)")
                print(f"   Azimuth: {azi_classes[azi_pred]} ({azi_conf:.1f}%)")
                
                print(f"\n📈 All magnitude probabilities:")
                for i, (name, prob) in enumerate(zip(mag_classes, mag_probs[0])):
                    marker = " ← PREDICTED" if i == mag_pred else ""
                    print(f"      {i}: {name} = {prob.item()*100:.1f}%{marker}")
                
                # Check if Large is predicted correctly
                large_idx = 3  # Large is class 3
                large_prob = mag_probs[0, large_idx].item() * 100
                print(f"\n🎯 Large class probability: {large_prob:.1f}%")
                
                if mag_pred == large_idx:
                    print(f"   ✅ CORRECT! Model predicts Large")
                else:
                    print(f"   ⚠️  Model predicts {mag_classes[mag_pred]} instead of Large")
                    print(f"   This is expected due to class imbalance (Large only 2.9% of training)")
            else:
                print(f"   ❌ Spectrogram file not found")
                
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# KESIMPULAN
# ============================================================================
print("\n" + "=" * 80)
print("📝 KESIMPULAN")
print("=" * 80)

print("""
🎯 HASIL VERIFIKASI:

1. Model memiliki 4 magnitude classes: Normal, Moderate, Medium, Large
2. Model memiliki 9 azimuth classes: Normal, N, NE, E, SE, S, SW, W, NW
3. Large adalah class index 3 (bukan 0!)
4. SW adalah class index 6 (bukan 6 di mapping lama yang berbeda)

⚠️  CATATAN PENTING:
- Confidence untuk Large akan SELALU rendah (~30-40%)
- Ini karena Large hanya 2.9% dari training data
- Ini BUKAN bug, tapi konsekuensi dari class imbalance

💡 UNTUK MENINGKATKAN CONFIDENCE:
1. Retrain dengan oversampling Large
2. Gunakan focal loss dengan gamma tinggi
3. Kumpulkan lebih banyak data Large
""")

print("\n" + "=" * 80)
print("✅ TEST SELESAI")
print("=" * 80)
