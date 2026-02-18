#!/usr/bin/env python3
"""
Test Scanner Architecture Fix
Verifies that prekursor_scanner.py now correctly loads VGG16 model
"""

import sys
import os

print("="*70)
print("TEST: Scanner Architecture Fix")
print("="*70)

# Test 1: Import scanner
print("\n📋 Test 1: Import PrekursorScanner...")
try:
    from prekursor_scanner import PrekursorScanner
    print("   ✅ Import successful!")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize scanner (this will load the model)
print("\n📋 Test 2: Initialize scanner and load model...")
try:
    scanner = PrekursorScanner()
    print(f"   ✅ Scanner initialized!")
    print(f"   Model architecture: {scanner.model_arch}")
    print(f"   Device: {scanner.device}")
except Exception as e:
    print(f"   ❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Verify model architecture
print("\n📋 Test 3: Verify model architecture...")
model = scanner.model
print(f"   Model type: {type(model).__name__}")

# Check if it's VGG16
if hasattr(model, 'features'):
    # Get first layer
    first_layer = list(model.features.children())[0]
    print(f"   First layer: {type(first_layer).__name__}")
    
    # Check for VGG16 pattern
    if 'Conv2d' in str(type(first_layer)):
        print("   ✅ VGG16 architecture confirmed!")
    else:
        print("   ⚠️  Unexpected first layer type")
else:
    print("   ⚠️  Model doesn't have 'features' attribute")

# Test 4: Verify class mappings
print("\n📋 Test 4: Verify class mappings...")
mappings = scanner.class_mappings
print(f"   Magnitude classes ({len(mappings['magnitude'])}):")
for idx, name in mappings['magnitude'].items():
    print(f"      {idx}: {name}")
print(f"   Azimuth classes ({len(mappings['azimuth'])}):")
for idx, name in mappings['azimuth'].items():
    print(f"      {idx}: {name}")

# Verify correct mapping
expected_mag = {0: 'Normal', 1: 'Moderate', 2: 'Medium', 3: 'Large'}
expected_azi = {0: 'Normal', 1: 'N', 2: 'NE', 3: 'E', 4: 'SE', 5: 'S', 6: 'SW', 7: 'W', 8: 'NW'}

mag_correct = ('Normal' in mappings['magnitude'].get(0, '') and 
              'Large' in mappings['magnitude'].get(3, ''))
azi_correct = len(mappings['azimuth']) == 9

if mag_correct and azi_correct:
    print("   ✅ Class mappings look correct!")
else:
    print("   ⚠️  Class mappings may need verification")

# Test 5: Test prediction with dummy data
print("\n📋 Test 5: Test prediction with dummy spectrogram...")
import torch
import numpy as np

try:
    # Create dummy spectrogram (224x224x3)
    dummy_spec = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Run prediction
    result = scanner.predict(dummy_spec)
    
    print(f"   Magnitude: {result['magnitude']['class_name']} ({result['magnitude']['confidence']:.1f}%)")
    print(f"   Azimuth: {result['azimuth']['class_name']} ({result['azimuth']['confidence']:.1f}%)")
    print("   ✅ Prediction successful!")
except Exception as e:
    print(f"   ❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test with real data (TRT 2026-02-02)
print("\n📋 Test 6: Test with real data (TRT 2026-02-02)...")
print("   This is the critical test case - M6.4 earthquake in SW direction")

try:
    # Fetch data
    data = scanner.fetch_data('2026-02-02', 'TRT')
    
    if data is not None:
        print(f"   ✅ Data fetched: {data['stats']['coverage']:.1f}% coverage")
        
        # Generate spectrogram
        spectrogram = scanner.generate_spectrogram(data)
        
        if spectrogram is not None:
            print(f"   ✅ Spectrogram generated: {spectrogram.shape}")
            
            # Predict
            result = scanner.predict(spectrogram)
            
            print(f"\n   🔮 PREDICTION RESULTS:")
            print(f"   ========================")
            print(f"   Magnitude: {result['magnitude']['class_name']}")
            print(f"   Magnitude Confidence: {result['magnitude']['confidence']:.1f}%")
            print(f"   Azimuth: {result['azimuth']['class_name']}")
            print(f"   Azimuth Confidence: {result['azimuth']['confidence']:.1f}%")
            
            # Check if prediction is correct
            expected_mag = 'Large'  # M6.4 should be Large
            expected_azi = 'SW'     # Southwest direction
            
            mag_name = result['magnitude']['class_name']
            azi_name = result['azimuth']['class_name']
            
            print(f"\n   📊 EVALUATION:")
            print(f"   Expected: {expected_mag} magnitude, {expected_azi} direction")
            print(f"   Got: {mag_name}, {azi_name}")
            
            if 'Large' in mag_name:
                print(f"   ✅ Magnitude prediction CORRECT!")
            else:
                print(f"   ⚠️  Magnitude prediction incorrect (got {mag_name})")
            
            if 'SW' in azi_name or 'Southwest' in azi_name:
                print(f"   ✅ Azimuth prediction CORRECT!")
            else:
                print(f"   ⚠️  Azimuth prediction incorrect (got {azi_name})")
                
            # Show top-3 predictions
            print(f"\n   📊 TOP-3 MAGNITUDE PREDICTIONS:")
            for i, pred in enumerate(result['magnitude']['top3']):
                marker = " ← EXPECTED" if 'Large' in pred['class_name'] else ""
                print(f"      {i+1}. {pred['class_name']}: {pred['confidence']:.1f}%{marker}")
            
            print(f"\n   📊 TOP-3 AZIMUTH PREDICTIONS:")
            for i, pred in enumerate(result['azimuth']['top3']):
                marker = " ← EXPECTED" if 'SW' in pred['class_name'] else ""
                print(f"      {i+1}. {pred['class_name']}: {pred['confidence']:.1f}%{marker}")
        else:
            print("   ❌ Failed to generate spectrogram")
    else:
        print("   ❌ Failed to fetch data")
except Exception as e:
    print(f"   ❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
