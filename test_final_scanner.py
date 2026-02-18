#!/usr/bin/env python3
"""Test final scanner with augmented model"""

from prekursor_scanner import PrekursorScanner

print("="*60)
print("TEST FINAL SCANNER WITH AUGMENTED MODEL")
print("="*60)

print("\nInitializing scanner...")
scanner = PrekursorScanner()
print(f"Model architecture: {scanner.model_arch}")

print("\nTesting prediction for TRT 2026-02-02 (M6.4 SW)...")
result = scanner.scan('2026-02-02', 'TRT', save_results=False)

if result:
    mag = result['predictions']['magnitude']
    azi = result['predictions']['azimuth']
    
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    print(f"Magnitude: {mag['class_name']} ({mag['confidence']:.1f}%)")
    print(f"Azimuth: {azi['class_name']} ({azi['confidence']:.1f}%)")
    print(f"Is Precursor: {result['predictions']['is_precursor']}")
    
    # Check if correct
    is_large = 'Large' in mag['class_name']
    is_sw = 'SW' in azi['class_name'] or 'Southwest' in azi['class_name']
    
    print("\n" + "="*60)
    print("EVALUATION:")
    print("="*60)
    print(f"Expected: Large (M6.4), SW direction")
    if is_large:
        print(f"✅ Magnitude CORRECT: {mag['class_name']}")
    else:
        print(f"⚠️  Magnitude: {mag['class_name']} (expected Large)")
    
    if is_sw:
        print(f"✅ Azimuth CORRECT: {azi['class_name']}")
    else:
        print(f"⚠️  Azimuth: {azi['class_name']} (expected SW)")
        # Show top-3
        print(f"\nTop-3 Azimuth predictions:")
        for i, pred in enumerate(azi.get('top3', [])):
            marker = " ← EXPECTED" if 'SW' in pred['class_name'] else ""
            print(f"   {i+1}. {pred['class_name']}: {pred['confidence']:.1f}%{marker}")
else:
    print("❌ Scan failed!")

print("\n" + "="*60)
