#!/usr/bin/env python3
"""
Test Prekursor Scanner
Quick test untuk memastikan scanner berfungsi dengan baik

Author: Earthquake Prediction Research Team
Date: 2 February 2026
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required modules can be imported"""
    print("\n" + "="*60)
    print("TEST 1: Checking imports...")
    print("="*60)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import pandas
        print(f"✅ Pandas: {pandas.__version__}")
    except ImportError as e:
        print(f"❌ Pandas: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib: {e}")
        return False
    
    try:
        import scipy
        print(f"✅ SciPy: {scipy.__version__}")
    except ImportError as e:
        print(f"❌ SciPy: {e}")
        return False
    
    try:
        import paramiko
        print(f"✅ Paramiko: {paramiko.__version__}")
    except ImportError as e:
        print(f"❌ Paramiko: {e}")
        return False
    
    print("\n✅ All imports successful!")
    return True


def test_files():
    """Test if all required files exist"""
    print("\n" + "="*60)
    print("TEST 2: Checking required files...")
    print("="*60)
    
    required_files = [
        'prekursor_scanner.py',
        'intial/lokasi_stasiun.csv',
        'intial/geomagnetic_fetcher.py',
        'intial/signal_processing.py',
        'earthquake_cnn_v3.py',
        'PANDUAN_PREKURSOR_SCANNER.md',
        'QUICK_START_SCANNER.txt',
        'run_scanner.bat'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("\n✅ All required files exist!")
    else:
        print("\n❌ Some files are missing!")
    
    return all_exist


def test_model():
    """Test if model can be loaded"""
    print("\n" + "="*60)
    print("TEST 3: Checking model...")
    print("="*60)
    
    # Check for model files
    exp_dir = Path('experiments_v4')
    if not exp_dir.exists():
        print("❌ experiments_v4 directory not found")
        print("   Please train model first")
        return False
    
    # Find latest experiment
    exp_folders = sorted(exp_dir.glob('exp_v4_phase1_*'))
    if not exp_folders:
        print("❌ No experiment folders found")
        print("   Please train model first")
        return False
    
    latest_exp = exp_folders[-1]
    model_path = latest_exp / 'best_model.pth'
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please train model first")
        return False
    
    print(f"✅ Model found: {model_path}")
    
    # Try to load model
    try:
        import torch
        from earthquake_cnn_v3 import EarthquakeCNNV3
        
        model = EarthquakeCNNV3(
            num_magnitude_classes=4,
            num_azimuth_classes=9,
            dropout_rate=0.3
        )
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False


def test_station_list():
    """Test if station list can be loaded"""
    print("\n" + "="*60)
    print("TEST 4: Checking station list...")
    print("="*60)
    
    try:
        import pandas as pd
        
        station_file = 'intial/lokasi_stasiun.csv'
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
        
        print(f"✅ Loaded {len(stations)} stations")
        print(f"   Stations: {', '.join(sorted(stations.keys())[:10])}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load station list: {e}")
        return False


def test_scanner_init():
    """Test if scanner can be initialized"""
    print("\n" + "="*60)
    print("TEST 5: Initializing scanner...")
    print("="*60)
    
    try:
        from prekursor_scanner import PrekursorScanner
        
        scanner = PrekursorScanner()
        
        print("✅ Scanner initialized successfully!")
        print(f"   Device: {scanner.device}")
        print(f"   Stations: {len(scanner.stations)}")
        print(f"   Model: Loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize scanner: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("                  PREKURSOR SCANNER - TEST SUITE")
    print("="*70)
    
    results = {
        'imports': test_imports(),
        'files': test_files(),
        'model': test_model(),
        'stations': test_station_list(),
        'scanner': test_scanner_init()
    }
    
    # Summary
    print("\n" + "="*70)
    print("                         TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("                    ✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n🎉 Scanner is ready to use!")
        print("\nQuick start:")
        print("  python prekursor_scanner.py -i")
        print("  or")
        print("  run_scanner.bat")
    else:
        print("                    ❌ SOME TESTS FAILED")
        print("="*70)
        print("\n⚠️  Please fix the issues above before using scanner")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Train model first: python train_with_improvements_v4.py")
        print("  - Check file paths and permissions")
    
    print("\n")
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
