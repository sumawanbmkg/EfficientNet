"""
Pre-Demo Checklist Script
Run this before professor demonstration to verify everything is ready

Author: Earthquake Prediction Research Team
Date: February 14, 2026
"""

import os
import sys
from pathlib import Path
import json

def check_model_files():
    """Check if model files exist"""
    print("\n" + "="*60)
    print("1. CHECKING MODEL FILES")
    print("="*60)
    
    models = {
        "Champion Phase 2.1": "experiments_v2/hierarchical/best_model.pth",
        "Experiment 3": "experiments_v2/experiment_3/best_model.pth"
    }
    
    all_ok = True
    for name, path in models.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {name}: {path} NOT FOUND")
            all_ok = False
    
    return all_ok

def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "="*60)
    print("2. CHECKING DEPENDENCIES")
    print("="*60)
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'plotly': 'Plotly'
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✅ {name}: Installed")
        except ImportError:
            print(f"❌ {name}: NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_dataset_files():
    """Check if dataset files exist"""
    print("\n" + "="*60)
    print("3. CHECKING DATASET FILES")
    print("="*60)
    
    datasets = {
        "Champion Metadata": "dataset_consolidation/metadata/metadata_final_phase21.csv",
        "Experiment 3 Metadata": "dataset_experiment_3/metadata_raw_exp3.csv",
        "Station Data": "mdata2/lokasi_stasiun.csv"
    }
    
    all_ok = True
    for name, path in datasets.items():
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} NOT FOUND")
            all_ok = False
    
    return all_ok

def check_inference_module():
    """Check if inference module is available"""
    print("\n" + "="*60)
    print("4. CHECKING INFERENCE MODULE")
    print("="*60)
    
    try:
        from custom_scanner_inference import load_model, preprocess_image, run_inference
        print("✅ custom_scanner_inference.py: Available")
        print("✅ Functions: load_model, preprocess_image, run_inference")
        return True
    except ImportError as e:
        print(f"❌ custom_scanner_inference.py: NOT FOUND or ERROR")
        print(f"   Error: {e}")
        return False

def check_demo_files():
    """Check if demo files exist"""
    print("\n" + "="*60)
    print("5. CHECKING DEMO FILES")
    print("="*60)
    
    demo_files = {
        "Demo Cases": "demo_sample_cases.json",
        "Demo Mode": "demo_mode.py",
        "Demo Guide": "DEMO_PROFESSOR_GUIDE.md"
    }
    
    all_ok = True
    for name, path in demo_files.items():
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} NOT FOUND")
            all_ok = False
    
    return all_ok

def test_model_loading():
    """Test if model can be loaded"""
    print("\n" + "="*60)
    print("6. TESTING MODEL LOADING")
    print("="*60)
    
    try:
        from custom_scanner_inference import load_model
        
        model_path = "experiments_v2/hierarchical/best_model.pth"
        if not os.path.exists(model_path):
            print(f"⚠️  Skipping test: Model file not found")
            return True
        
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, device='cpu')
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def check_memory():
    """Check available memory"""
    print("\n" + "="*60)
    print("7. CHECKING SYSTEM MEMORY")
    print("="*60)
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
        
        print(f"Total Memory: {total_gb:.1f} GB")
        print(f"Available Memory: {available_gb:.1f} GB")
        
        if available_gb >= 1.0:
            print("✅ Sufficient memory for demo (>1 GB available)")
            return True
        else:
            print("⚠️  Low memory (<1 GB available)")
            print("   Recommendation: Close other applications")
            return False
    except ImportError:
        print("⚠️  psutil not installed, skipping memory check")
        return True

def check_dashboard_file():
    """Check if dashboard file exists"""
    print("\n" + "="*60)
    print("8. CHECKING DASHBOARD FILE")
    print("="*60)
    
    if os.path.exists("project_dashboard_v3.py"):
        print("✅ project_dashboard_v3.py: Available")
        return True
    else:
        print("❌ project_dashboard_v3.py: NOT FOUND")
        return False

def generate_report():
    """Generate summary report"""
    print("\n" + "="*60)
    print("PRE-DEMO CHECKLIST SUMMARY")
    print("="*60)
    
    checks = [
        ("Model Files", check_model_files()),
        ("Dependencies", check_dependencies()),
        ("Dataset Files", check_dataset_files()),
        ("Inference Module", check_inference_module()),
        ("Demo Files", check_demo_files()),
        ("Model Loading Test", test_model_loading()),
        ("System Memory", check_memory()),
        ("Dashboard File", check_dashboard_file())
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    print(f"\nResults: {passed}/{total} checks passed")
    print("\nDetailed Results:")
    for name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")
    
    if passed == total:
        print("\n" + "="*60)
        print("🎉 ALL CHECKS PASSED!")
        print("="*60)
        print("\n✅ System is ready for professor demonstration!")
        print("\nNext steps:")
        print("  1. Run: streamlit run project_dashboard_v3.py")
        print("  2. Enable: Pre-load Model (in sidebar)")
        print("  3. Enable: Demo Mode (in sidebar)")
        print("  4. Navigate: Custom Scanner")
        print("  5. Enable: Use Real AI Model checkbox")
        print("  6. Start: Demonstration!")
    else:
        print("\n" + "="*60)
        print("⚠️  SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before demonstration.")
        print("\nCommon fixes:")
        print("  • Install missing packages: pip install -r requirements.txt")
        print("  • Verify model files are in correct location")
        print("  • Check dataset files exist")
        print("  • Close other applications to free memory")
    
    return passed == total

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PRE-DEMO CHECKLIST FOR PROFESSOR DEMONSTRATION")
    print("="*60)
    print("This script will verify that everything is ready for demo")
    print("="*60)
    
    success = generate_report()
    
    print("\n" + "="*60)
    if success:
        print("Status: ✅ READY FOR DEMO")
    else:
        print("Status: ❌ NOT READY - FIX ISSUES ABOVE")
    print("="*60)
    
    sys.exit(0 if success else 1)
