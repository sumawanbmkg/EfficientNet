#!/usr/bin/env python3
"""
EARTHQUAKE PREDICTION DASHBOARD LAUNCHER
Python launcher script for cross-platform compatibility
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+"""
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"[INFO] Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_module(module_name):
    """Check if a Python module is installed"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n[INFO] Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements_dashboard.txt"
        ])
        print("[SUCCESS] Dependencies installed successfully\n")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] Failed to install dependencies")
        return False

def check_data_files():
    """Check if required data files exist"""
    files = [
        "experiments_v3/exp_v3_20260131_172406/training_history.csv",
        "experiments_v3/exp_v3_20260131_172406/config.json",
        "dataset_unified/metadata/unified_metadata.csv"
    ]
    
    missing_files = []
    for file_path in files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("\n[WARNING] Some data files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nDashboard will still run but some features may not work.")
        print("Please ensure data files are in the correct location.\n")
    else:
        print("[INFO] All data files found\n")

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("=" * 50)
    print("  EARTHQUAKE PREDICTION DASHBOARD")
    print("  BMKG & ITS Collaboration")
    print("=" * 50)
    print()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check if Streamlit is installed
    if not check_module('streamlit'):
        print("[WARNING] Streamlit is not installed")
        if not install_dependencies():
            return False
    else:
        print("[INFO] Streamlit is installed")
    
    # Check other required modules
    required_modules = ['plotly', 'pandas', 'numpy', 'matplotlib', 'seaborn']
    missing_modules = [m for m in required_modules if not check_module(m)]
    
    if missing_modules:
        print(f"[WARNING] Missing modules: {', '.join(missing_modules)}")
        if not install_dependencies():
            return False
    
    # Check data files
    check_data_files()
    
    # Run dashboard
    print("[INFO] Starting dashboard...")
    print("\nDashboard will open in your default browser at:")
    print("http://localhost:8501")
    print("\nPress Ctrl+C to stop the dashboard\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "project_dashboard.py"
        ])
    except KeyboardInterrupt:
        print("\n\n[INFO] Dashboard stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Failed to run dashboard: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_dashboard()
    sys.exit(0 if success else 1)
