#!/usr/bin/env python3
"""
Configuration for Earthquake Precursor Dashboard
Edit this file to customize your deployment
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
APP_DIR = BASE_DIR / "app"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if not exist
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_CONFIG = {
    "model_path": MODELS_DIR / "best_final_model.pth",
    "class_mappings_path": MODELS_DIR / "class_mappings.json",
    "model_type": "efficientnet_b0",
    "input_size": (224, 224),
    "num_mag_classes": 4,
    "num_azi_classes": 9,
}

# =============================================================================
# SSH CONFIGURATION (untuk akses data BMKG)
# =============================================================================
# PENTING: Ganti dengan credentials Anda
# Untuk keamanan, gunakan environment variables
SSH_CONFIG = {
    "hostname": os.environ.get("BMKG_SSH_HOST", "your-bmkg-server.go.id"),
    "port": int(os.environ.get("BMKG_SSH_PORT", 22)),
    "username": os.environ.get("BMKG_SSH_USER", "your_username"),
    "password": os.environ.get("BMKG_SSH_PASS", None),  # Prefer key auth
    "key_path": os.environ.get("BMKG_SSH_KEY", str(BASE_DIR / "ssh_keys" / "id_rsa")),
    "data_path": "/path/to/geomagnetic/data",
    "timeout": 30,
}

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================
DASHBOARD_CONFIG = {
    "title": "Earthquake Precursor Detection Dashboard",
    "version": "2.1",
    "port": int(os.environ.get("DASHBOARD_PORT", 8501)),
    "host": "0.0.0.0",  # Listen on all interfaces
    "debug": os.environ.get("DEBUG", "false").lower() == "true",
}

# =============================================================================
# SCANNER CONFIGURATION
# =============================================================================
SCANNER_CONFIG = {
    "stations": ["SCN", "MLB", "GTO", "TRD", "PLU", "GSI", "AMB", "ALR"],
    "default_station": "SCN",
    "hours_before": 24,
    "spectrogram_params": {
        "nperseg": 3600,
        "noverlap": 1800,
        "nfft": 4096,
        "freq_min": 0.001,
        "freq_max": 0.1,
    },
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "dashboard.log",
    "max_bytes": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5,
}

# =============================================================================
# VALIDATION RESULTS PATHS
# =============================================================================
VALIDATION_PATHS = {
    "loeo_results": ASSETS_DIR / "loeo_validation_results",
    "loso_results": ASSETS_DIR / "loso_validation_results",
    "gradcam_results": ASSETS_DIR / "gradcam_comparison",
    "paper_figures": ASSETS_DIR / "paper_figures",
    "q1_report": ASSETS_DIR / "q1_comprehensive_report",
}

# =============================================================================
# MODEL METRICS (untuk display)
# =============================================================================
MODEL_METRICS = {
    "magnitude_accuracy": 98.94,
    "azimuth_accuracy": 83.92,
    "loeo_magnitude": 97.53,
    "loeo_azimuth": 69.51,
    "loso_magnitude": 97.57,
    "loso_azimuth": 69.73,
    "model_size_mb": 20,
    "inference_time_ms": 50,
}
