#!/usr/bin/env python3
"""
Prekursor Scanner Module for VPS Deployment
Handles SSH connection to BMKG server and spectrogram generation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import io

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal

try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False
    
from config import SSH_CONFIG, MODEL_CONFIG, SCANNER_CONFIG

# Setup logging
logger = logging.getLogger(__name__)


class EarthquakeCNN(nn.Module):
    """EfficientNet-B0 based model for earthquake precursor detection"""
    
    def __init__(self, num_mag_classes=4, num_azi_classes=9):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.mag_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_mag_classes)
        )
        
        self.azi_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_azi_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        mag_out = self.mag_head(features)
        azi_out = self.azi_head(features)
        return mag_out, azi_out


class PrekursorScanner:
    """Scanner for earthquake precursor detection"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_mappings = None
        self.ssh_client = None
        self.sftp = None
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self._load_model()
        self._load_class_mappings()
    
    def _load_model(self):
        """Load the trained model"""
        model_path = MODEL_CONFIG["model_path"]
        
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}")
            return
        
        try:
            self.model = EarthquakeCNN(
                num_mag_classes=MODEL_CONFIG["num_mag_classes"],
                num_azi_classes=MODEL_CONFIG["num_azi_classes"]
            )
            
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def _load_class_mappings(self):
        """Load class mappings"""
        mappings_path = MODEL_CONFIG["class_mappings_path"]
        
        if mappings_path.exists():
            with open(mappings_path, 'r') as f:
                self.class_mappings = json.load(f)
        else:
            # Default mappings
            self.class_mappings = {
                "magnitude": {0: "Medium", 1: "Normal", 2: "Large", 3: "Moderate"},
                "azimuth": {0: "E", 1: "N", 2: "NE", 3: "NW", 4: "Normal", 
                           5: "S", 6: "SE", 7: "SW", 8: "W"}
            }
    
    def connect_ssh(self):
        """Establish SSH connection to BMKG server"""
        if not HAS_PARAMIKO:
            raise ImportError("paramiko is required for SSH connection")
        
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Try key-based auth first
            key_path = SSH_CONFIG.get("key_path")
            if key_path and Path(key_path).exists():
                self.ssh_client.connect(
                    hostname=SSH_CONFIG["hostname"],
                    port=SSH_CONFIG["port"],
                    username=SSH_CONFIG["username"],
                    key_filename=key_path,
                    timeout=SSH_CONFIG["timeout"]
                )
            else:
                # Fall back to password auth
                self.ssh_client.connect(
                    hostname=SSH_CONFIG["hostname"],
                    port=SSH_CONFIG["port"],
                    username=SSH_CONFIG["username"],
                    password=SSH_CONFIG["password"],
                    timeout=SSH_CONFIG["timeout"]
                )
            
            self.sftp = self.ssh_client.open_sftp()
            logger.info("SSH connection established")
            return True
            
        except Exception as e:
            logger.error(f"SSH connection failed: {e}")
            return False
    
    def disconnect_ssh(self):
        """Close SSH connection"""
        if self.sftp:
            self.sftp.close()
        if self.ssh_client:
            self.ssh_client.close()
        logger.info("SSH connection closed")
    
    def fetch_data_ssh(self, station: str, date: datetime) -> pd.DataFrame:
        """Fetch geomagnetic data from BMKG server via SSH"""
        if not self.sftp:
            if not self.connect_ssh():
                raise ConnectionError("Cannot establish SSH connection")
        
        # Construct file path on remote server
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        
        remote_path = f"{SSH_CONFIG['data_path']}/{station}/{year}/{month}/{station}_{year}{month}{day}.csv"
        
        try:
            with self.sftp.open(remote_path, 'r') as f:
                df = pd.read_csv(f)
            logger.info(f"Data fetched: {remote_path}")
            return df
        except FileNotFoundError:
            logger.warning(f"File not found: {remote_path}")
            return None
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def generate_spectrogram(self, data: np.ndarray, fs: float = 1.0) -> np.ndarray:
        """Generate spectrogram from time series data"""
        params = SCANNER_CONFIG["spectrogram_params"]
        
        f, t, Sxx = signal.spectrogram(
            data,
            fs=fs,
            nperseg=params["nperseg"],
            noverlap=params["noverlap"],
            nfft=params["nfft"]
        )
        
        # Filter frequency range
        freq_mask = (f >= params["freq_min"]) & (f <= params["freq_max"])
        f_filtered = f[freq_mask]
        Sxx_filtered = Sxx[freq_mask, :]
        
        return f_filtered, t, Sxx_filtered
    
    def create_spectrogram_image(self, f, t, Sxx) -> Image.Image:
        """Create spectrogram image for model input"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot spectrogram
        im = ax.pcolormesh(t/3600, f*1000, 10*np.log10(Sxx + 1e-10),
                          shading='gouraud', cmap='jet')
        
        ax.set_ylabel('Frequency (mHz)')
        ax.set_xlabel('Time (hours)')
        ax.set_title('ULF Spectrogram')
        plt.colorbar(im, ax=ax, label='Power (dB)')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        return Image.open(buf).convert('RGB')
    
    def predict(self, image: Image.Image) -> dict:
        """Run prediction on spectrogram image"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Transform image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            mag_out, azi_out = self.model(img_tensor)
            
            mag_probs = torch.softmax(mag_out, dim=1)
            azi_probs = torch.softmax(azi_out, dim=1)
            
            mag_pred = torch.argmax(mag_probs, dim=1).item()
            azi_pred = torch.argmax(azi_probs, dim=1).item()
            
            mag_conf = mag_probs[0, mag_pred].item()
            azi_conf = azi_probs[0, azi_pred].item()
        
        # Map to class names
        mag_class = self.class_mappings["magnitude"].get(str(mag_pred), f"Class_{mag_pred}")
        azi_class = self.class_mappings["azimuth"].get(str(azi_pred), f"Class_{azi_pred}")
        
        return {
            "magnitude": {
                "class": mag_class,
                "confidence": mag_conf,
                "probabilities": mag_probs[0].cpu().numpy().tolist()
            },
            "azimuth": {
                "class": azi_class,
                "confidence": azi_conf,
                "probabilities": azi_probs[0].cpu().numpy().tolist()
            },
            "is_precursor": mag_class != "Normal" and azi_class != "Normal"
        }
    
    def scan(self, station: str, date: datetime) -> dict:
        """Full scan pipeline: fetch data, generate spectrogram, predict"""
        results = {
            "station": station,
            "date": date.strftime("%Y-%m-%d"),
            "status": "pending",
            "prediction": None,
            "spectrogram": None,
            "error": None
        }
        
        try:
            # Fetch data
            df = self.fetch_data_ssh(station, date)
            if df is None:
                results["status"] = "error"
                results["error"] = "Data not available"
                return results
            
            # Extract H component (or use available column)
            if 'H' in df.columns:
                data = df['H'].values
            elif 'h' in df.columns:
                data = df['h'].values
            else:
                data = df.iloc[:, 1].values  # Use second column
            
            # Generate spectrogram
            f, t, Sxx = self.generate_spectrogram(data)
            spec_image = self.create_spectrogram_image(f, t, Sxx)
            results["spectrogram"] = spec_image
            
            # Predict
            prediction = self.predict(spec_image)
            results["prediction"] = prediction
            results["status"] = "success"
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            logger.error(f"Scan error: {e}")
        
        return results
    
    def scan_from_file(self, file_path: str) -> dict:
        """Scan from local spectrogram file"""
        results = {
            "file": file_path,
            "status": "pending",
            "prediction": None,
            "error": None
        }
        
        try:
            image = Image.open(file_path).convert('RGB')
            prediction = self.predict(image)
            results["prediction"] = prediction
            results["status"] = "success"
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            logger.error(f"File scan error: {e}")
        
        return results


# Singleton instance
_scanner_instance = None

def get_scanner() -> PrekursorScanner:
    """Get or create scanner instance"""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = PrekursorScanner()
    return _scanner_instance
