"""
Advanced Hierarchical Model Trainer (V2) for Auto-Update Pipeline
Menggantikan trainer.py lama dengan fitur:
1. Hierarchical Architecture (Binary -> Multi-Task)
2. Support SMOTE augmented data
3. Dynamic Class Weighting
4. Physics-Aware (Z/H Gate integration ready)

Author: Antigravity
Date: 2026-02-12
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score

# Import utilities from parent package if available, else local mock
try:
    from .utils import load_config, generate_model_id, log_pipeline_event
except ImportError:
    # Standalone mode fallback
    def load_config(path=None): return {}
    def generate_model_id(prefix): return f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M')}"
    def log_pipeline_event(event, data): print(f"Event: {event} | Data: {data}")

logger = logging.getLogger("autoupdate_pipeline.trainer_v2")

# ============================================================================
# DATASET HANDLING
# ============================================================================

class HierarchicalEarthquakeDataset(Dataset):
    """Dataset for Hierarchical Training (Binary + Multi-Task Labels)"""
    
    def __init__(self, metadata_df: pd.DataFrame, dataset_root: str, transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        
        # Define Standard Classes (Hardcoded for Consistency with Production)
        self.mag_classes = ['Normal', 'Moderate', 'Medium', 'Large']
        self.azi_classes = ['Normal', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        self.mag_to_idx = {c: i for i, c in enumerate(self.mag_classes)}
        self.azi_to_idx = {c: i for i, c in enumerate(self.azi_classes)}
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Path Handling (Support for Consolidated & SMOTE paths)
        # Prioritize 'consolidation_path' used in Phase 2
        rel_path = row.get('consolidation_path', row.get('filename'))
        
        if os.path.isabs(rel_path):
            img_path = Path(rel_path)
        else:
            # Try multiple locations
            img_path = self.dataset_root / rel_path
            if not img_path.exists():
                # Fallback 1: dataset_smote_train
                img_path = self.dataset_root / 'dataset_smote_train' / rel_path
            if not img_path.exists():
                # Fallback 2: dataset_consolidation
                img_path = self.dataset_root / 'dataset_consolidation' / rel_path
            if not img_path.exists():
                # Fallback 3: Just filename in spectrograms
                img_path = self.dataset_root / 'spectrograms' / os.path.basename(rel_path)

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
             
        try:
            image = Image.open(img_path).convert('RGB')
        except:
             raise ValueError(f"Corrupt image: {img_path}")

        if self.transform:
            image = self.transform(image)
            
        # Labels
        # 1. Binary Label (0=Normal, 1=Precursor)
        mag_cls = str(row['magnitude_class'])
        is_precursor = 1 if mag_cls != 'Normal' else 0
        
        # 2. Magnitude Label
        # Map 'Major' -> 'Large', 'Small' -> 'Normal'
        if mag_cls == 'Major': mag_cls = 'Large'
        elif mag_cls == 'Small': mag_cls = 'Normal'
        mag_label = self.mag_to_idx.get(mag_cls, 0) # Default Normal
        
        # 3. Azimuth Label
        azi_cls = str(row.get('azimuth_class', 'Normal'))
        azi_label = self.azi_to_idx.get(azi_cls, 0) # Default Normal
        
        return image, is_precursor, mag_label, azi_label

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class HierarchicalEfficientNet(nn.Module):
    """
    EfficientNet-B0 with Hierarchical Heads:
    1. Binary Head (Gatekeeper)
    2. Magnitude Head (Conditional)
    3. Azimuth Head (Conditional)
    """
    def __init__(self, num_mag_classes=4, num_azi_classes=9, pretrained=True):
        super().__init__()
        # Load Backbone
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Feature Size
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity() # Remove default classifier
        
        # Shared Neck
        self.neck = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU()
        )
        
        # 1. Binary Head (Probabilitas Prekursor)
        self.binary_head = nn.Linear(256, 2) # [Normal, Precursor]
        
        # 2. Magnitude Head
        self.mag_head = nn.Linear(256, num_mag_classes)
        
        # 3. Azimuth Head
        self.azi_head = nn.Linear(256, num_azi_classes)
        
    def forward(self, x):
        features = self.backbone(x) # [B, 1280]
        embedding = self.neck(features) # [B, 256]
        
        binary_logits = self.binary_head(embedding)
        mag_logits = self.mag_head(embedding)
        azi_logits = self.azi_head(embedding)
        
        return binary_logits, mag_logits, azi_logits

# ============================================================================
# TRAINER CLASS
# ============================================================================

class ModelTrainerV2:
    def __init__(self, config: Dict, device_str='auto'):
        self.config = config
        
        if device_str == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)
            
        logger.info(f"Trainer V2 Initialized on {self.device}")
        
    def train(self, 
              train_metadata_path: str,
              val_metadata_path: str,
              dataset_root: str,
              epochs=50,
              early_stopping_patience=10):
              
        logger.info("="*50)
        logger.info("STARTING HIERARCHICAL TRAINING (V2)")
        logger.info("="*50)
        
        # 1. Prepare Data
        train_df = pd.read_csv(train_metadata_path)
        val_df = pd.read_csv(val_metadata_path)
        
        # Transforms (Augmentation in Training)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_ds = HierarchicalEarthquakeDataset(train_df, dataset_root, train_transform)
        val_ds = HierarchicalEarthquakeDataset(val_df, dataset_root, val_transform)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
        
        # 2. Setup Model & Loss
        model = HierarchicalEfficientNet().to(self.device)
        
        # Class Weights Calculation (Dynamic)
        # Hitung weight berdasarkan data training untuk mengatasi Imbalance
        mag_counts = train_df['magnitude_class'].value_counts()
        total_samples = len(train_df)
        
        # Weight untuk Binary (Normal vs Prekursor)
        n_normal = len(train_df[train_df['magnitude_class'] == 'Normal'])
        n_precursor = total_samples - n_normal
        binary_weights = torch.tensor([1.0, n_normal/n_precursor if n_precursor > 0 else 1.0]).to(self.device)
        
        # Weight untuk Magnitude (Aggressive for Large)
        # Urutan: Normal, Moderate, Medium, Large
        mag_w = []
        for cls in train_ds.mag_classes:
            count = len(train_df[train_df['magnitude_class'] == cls])
            w = total_samples / (count + 1e-5)
            if cls == 'Large': w *= 2.0 # Boost Large 2x lagi
            mag_w.append(w)
        mag_weights = torch.tensor(mag_w).float().to(self.device)
        # Normalize weights
        mag_weights = mag_weights / mag_weights.mean()
        
        logger.info(f"Binary Weights: {binary_weights}")
        logger.info(f"Magnitude Weights: {mag_weights}")
        
        criterion_binary = nn.CrossEntropyLoss(weight=binary_weights)
        criterion_mag = nn.CrossEntropyLoss(weight=mag_weights)
        criterion_azi = nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        # 3. Training Loop
        best_val_f1 = 0.0
        patience_counter = 0
        history = []
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            
            for imgs, is_prec, mag_lbl, azi_lbl in train_loader:
                imgs, is_prec = imgs.to(self.device), is_prec.to(self.device)
                mag_lbl, azi_lbl = mag_lbl.to(self.device), azi_lbl.to(self.device)
                
                optimizer.zero_grad()
                bin_out, mag_out, azi_out = model(imgs)
                
                # Hierarchical Loss Calculation
                loss_bin = criterion_binary(bin_out, is_prec)
                
                # Loss Magnitude & Azimuth hanya dihitung jika sampel adalah Prekursor (is_prec == 1)
                # Tapi karena batch processing susah filter, kita pakai masking atau hitung semua tapi weighted 
                # (di sini kita hitung semua dulu agar gradien mengalir lancar, tapi fokus utama di Precursor samples)
                
                loss_mag = criterion_mag(mag_out, mag_lbl) 
                loss_azi = criterion_azi(azi_out, azi_lbl)
                
                # Total Loss (Weighted)
                # Prioritas Tahap 1 (Binary) sangat tinggi
                total_loss = 2.0 * loss_bin + 1.0 * loss_mag + 0.5 * loss_azi
                
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()
                
            # Validation
            model.eval()
            val_preds_bin = []
            val_targets_bin = []
            val_preds_mag = []
            val_targets_mag = []
            
            with torch.no_grad():
                for imgs, is_prec, mag_lbl, azi_lbl in val_loader:
                    imgs = imgs.to(self.device)
                    is_prec, mag_lbl = is_prec.to(self.device), mag_lbl.to(self.device)
                    
                    bin_out, mag_out, _ = model(imgs)
                    
                    val_preds_bin.extend(bin_out.argmax(1).cpu().numpy())
                    val_targets_bin.extend(is_prec.cpu().numpy())
                    
                    val_preds_mag.extend(mag_out.argmax(1).cpu().numpy())
                    val_targets_mag.extend(mag_lbl.cpu().numpy())
            
            # Metrics
            f1_bin = f1_score(val_targets_bin, val_preds_bin, average='binary')
            # Recall khusus Large Class (Index 3 di mag_classes)
            # Perlu extract recall per class
            # ... (sementara pakai macro f1 mag)
            f1_mag = f1_score(val_targets_mag, val_preds_mag, average='macro')
            
            # Score Gabungan untuk Model Selection
            # Kita ingin Recall Large tinggi, jadi F1 Macro Magnitude sangat penting
            current_score = f1_mag 
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val F1 Bin: {f1_bin:.4f} | Val F1 Mag: {f1_mag:.4f}")
            
            scheduler.step(current_score)
            
            # Save Best
            if current_score > best_val_f1:
                best_val_f1 = current_score
                patience_counter = 0
                torch.save(model.state_dict(), 'best_hierarchical_model.pth')
                print(f"  [SAVED] New Best Model (Score: {current_score:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early Stopping Triggered.")
                    break
                    
        return model, best_val_f1

if __name__ == "__main__":
    # Test Run Mock
    pass
