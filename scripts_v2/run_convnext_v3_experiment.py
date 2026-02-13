#!/usr/bin/env python3
"""
Research Experiment: ConvNeXt for Modern Earthquake Precursor Detection
=======================================================================
Script ini mengadaptasi arsitektur ConvNeXt-Tiny untuk dataset Experiment 3 (Modern 2025).
Menggunakan dataset yang sudah dihomogenisasi dan di-SMOTE.

Author: Antigravity AI
Date: 13 February 2026
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, classification_report

# Tambahkan root project ke path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- KONFIGURASI ---
CONFIG = {
    "train_meta": "dataset_experiment_3/final_metadata/train_exp3.csv",
    "val_meta": "dataset_experiment_3/final_metadata/val_exp3.csv",
    "test_meta": "dataset_experiment_3/final_metadata/test_exp3.csv",
    "dataset_root": os.getcwd(),
    "output_dir": "experiments_v2/experiment_convnext_v3",
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "patience": 10
}

# Mapping Kelas Standar
MAG_CLASSES = ['Normal', 'Moderate', 'Medium', 'Large']
AZI_CLASSES = ['Normal', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

# --- DATASET HANDLING ---
class EarthquakeDataset(Dataset):
    def __init__(self, metadata_df, dataset_root, transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        
        self.mag_to_idx = {c: i for i, c in enumerate(MAG_CLASSES)}
        self.azi_to_idx = {c: i for i, c in enumerate(AZI_CLASSES)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Penanganan path relatif Experiment 3
        # Path di metadata biasanya 'spectrograms/...' atau 'spectrograms_smote/...'
        rel_path = row.get('consolidation_path', row.get('filename'))
        
        # Cari file di dataset_experiment_3 atau root
        img_path = self.dataset_root / 'dataset_experiment_3' / rel_path
        if not img_path.exists():
            img_path = self.dataset_root / rel_path

        if not img_path.exists():
            # Terakhir cek folder SMOTE
            img_path = self.dataset_root / 'dataset_experiment_3' / 'spectrograms_smote' / os.path.basename(rel_path)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback jika gambar rusak (biasanya tidak terjadi di clean dataset)
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)

        # Labels
        mag_cls = str(row['magnitude_class'])
        if mag_cls == 'Major': mag_cls = 'Large'
        mag_label = self.mag_to_idx.get(mag_cls, 0)
        
        azi_label = self.azi_to_idx.get(str(row.get('azimuth_class', 'Normal')), 0)
        
        return image, mag_label, azi_label

# --- MODEL ARCHITECTURE ---
class ConvNeXtEarthquake(nn.Module):
    """ConvNeXt-Tiny dengan Multi-Task Heads (Magnitude & Azimuth)"""
    def __init__(self, num_mag_classes=4, num_azi_classes=9):
        super().__init__()
        
        # Load ConvNeXt Backbone
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.backbone = models.convnext_tiny(weights=weights)
        num_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity() # Remove default head
        
        # Joint Head Structure (Modernized)
        self.mag_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Linear(512, num_mag_classes)
        )
        
        self.azi_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Linear(512, num_azi_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.flatten(1)
            
        mag_out = self.mag_head(features)
        azi_out = self.azi_head(features)
        
        return mag_out, azi_out

# --- TRAINER ---
def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(CONFIG["output_dir"], "training.log"), filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on Device: {device}")

    # Load Data
    train_df = pd.read_csv(CONFIG["train_meta"])
    val_df = pd.read_csv(CONFIG["val_meta"])
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = EarthquakeDataset(train_df, CONFIG["dataset_root"], train_transform)
    val_ds = EarthquakeDataset(val_df, CONFIG["dataset_root"], val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    # Weights Calculation (Handling Imbalance)
    counts = train_df['magnitude_class'].value_counts()
    weights = [train_df.shape[0] / (counts.get(c, 1)) for c in MAG_CLASSES]
    # Penalize 'Normal' slightly less to maintain stability
    class_weights = torch.tensor(weights).float().to(device)
    class_weights = class_weights / class_weights.mean()

    # Model, Loss, Optimizer
    model = ConvNeXtEarthquake().to(device)
    criterion_mag = nn.CrossEntropyLoss(weight=class_weights)
    criterion_azi = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_f1 = 0
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        
        for imgs, mag_lbl, azi_lbl in train_loader:
            imgs, mag_lbl, azi_lbl = imgs.to(device), mag_lbl.to(device), azi_lbl.to(device)
            
            optimizer.zero_grad()
            mag_out, azi_out = model(imgs)
            
            loss = criterion_mag(mag_out, mag_lbl) + 0.5 * criterion_azi(azi_out, azi_lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        preds_mag, targets_mag = [], []
        with torch.no_grad():
            for imgs, mag_lbl, azi_lbl in val_loader:
                imgs, mag_lbl = imgs.to(device), mag_lbl.to(device)
                mag_out, _ = model(imgs)
                preds_mag.extend(mag_out.argmax(1).cpu().numpy())
                targets_mag.extend(mag_lbl.cpu().numpy())

        f1 = f1_score(targets_mag, preds_mag, average='macro')
        logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {total_loss:.4f} | Val F1 (Mag): {f1:.4f}")
        
        scheduler.step(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG["output_dir"], "best_convnext_v3.pth"))
            logger.info(f"  [SAVED] New best model: {f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                logger.info("Early stopping triggered.")
                break

    logger.info("="*50)
    logger.info(f"TRAINING COMPLETE. Best Val F1: {best_f1:.4f}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
