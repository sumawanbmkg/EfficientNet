#!/usr/bin/env python3
"""
Training Model Perbaikan - Implementasi Semua Rekomendasi

Perbaikan yang diimplementasikan:
1. SMOTE Augmentation untuk minority classes
2. Balanced Focal Loss dengan automatic class weighting
3. Stronger regularization (dropout, weight decay)
4. Mixup + CutMix augmentation
5. Early stopping yang lebih ketat
6. SE Blocks untuk channel attention
7. Gradient clipping
8. Learning rate warmup dengan cosine decay

Author: Earthquake Prediction Research Team
Date: 5 February 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Install with: pip install imbalanced-learn")


# ============================================================================
# CONFIGURATION - CONSERVATIVE VERSION
# ============================================================================
CONFIG = {
    # Data
    "metadata_path": "dataset_unified/metadata/unified_metadata.csv",
    "dataset_dir": "dataset_unified",
    "image_size": 224,
    
    # Model
    "model_type": "efficientnet_b0",
    "num_mag_classes": 4,
    "num_azi_classes": 9,
    "dropout_rate": 0.3,  # Conservative dropout
    "use_se_blocks": True,
    
    # Training
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 1e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,  # Lower weight decay
    "warmup_epochs": 3,
    
    # Augmentation - DISABLED for conservative training
    "use_mixup": False,
    "mixup_alpha": 0.0,
    "use_cutmix": False,
    "cutmix_alpha": 0.0,
    "mixup_prob": 0.0,
    
    # Loss - Conservative settings
    "use_focal_loss": True,
    "focal_gamma": 1.0,  # Lower gamma (less aggressive)
    "label_smoothing": 0.05,  # Lower smoothing
    
    # Class Balancing - Use simple class weights only
    "use_class_weights": True,
    "use_smote": False,
    "smote_k_neighbors": 3,
    
    # Early Stopping
    "patience": 10,  # More patience
    "min_delta": 0.001,
    
    # Output
    "output_dir": "experiments_improved",
    "save_best_only": True,
}

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(CONFIG["output_dir"]) / f"improved_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save config
with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)


# ============================================================================
# BALANCED FOCAL LOSS
# ============================================================================
class BalancedFocalLoss(nn.Module):
    """
    Focal Loss dengan automatic class balancing.
    
    Combines:
    1. Focal Loss: (1-pt)^gamma * CE untuk fokus pada hard examples
    2. Class Balancing: Effective number of samples weighting
    3. Label Smoothing: Regularization
    """
    
    def __init__(self, num_classes, samples_per_class=None, gamma=2.0, 
                 beta=0.999, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.beta = beta
        self.label_smoothing = label_smoothing
        
        # Calculate class weights based on effective number of samples
        if samples_per_class is not None:
            effective_num = 1.0 - np.power(beta, samples_per_class)
            weights = (1.0 - beta) / (effective_num + 1e-8)
            weights = weights / weights.sum() * num_classes
            self.register_buffer('class_weights', torch.FloatTensor(weights))
        else:
            self.class_weights = None
    
    def forward(self, logits, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            n_classes = logits.size(-1)
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Compute probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Focal weight: (1 - pt)^gamma
        pt = probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        
        # Cross entropy loss
        if self.label_smoothing > 0:
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        
        # Apply class weights
        if self.class_weights is not None:
            class_weight = self.class_weights[targets]
            focal_loss = focal_weight * ce_loss * class_weight
        else:
            focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


# ============================================================================
# SE BLOCK (Squeeze-and-Excitation)
# ============================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block untuk channel attention"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ============================================================================
# IMPROVED MODEL
# ============================================================================
class ImprovedMultiTaskModel(nn.Module):
    """
    Model perbaikan dengan:
    1. EfficientNet-B0 backbone
    2. SE Blocks untuk channel attention
    3. Stronger dropout
    4. Separate task heads dengan regularization
    """
    
    def __init__(self, num_mag_classes=4, num_azi_classes=9, 
                 dropout_rate=0.5, use_se=True):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base_model.features
        feature_dim = base_model.classifier[1].in_features  # 1280
        
        # SE Block (optional)
        self.use_se = use_se
        if use_se:
            self.se_block = SEBlock(feature_dim, reduction=16)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Shared representation dengan stronger regularization
        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
        )
        
        # Magnitude head
        self.mag_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_mag_classes)
        )
        
        # Azimuth head (lebih complex karena 9 classes)
        self.azi_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(128, num_azi_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.shared_fc, self.mag_head, self.azi_head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        
        # SE attention
        if self.use_se:
            x = self.se_block(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Shared representation
        x = self.shared_fc(x)
        
        # Task-specific outputs
        mag_out = self.mag_head(x)
        azi_out = self.azi_head(x)
        
        return mag_out, azi_out
    
    def get_features(self, x):
        """Get feature embeddings untuk visualization"""
        x = self.features(x)
        if self.use_se:
            x = self.se_block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.shared_fc(x)


# ============================================================================
# DATASET WITH SMOTE SUPPORT
# ============================================================================
class ImprovedDataset(Dataset):
    """Dataset dengan support untuk SMOTE augmented data"""
    
    def __init__(self, metadata_df, dataset_dir, transform=None,
                 mag_mapping=None, azi_mapping=None, 
                 smote_features=None, smote_labels=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.mag_mapping = mag_mapping or {}
        self.azi_mapping = azi_mapping or {}
        
        # SMOTE augmented data (optional)
        self.smote_features = smote_features
        self.smote_labels = smote_labels
        self.use_smote = smote_features is not None
        
        if self.use_smote:
            self.total_len = len(self.metadata) + len(smote_features)
        else:
            self.total_len = len(self.metadata)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if self.use_smote and idx >= len(self.metadata):
            # Return SMOTE augmented sample
            smote_idx = idx - len(self.metadata)
            features = torch.FloatTensor(self.smote_features[smote_idx])
            mag_label, azi_label = self.smote_labels[smote_idx]
            return features, mag_label, azi_label
        
        # Return original sample
        row = self.metadata.iloc[idx]
        
        # Load image - handle various path columns
        img_path = None
        
        # Try unified_path first (already contains relative path like spectrograms/original/...)
        if 'unified_path' in row.index and pd.notna(row['unified_path']):
            img_path = self.dataset_dir / row['unified_path']
        
        # If not found, try spectrogram_file in various locations
        if img_path is None or not img_path.exists():
            spec_file = row.get('spectrogram_file') if pd.notna(row.get('spectrogram_file')) else None
            if spec_file:
                # Check in various subdirectories
                for subdir in ['original', 'augmented', 'normal', '']:
                    if subdir:
                        test_path = self.dataset_dir / 'spectrograms' / subdir / spec_file
                    else:
                        test_path = self.dataset_dir / 'spectrograms' / spec_file
                    if test_path.exists():
                        img_path = test_path
                        break
        
        # Try filename column
        if img_path is None or not img_path.exists():
            filename = row.get('filename') if pd.notna(row.get('filename')) else None
            if filename:
                for subdir in ['original', 'augmented', 'normal', '']:
                    if subdir:
                        test_path = self.dataset_dir / 'spectrograms' / subdir / filename
                    else:
                        test_path = self.dataset_dir / 'spectrograms' / filename
                    if test_path.exists():
                        img_path = test_path
                        break
        
        # Last resort: check for noisy files in root spectrograms folder
        if img_path is None or not img_path.exists():
            station = row.get('station', '')
            date = str(row.get('date', '')).replace('-', '')
            # Try noisy pattern
            noisy_pattern = f"noisy_{station}_{date}"
            for f in (self.dataset_dir / 'spectrograms').glob(f"{noisy_pattern}*.png"):
                img_path = f
                break
        
        if img_path is None or not img_path.exists():
            raise FileNotFoundError(f"Cannot find image for row {idx}: station={row.get('station')}, date={row.get('date')}, unified_path={row.get('unified_path')}, spectrogram_file={row.get('spectrogram_file')}")
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        mag_label = self.mag_mapping.get(str(row['magnitude_class']), 0)
        azi_label = self.azi_mapping.get(str(row['azimuth_class']), 0)
        
        return image, mag_label, azi_label


# ============================================================================
# MIXUP AND CUTMIX
# ============================================================================
def mixup_data(x, y_mag, y_azi, alpha=0.4):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_mag_a, y_mag_b = y_mag, y_mag[index]
    y_azi_a, y_azi_b = y_azi, y_azi[index]
    
    return mixed_x, y_mag_a, y_mag_b, y_azi_a, y_azi_b, lam


def cutmix_data(x, y_mag, y_azi, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_mag_a, y_mag_b = y_mag, y_mag[index]
    y_azi_a, y_azi_b = y_azi, y_azi[index]
    
    return x, y_mag_a, y_mag_b, y_azi_a, y_azi_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup/cutmix"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# LEARNING RATE SCHEDULER WITH WARMUP
# ============================================================================
class WarmupCosineScheduler:
    """Cosine LR scheduler dengan linear warmup"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, 
                 warmup_lr=1e-6, min_lr=1e-6, steps_per_epoch=1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            progress = self.current_step / self.warmup_steps
            for i, pg in enumerate(self.optimizer.param_groups):
                pg['lr'] = self.warmup_lr + progress * (self.base_lrs[i] - self.warmup_lr)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            for i, pg in enumerate(self.optimizer.param_groups):
                pg['lr'] = self.min_lr + 0.5 * (self.base_lrs[i] - self.min_lr) * (1 + np.cos(np.pi * progress))
    
    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ============================================================================
# EARLY STOPPING
# ============================================================================
class EarlyStopping:
    """Early stopping dengan patience dan min_delta"""
    
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop



# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, 
                criterion_mag, criterion_azi, device, scaler, epoch):
    """Train for one epoch - Conservative version without Mixup/CutMix"""
    model.train()
    
    total_loss = 0
    mag_correct = 0
    azi_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, mag_labels, azi_labels) in enumerate(pbar):
        images = images.to(device)
        mag_labels = mag_labels.to(device)
        azi_labels = azi_labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            mag_out, azi_out = model(images)
            loss_mag = criterion_mag(mag_out, mag_labels)
            loss_azi = criterion_azi(azi_out, azi_labels)
            
            # Combined loss (magnitude lebih penting)
            loss = loss_mag + 0.5 * loss_azi
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update scheduler
        scheduler.step()
        
        # Statistics
        with torch.no_grad():
            total_loss += loss.item() * images.size(0)
            mag_pred = torch.argmax(mag_out, dim=1)
            azi_pred = torch.argmax(azi_out, dim=1)
            mag_correct += (mag_pred == mag_labels).sum().item()
            azi_correct += (azi_pred == azi_labels).sum().item()
            total_samples += images.size(0)
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mag': f'{100*mag_correct/total_samples:.1f}%',
            'azi': f'{100*azi_correct/total_samples:.1f}%',
            'lr': f'{current_lr:.2e}'
        })
    
    return {
        'loss': total_loss / total_samples,
        'mag_acc': 100 * mag_correct / total_samples,
        'azi_acc': 100 * azi_correct / total_samples
    }


@torch.no_grad()
def validate(model, dataloader, criterion_mag, criterion_azi, device):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    all_mag_preds, all_mag_labels = [], []
    all_azi_preds, all_azi_labels = [], []
    
    for images, mag_labels, azi_labels in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        mag_labels = mag_labels.to(device)
        azi_labels = azi_labels.to(device)
        
        mag_out, azi_out = model(images)
        
        loss_mag = criterion_mag(mag_out, mag_labels)
        loss_azi = criterion_azi(azi_out, azi_labels)
        loss = loss_mag + 0.5 * loss_azi
        
        total_loss += loss.item() * images.size(0)
        
        mag_pred = torch.argmax(mag_out, dim=1)
        azi_pred = torch.argmax(azi_out, dim=1)
        
        all_mag_preds.extend(mag_pred.cpu().numpy())
        all_mag_labels.extend(mag_labels.cpu().numpy())
        all_azi_preds.extend(azi_pred.cpu().numpy())
        all_azi_labels.extend(azi_labels.cpu().numpy())
    
    # Calculate metrics
    mag_acc = accuracy_score(all_mag_labels, all_mag_preds) * 100
    azi_acc = accuracy_score(all_azi_labels, all_azi_preds) * 100
    mag_f1 = f1_score(all_mag_labels, all_mag_preds, average='weighted') * 100
    azi_f1 = f1_score(all_azi_labels, all_azi_preds, average='weighted') * 100
    
    return {
        'loss': total_loss / len(dataloader.dataset),
        'mag_acc': mag_acc,
        'azi_acc': azi_acc,
        'mag_f1': mag_f1,
        'azi_f1': azi_f1,
        'mag_preds': all_mag_preds,
        'mag_labels': all_mag_labels,
        'azi_preds': all_azi_preds,
        'azi_labels': all_azi_labels
    }


def create_class_mappings(metadata_df):
    """Create class mappings from metadata"""
    mag_classes = sorted(metadata_df['magnitude_class'].dropna().unique())
    azi_classes = sorted(metadata_df['azimuth_class'].dropna().unique())
    
    mag_mapping = {str(cls): idx for idx, cls in enumerate(mag_classes)}
    azi_mapping = {str(cls): idx for idx, cls in enumerate(azi_classes)}
    
    return mag_mapping, azi_mapping, mag_classes, azi_classes


def get_class_counts(metadata_df, mag_mapping, azi_mapping):
    """Get class counts for loss weighting"""
    mag_counts = np.zeros(len(mag_mapping))
    azi_counts = np.zeros(len(azi_mapping))
    
    for _, row in metadata_df.iterrows():
        mag_idx = mag_mapping.get(str(row['magnitude_class']), 0)
        azi_idx = azi_mapping.get(str(row['azimuth_class']), 0)
        mag_counts[mag_idx] += 1
        azi_counts[azi_idx] += 1
    
    return mag_counts, azi_counts


def get_transforms(is_training=True, image_size=224):
    """Get data transforms"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_weighted_sampler(metadata_df, mag_mapping, azi_mapping):
    """Create weighted sampler untuk class balancing"""
    # Combine magnitude and azimuth for stratification
    combined_labels = []
    for _, row in metadata_df.iterrows():
        mag_idx = mag_mapping.get(str(row['magnitude_class']), 0)
        azi_idx = azi_mapping.get(str(row['azimuth_class']), 0)
        combined_labels.append(mag_idx * len(azi_mapping) + azi_idx)
    
    # Calculate weights
    class_counts = Counter(combined_labels)
    weights = [1.0 / class_counts[label] for label in combined_labels]
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("TRAINING MODEL PERBAIKAN")
    print("=" * 70)
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Load metadata
    print("Loading metadata...")
    metadata_df = pd.read_csv(CONFIG["metadata_path"])
    metadata_df = metadata_df.dropna(subset=['magnitude_class', 'azimuth_class'])
    print(f"Total samples: {len(metadata_df)}")
    
    # Create class mappings
    mag_mapping, azi_mapping, mag_classes, azi_classes = create_class_mappings(metadata_df)
    print(f"Magnitude classes: {mag_classes}")
    print(f"Azimuth classes: {azi_classes}")
    
    # Get class counts
    mag_counts, azi_counts = get_class_counts(metadata_df, mag_mapping, azi_mapping)
    print(f"\nMagnitude distribution: {dict(zip(mag_classes, mag_counts.astype(int)))}")
    print(f"Azimuth distribution: {dict(zip(azi_classes, azi_counts.astype(int)))}")
    
    # Split data (event-based)
    metadata_df['event_id'] = metadata_df['station'] + '_' + metadata_df['date'].astype(str)
    unique_events = metadata_df['event_id'].unique()
    
    np.random.seed(42)
    np.random.shuffle(unique_events)
    
    n_train = int(0.7 * len(unique_events))
    n_val = int(0.15 * len(unique_events))
    
    train_events = unique_events[:n_train]
    val_events = unique_events[n_train:n_train+n_val]
    test_events = unique_events[n_train+n_val:]
    
    train_df = metadata_df[metadata_df['event_id'].isin(train_events)]
    val_df = metadata_df[metadata_df['event_id'].isin(val_events)]
    test_df = metadata_df[metadata_df['event_id'].isin(test_events)]
    
    print(f"\nTrain: {len(train_df)} samples ({len(train_events)} events)")
    print(f"Val: {len(val_df)} samples ({len(val_events)} events)")
    print(f"Test: {len(test_df)} samples ({len(test_events)} events)")
    
    # Create datasets
    train_transform = get_transforms(is_training=True, image_size=CONFIG["image_size"])
    val_transform = get_transforms(is_training=False, image_size=CONFIG["image_size"])
    
    train_dataset = ImprovedDataset(
        train_df, CONFIG["dataset_dir"], train_transform,
        mag_mapping, azi_mapping
    )
    val_dataset = ImprovedDataset(
        val_df, CONFIG["dataset_dir"], val_transform,
        mag_mapping, azi_mapping
    )
    test_dataset = ImprovedDataset(
        test_df, CONFIG["dataset_dir"], val_transform,
        mag_mapping, azi_mapping
    )
    
    # Create dataloaders (optimized for CPU) - use simple shuffle instead of weighted sampler
    num_workers = 0 if device.type == 'cpu' else 4
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )
    
    # Create model
    print("\nCreating model...")
    model = ImprovedMultiTaskModel(
        num_mag_classes=len(mag_classes),
        num_azi_classes=len(azi_classes),
        dropout_rate=CONFIG["dropout_rate"],
        use_se=CONFIG["use_se_blocks"]
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss functions with class balancing
    print("\nCreating loss functions with class balancing...")
    criterion_mag = BalancedFocalLoss(
        num_classes=len(mag_classes),
        samples_per_class=mag_counts,
        gamma=CONFIG["focal_gamma"],
        label_smoothing=CONFIG["label_smoothing"]
    ).to(device)
    
    criterion_azi = BalancedFocalLoss(
        num_classes=len(azi_classes),
        samples_per_class=azi_counts,
        gamma=CONFIG["focal_gamma"],
        label_smoothing=CONFIG["label_smoothing"]
    ).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Create scheduler
    steps_per_epoch = len(train_loader)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=CONFIG["warmup_epochs"],
        total_epochs=CONFIG["epochs"],
        warmup_lr=1e-6,
        min_lr=CONFIG["min_lr"],
        steps_per_epoch=steps_per_epoch
    )
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=CONFIG["patience"],
        min_delta=CONFIG["min_delta"],
        mode='max'
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    best_val_f1 = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{CONFIG['epochs']}")
        print(f"{'='*70}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion_mag, criterion_azi, device, scaler, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion_mag, criterion_azi, device)
        
        # Log metrics
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Mag: {train_metrics['mag_acc']:.2f}%, Azi: {train_metrics['azi_acc']:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Mag: {val_metrics['mag_acc']:.2f}% (F1: {val_metrics['mag_f1']:.2f}%), "
              f"Azi: {val_metrics['azi_acc']:.2f}% (F1: {val_metrics['azi_f1']:.2f}%)")
        
        # Combined F1 score
        combined_f1 = (val_metrics['mag_f1'] + val_metrics['azi_f1']) / 2
        
        # Save best model
        if combined_f1 > best_val_f1:
            best_val_f1 = combined_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': CONFIG
            }, OUTPUT_DIR / 'best_model.pth')
            print(f"✅ New best model saved! Combined F1: {combined_f1:.2f}%")
        
        # Early stopping check
        if early_stopping(combined_f1):
            print(f"\n⚠️ Early stopping triggered at epoch {epoch}")
            break
    
    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(OUTPUT_DIR / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, criterion_mag, criterion_azi, device)
    
    print(f"\nTest Results:")
    print(f"  Magnitude Accuracy: {test_metrics['mag_acc']:.2f}%")
    print(f"  Magnitude F1: {test_metrics['mag_f1']:.2f}%")
    print(f"  Azimuth Accuracy: {test_metrics['azi_acc']:.2f}%")
    print(f"  Azimuth F1: {test_metrics['azi_f1']:.2f}%")
    
    # Save classification reports
    print("\nMagnitude Classification Report:")
    mag_report = classification_report(
        test_metrics['mag_labels'], test_metrics['mag_preds'],
        target_names=mag_classes, digits=4
    )
    print(mag_report)
    
    print("\nAzimuth Classification Report:")
    azi_report = classification_report(
        test_metrics['azi_labels'], test_metrics['azi_preds'],
        target_names=azi_classes, digits=4
    )
    print(azi_report)
    
    # Save results
    results = {
        'test_metrics': {
            'mag_acc': test_metrics['mag_acc'],
            'mag_f1': test_metrics['mag_f1'],
            'azi_acc': test_metrics['azi_acc'],
            'azi_f1': test_metrics['azi_f1']
        },
        'best_epoch': checkpoint['epoch'],
        'config': CONFIG,
        'history': history
    }
    
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Plot training curves
    plot_training_curves(history, OUTPUT_DIR)
    
    # Plot confusion matrices
    plot_confusion_matrices(
        test_metrics, mag_classes, azi_classes, OUTPUT_DIR
    )
    
    print(f"\n✅ Training complete! Results saved to: {OUTPUT_DIR}")
    
    return results


def plot_training_curves(history, output_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, [h['loss'] for h in history['train']], 'b-', label='Train')
    axes[0, 0].plot(epochs, [h['loss'] for h in history['val']], 'r-', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Magnitude Accuracy
    axes[0, 1].plot(epochs, [h['mag_acc'] for h in history['train']], 'b-', label='Train')
    axes[0, 1].plot(epochs, [h['mag_acc'] for h in history['val']], 'r-', label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Magnitude Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Azimuth Accuracy
    axes[1, 0].plot(epochs, [h['azi_acc'] for h in history['train']], 'b-', label='Train')
    axes[1, 0].plot(epochs, [h['azi_acc'] for h in history['val']], 'r-', label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Azimuth Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Scores
    axes[1, 1].plot(epochs, [h['mag_f1'] for h in history['val']], 'g-', label='Mag F1')
    axes[1, 1].plot(epochs, [h['azi_f1'] for h in history['val']], 'm-', label='Azi F1')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score (%)')
    axes[1, 1].set_title('Validation F1 Scores')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()
    print(f"✅ Training curves saved")


def plot_confusion_matrices(test_metrics, mag_classes, azi_classes, output_dir):
    """Plot confusion matrices"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Magnitude confusion matrix
    mag_cm = confusion_matrix(test_metrics['mag_labels'], test_metrics['mag_preds'])
    sns.heatmap(mag_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=mag_classes, yticklabels=mag_classes, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Magnitude Confusion Matrix')
    
    # Azimuth confusion matrix
    azi_cm = confusion_matrix(test_metrics['azi_labels'], test_metrics['azi_preds'])
    sns.heatmap(azi_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=azi_classes, yticklabels=azi_classes, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Azimuth Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=150)
    plt.close()
    print(f"✅ Confusion matrices saved")


if __name__ == '__main__':
    main()
