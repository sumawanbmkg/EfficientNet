#!/usr/bin/env python3
"""
Retrain Model with Heavy Augmentation for Large Class
Solusi untuk masalah class imbalance pada deteksi gempa besar

Strategi:
1. Augmentasi agresif untuk Large class (target: 500+ samples)
2. Focal Loss untuk fokus pada hard examples
3. Class weights 100x untuk Large
4. Stratified sampling per batch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import random

print("="*70)
print("RETRAIN MODEL WITH AUGMENTATION")
print("Solving class imbalance for Large earthquake detection")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'dataset_dir': 'dataset_unified',
    'img_size': 224,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.0001,
    'patience': 15,
    'focal_gamma': 2.0,  # Focal loss gamma
    'large_weight': 100.0,  # Weight for Large class
    'augment_large_factor': 20,  # Augment Large samples 20x
}

# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss untuk mengatasi class imbalance
    Memberikan penalti lebih besar pada misclassification
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ============================================================================
# AUGMENTATION TRANSFORMS
# ============================================================================

# Heavy augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extra heavy augmentation specifically for Large class
large_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),  # More rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# DATASET WITH OVERSAMPLING
# ============================================================================

class AugmentedEarthquakeDataset(Dataset):
    """Dataset dengan oversampling untuk Large class"""
    
    def __init__(self, dataframe, dataset_dir, transform=None, 
                 oversample_large=False, large_factor=10):
        self.df = dataframe.copy()
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.large_transform = large_transform
        self.oversample_large = oversample_large
        self.large_factor = large_factor
        
        # Source directories
        self.source_dirs = {
            'v2.1_original': Path('dataset_spectrogram_ssh_v22') / 'spectrograms',
            'augmented': Path('dataset_augmented') / 'spectrograms',
            'quiet_days.csv': Path('dataset_normal') / 'spectrograms'
        }
        
        # Oversample Large class
        if oversample_large:
            large_samples = self.df[self.df['magnitude_class'] == 'Large']
            print(f"   Original Large samples: {len(large_samples)}")
            
            # Duplicate Large samples
            for _ in range(large_factor - 1):
                self.df = pd.concat([self.df, large_samples], ignore_index=True)
            
            print(f"   After oversampling: {len(self.df[self.df['magnitude_class'] == 'Large'])} Large samples")
            print(f"   Total samples: {len(self.df)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Find spectrogram file
        dataset_source = row.get('dataset_source', 'v2.1_original')
        source_dir = self.source_dirs.get(dataset_source, self.source_dirs['v2.1_original'])
        filename = row['spectrogram_file']
        
        # Try multiple paths
        spec_path = None
        for path_variant in [filename, filename.replace('_normal_', '_'), filename.replace('_aug', '')]:
            for src_dir in self.source_dirs.values():
                test_path = src_dir / path_variant
                if test_path.exists():
                    spec_path = test_path
                    break
            if spec_path:
                break
        
        if spec_path is None or not spec_path.exists():
            raise FileNotFoundError(f"Cannot find: {filename}")
        
        image = Image.open(spec_path).convert('RGB')
        
        # Use heavy augmentation for Large class
        is_large = row['magnitude_class'] == 'Large'
        if is_large and self.oversample_large:
            image = self.large_transform(image)
        elif self.transform:
            image = self.transform(image)
        
        mag_label = row['magnitude_label']
        azi_label = row['azimuth_label']
        
        return image, (mag_label, azi_label)

# ============================================================================
# MODEL
# ============================================================================

class MultiTaskEfficientNet(nn.Module):
    def __init__(self, num_magnitude_classes=4, num_azimuth_classes=9, dropout_rate=0.4):
        super(MultiTaskEfficientNet, self).__init__()
        base_model = models.efficientnet_b0(pretrained=True)
        feature_dim = base_model.classifier[1].in_features
        
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
        )
        
        self.magnitude_head = nn.Linear(512, num_magnitude_classes)
        self.azimuth_head = nn.Linear(512, num_azimuth_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared_fc(x)
        return self.magnitude_head(x), self.azimuth_head(x)

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    # Load data
    print(f"\n📊 Loading data...")
    
    split_file = Path(CONFIG['dataset_dir']) / 'metadata' / 'fixed_split_indices.json'
    with open(split_file, 'r') as f:
        split_indices = json.load(f)
    
    metadata_file = Path(CONFIG['dataset_dir']) / 'metadata' / 'unified_metadata.csv'
    df = pd.read_csv(metadata_file)
    
    train_df = df.iloc[split_indices['train_indices']].reset_index(drop=True)
    val_df = df.iloc[split_indices['val_indices']].reset_index(drop=True)
    
    # Class mappings
    magnitude_classes = ['Large', 'Medium', 'Moderate', 'Normal']
    azimuth_classes = ['E', 'N', 'NE', 'NW', 'Normal', 'S', 'SE', 'SW', 'W']
    
    magnitude_to_idx = {cls: idx for idx, cls in enumerate(magnitude_classes)}
    azimuth_to_idx = {cls: idx for idx, cls in enumerate(azimuth_classes)}
    
    # Encode labels
    for split_df in [train_df, val_df]:
        split_df['magnitude_label'] = split_df['magnitude_class'].map(magnitude_to_idx)
        split_df['azimuth_label'] = split_df['azimuth_class'].map(azimuth_to_idx)
    
    # Drop unmapped
    train_df = train_df.dropna(subset=['magnitude_label', 'azimuth_label']).reset_index(drop=True)
    val_df = val_df.dropna(subset=['magnitude_label', 'azimuth_label']).reset_index(drop=True)
    
    train_df['magnitude_label'] = train_df['magnitude_label'].astype(int)
    train_df['azimuth_label'] = train_df['azimuth_label'].astype(int)
    val_df['magnitude_label'] = val_df['magnitude_label'].astype(int)
    val_df['azimuth_label'] = val_df['azimuth_label'].astype(int)
    
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Class distribution before oversampling
    print(f"\n📊 Class distribution (before oversampling):")
    for cls in magnitude_classes:
        count = len(train_df[train_df['magnitude_class'] == cls])
        print(f"   {cls}: {count}")
    
    # Create datasets with oversampling
    print(f"\n📊 Creating datasets with Large class oversampling...")
    train_dataset = AugmentedEarthquakeDataset(
        train_df, CONFIG['dataset_dir'], 
        transform=train_transform,
        oversample_large=True,
        large_factor=CONFIG['augment_large_factor']
    )
    
    val_dataset = AugmentedEarthquakeDataset(
        val_df, CONFIG['dataset_dir'],
        transform=val_transform,
        oversample_large=False
    )
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=0, pin_memory=True)
    
    # Model
    print(f"\n📊 Building model...")
    model = MultiTaskEfficientNet(len(magnitude_classes), len(azimuth_classes))
    model = model.to(device)
    
    # Class weights - VERY HIGH for Large
    magnitude_weights = torch.FloatTensor([
        CONFIG['large_weight'],  # Large: 100x
        2.0,   # Medium
        10.0,  # Moderate
        1.0    # Normal
    ]).to(device)
    
    azimuth_weights = torch.ones(len(azimuth_classes)).to(device)
    
    print(f"   Magnitude weights: {magnitude_weights}")
    
    # Focal Loss
    criterion_magnitude = FocalLoss(alpha=magnitude_weights, gamma=CONFIG['focal_gamma'])
    criterion_azimuth = FocalLoss(alpha=azimuth_weights, gamma=CONFIG['focal_gamma'])
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # Training
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path('experiments_augmented') / f'exp_aug_{timestamp}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mag_f1': [], 'val_azi_f1': []}
    
    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        for images, (mag_labels, azi_labels) in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            images = images.to(device)
            mag_labels = mag_labels.to(device)
            azi_labels = azi_labels.to(device)
            
            optimizer.zero_grad()
            mag_out, azi_out = model(images)
            
            loss_mag = criterion_magnitude(mag_out, mag_labels)
            loss_azi = criterion_azimuth(azi_out, azi_labels)
            loss = loss_mag + loss_azi
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_mag_preds, all_mag_labels = [], []
        all_azi_preds, all_azi_labels = [], []
        
        with torch.no_grad():
            for images, (mag_labels, azi_labels) in val_loader:
                images = images.to(device)
                mag_labels = mag_labels.to(device)
                azi_labels = azi_labels.to(device)
                
                mag_out, azi_out = model(images)
                
                loss_mag = criterion_magnitude(mag_out, mag_labels)
                loss_azi = criterion_azimuth(azi_out, azi_labels)
                val_loss += (loss_mag + loss_azi).item()
                
                all_mag_preds.extend(torch.argmax(mag_out, 1).cpu().numpy())
                all_mag_labels.extend(mag_labels.cpu().numpy())
                all_azi_preds.extend(torch.argmax(azi_out, 1).cpu().numpy())
                all_azi_labels.extend(azi_labels.cpu().numpy())
        
        # Metrics
        mag_f1 = f1_score(all_mag_labels, all_mag_preds, average='macro')
        azi_f1 = f1_score(all_azi_labels, all_azi_preds, average='macro')
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_mag_f1'].append(mag_f1)
        history['val_azi_f1'].append(azi_f1)
        
        print(f"\nEpoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
              f"Val Mag F1={mag_f1:.4f}, Val Azi F1={azi_f1:.4f}")
        
        # Check Large class specifically
        large_idx = magnitude_to_idx['Large']
        large_mask = np.array(all_mag_labels) == large_idx
        if large_mask.sum() > 0:
            large_correct = (np.array(all_mag_preds)[large_mask] == large_idx).sum()
            large_recall = large_correct / large_mask.sum()
            print(f"   Large class recall: {large_recall:.2%} ({large_correct}/{large_mask.sum()})")
        
        # Save best
        if mag_f1 > best_val_f1:
            best_val_f1 = mag_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mag_f1': mag_f1,
                'val_azi_f1': azi_f1,
            }, exp_dir / 'best_model.pth')
            print(f"   ✅ Best model saved! (F1={best_val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['patience']:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    # Save class mappings
    with open(exp_dir / 'class_mappings.json', 'w') as f:
        json.dump({
            'magnitude_classes': magnitude_classes,
            'azimuth_classes': azimuth_classes,
            'magnitude_to_idx': magnitude_to_idx,
            'azimuth_to_idx': azimuth_to_idx
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"Best Val Mag F1: {best_val_f1:.4f}")
    print(f"Model saved to: {exp_dir}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
