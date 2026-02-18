#!/usr/bin/env python3
"""
Retrain ConvNeXt with Heavy Augmentation for Large Class
Same strategy as EfficientNet augmented training

Strategi:
1. Focal Loss untuk fokus pada hard examples
2. Class weights 100x untuk Large
3. Oversampling Large class 20x
4. Heavy augmentation
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
print("RETRAIN CONVNEXT WITH AUGMENTATION")
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
    'batch_size': 16,  # Smaller batch for ConvNeXt (more memory)
    'epochs': 50,
    'learning_rate': 0.0001,
    'patience': 15,
    'focal_gamma': 2.0,
    'large_weight': 100.0,
    'augment_large_factor': 20,
}

# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
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

large_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# DATASET
# ============================================================================

class AugmentedDataset(Dataset):
    def __init__(self, metadata, indices, transform, large_transform=None, 
                 augment_large=False, augment_factor=20):
        self.metadata = metadata.iloc[indices].reset_index(drop=True)
        self.transform = transform
        self.large_transform = large_transform
        self.augment_large = augment_large
        self.augment_factor = augment_factor
        
        # Source directories - same as EfficientNet training
        self.source_dirs = {
            'v2.1_original': Path('dataset_spectrogram_ssh_v22') / 'spectrograms',
            'augmented': Path('dataset_augmented') / 'spectrograms',
            'quiet_days.csv': Path('dataset_normal') / 'spectrograms'
        }
        
        # Build sample list with oversampling
        self.samples = []
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            mag_label = row['magnitude_label']
            
            if augment_large and mag_label == 'Large':
                # Add multiple copies for Large class
                for _ in range(augment_factor):
                    self.samples.append((idx, True))  # True = use heavy augmentation
            else:
                self.samples.append((idx, False))
        
        print(f"  Dataset: {len(self.metadata)} original → {len(self.samples)} with oversampling")
    
    def __len__(self):
        return len(self.samples)
    
    def _find_spectrogram(self, filename):
        """Find spectrogram file in multiple directories"""
        for path_variant in [filename, filename.replace('_normal_', '_'), filename.replace('_aug', '')]:
            for src_dir in self.source_dirs.values():
                test_path = src_dir / path_variant
                if test_path.exists():
                    return test_path
        return None
    
    def __getitem__(self, idx):
        orig_idx, use_heavy_aug = self.samples[idx]
        row = self.metadata.iloc[orig_idx]
        
        filename = row['spectrogram_file']
        img_path = self._find_spectrogram(filename)
        
        if img_path is None:
            raise FileNotFoundError(f"Cannot find: {filename}")
        
        image = Image.open(img_path).convert('RGB')
        
        if use_heavy_aug and self.large_transform:
            image = self.large_transform(image)
        else:
            image = self.transform(image)
        
        return image, row['magnitude_idx'], row['azimuth_idx']

# ============================================================================
# CONVNEXT MODEL
# ============================================================================

class MultiTaskConvNeXt(nn.Module):
    def __init__(self, num_magnitude_classes=4, num_azimuth_classes=9):
        super(MultiTaskConvNeXt, self).__init__()
        self.backbone = models.convnext_tiny(pretrained=True)
        num_features = self.backbone.classifier[2].in_features  # 768
        self.backbone.classifier = nn.Identity()
        
        self.mag_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Flatten(start_dim=1),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_magnitude_classes)
        )
        
        self.azi_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Flatten(start_dim=1),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_azimuth_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.mean([-2, -1])
        return self.mag_head(features), self.azi_head(features)

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print(f"\n📊 Loading data...")
    
    split_file = Path(CONFIG['dataset_dir']) / 'metadata' / 'fixed_split_indices.json'
    with open(split_file, 'r') as f:
        split_indices = json.load(f)
    
    metadata_file = Path(CONFIG['dataset_dir']) / 'metadata' / 'unified_metadata.csv'
    metadata = pd.read_csv(metadata_file)
    
    # Get class info - use correct column names
    magnitude_classes = sorted(metadata['magnitude_class'].unique())
    azimuth_classes = sorted(metadata['azimuth_class'].unique())
    
    mag_to_idx = {c: i for i, c in enumerate(magnitude_classes)}
    azi_to_idx = {c: i for i, c in enumerate(azimuth_classes)}
    
    metadata['magnitude_idx'] = metadata['magnitude_class'].map(mag_to_idx)
    metadata['azimuth_idx'] = metadata['azimuth_class'].map(azi_to_idx)
    
    # Add label columns for compatibility
    metadata['magnitude_label'] = metadata['magnitude_class']
    metadata['azimuth_label'] = metadata['azimuth_class']
    
    print(f"  Magnitude classes: {magnitude_classes}")
    print(f"  Azimuth classes: {azimuth_classes}")
    
    # Count Large samples - use correct key names
    train_meta = metadata.iloc[split_indices['train_indices']]
    large_count = (train_meta['magnitude_label'] == 'Large').sum()
    print(f"\n📊 Large samples in training: {large_count}")
    print(f"   After {CONFIG['augment_large_factor']}x oversampling: {large_count * CONFIG['augment_large_factor']}")
    
    # Create datasets
    print(f"\n📦 Creating datasets...")
    train_dataset = AugmentedDataset(
        metadata, split_indices['train_indices'], train_transform, large_transform,
        augment_large=True, augment_factor=CONFIG['augment_large_factor']
    )
    val_dataset = AugmentedDataset(
        metadata, split_indices['val_indices'], val_transform,
        augment_large=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # Model
    print(f"\n📊 Building ConvNeXt model...")
    model = MultiTaskConvNeXt(len(magnitude_classes), len(azimuth_classes))
    model = model.to(device)
    
    # Class weights for Focal Loss
    mag_weights = torch.ones(len(magnitude_classes)).to(device)
    large_idx = mag_to_idx.get('Large', 0)
    mag_weights[large_idx] = CONFIG['large_weight']
    print(f"\n⚖️  Magnitude weights: {mag_weights.tolist()}")
    
    azi_weights = torch.ones(len(azimuth_classes)).to(device)
    
    # Loss functions
    mag_criterion = FocalLoss(alpha=mag_weights, gamma=CONFIG['focal_gamma'])
    azi_criterion = FocalLoss(alpha=azi_weights, gamma=CONFIG['focal_gamma'])
    
    # Optimizer - AdamW for ConvNeXt
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # Training
    print(f"\n🚀 Starting training...")
    best_f1 = 0
    patience_counter = 0
    
    exp_name = f"exp_convnext_aug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = Path('experiments_augmented') / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    history = {'train_loss': [], 'val_loss': [], 'val_mag_f1': [], 'val_azi_f1': []}
    
    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for images, mag_labels, azi_labels in pbar:
            images = images.to(device)
            mag_labels = mag_labels.to(device)
            azi_labels = azi_labels.to(device)
            
            optimizer.zero_grad()
            mag_out, azi_out = model(images)
            
            loss_mag = mag_criterion(mag_out, mag_labels)
            loss_azi = azi_criterion(azi_out, azi_labels)
            loss = loss_mag + 0.5 * loss_azi
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_mag_preds, all_mag_labels = [], []
        all_azi_preds, all_azi_labels = [], []
        
        with torch.no_grad():
            for images, mag_labels, azi_labels in val_loader:
                images = images.to(device)
                mag_labels = mag_labels.to(device)
                azi_labels = azi_labels.to(device)
                
                mag_out, azi_out = model(images)
                
                loss_mag = mag_criterion(mag_out, mag_labels)
                loss_azi = azi_criterion(azi_out, azi_labels)
                loss = loss_mag + 0.5 * loss_azi
                val_loss += loss.item()
                
                all_mag_preds.extend(mag_out.argmax(dim=1).cpu().numpy())
                all_mag_labels.extend(mag_labels.cpu().numpy())
                all_azi_preds.extend(azi_out.argmax(dim=1).cpu().numpy())
                all_azi_labels.extend(azi_labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        mag_f1 = f1_score(all_mag_labels, all_mag_preds, average='macro')
        azi_f1 = f1_score(all_azi_labels, all_azi_preds, average='macro')
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mag_f1'].append(mag_f1)
        history['val_azi_f1'].append(azi_f1)
        
        # Check Large class performance
        large_correct = sum(1 for p, l in zip(all_mag_preds, all_mag_labels) if p == l == large_idx)
        large_total = sum(1 for l in all_mag_labels if l == large_idx)
        large_acc = large_correct / large_total * 100 if large_total > 0 else 0
        
        print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"  Mag F1={mag_f1:.4f}, Azi F1={azi_f1:.4f}")
        print(f"  🎯 Large class accuracy: {large_acc:.1f}% ({large_correct}/{large_total})")
        
        # Save best model
        if mag_f1 > best_f1:
            best_f1 = mag_f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'mag_f1': mag_f1,
                'azi_f1': azi_f1,
            }, exp_dir / 'best_model.pth')
            print(f"  ✅ New best model saved! F1={mag_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
    
    # Save class mappings
    class_mappings = {
        'magnitude_classes': magnitude_classes,
        'azimuth_classes': azimuth_classes,
        'magnitude_to_idx': mag_to_idx,
        'azimuth_to_idx': azi_to_idx
    }
    with open(exp_dir / 'class_mappings.json', 'w') as f:
        json.dump(class_mappings, f, indent=2)
    
    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"   Best Mag F1: {best_f1:.4f}")
    print(f"   Model saved to: {exp_dir}")

if __name__ == '__main__':
    main()
