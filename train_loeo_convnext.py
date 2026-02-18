#!/usr/bin/env python3
"""
Leave-One-Event-Out (LOEO) Cross-Validation for ConvNeXt Model
Stratified 10-Fold for ConvNeXt-Tiny with On-The-Fly Augmentation

This script implements event-based cross-validation to prove
true generalization to unseen earthquake events using ConvNeXt architecture.

Features:
- On-the-fly data augmentation (flip, rotation, color jitter, etc.)
- Weighted sampling for imbalanced classes
- MixUp augmentation (optional)
- Focal Loss for class imbalance

Author: Earthquake Prediction Research Team
Date: 6 February 2026 (Updated with augmentation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'metadata_path': 'dataset_unified/metadata/unified_metadata.csv',
    'dataset_dir': 'dataset_unified',
    'n_folds': 10,
    'batch_size': 16,
    'epochs': 15,
    'learning_rate': 1e-4,
    'weight_decay': 0.05,
    'patience': 5,
    'dropout': 0.5,
    'image_size': 224,
    # Augmentation settings
    'use_augmentation': True,
    'use_mixup': True,
    'mixup_alpha': 0.4,
    'use_weighted_sampling': True,
    'use_focal_loss': True,
    'focal_gamma': 2.0,
}


# ============================================================================
# Data Augmentation Transforms
# ============================================================================

def get_train_transforms(image_size=224, use_augmentation=True):
    """
    Get training transforms with on-the-fly augmentation.
    
    Augmentation techniques:
    - RandomHorizontalFlip: 50% chance to flip horizontally
    - RandomRotation: ±15 degrees rotation
    - ColorJitter: brightness, contrast, saturation variation
    - RandomAffine: translation up to 10%
    - RandomErasing: 10% chance to erase random region
    """
    if use_augmentation:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),
        ])
    else:
        return get_val_transforms(image_size)


def get_val_transforms(image_size=224):
    """Get validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ============================================================================
# MixUp Augmentation
# ============================================================================

class MixUp:
    """
    MixUp augmentation for batch-level mixing.
    
    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
    """
    
    def __init__(self, alpha=0.4):
        self.alpha = alpha
    
    def __call__(self, images, mag_labels, azi_labels):
        """Apply MixUp to batch."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        
        return mixed_images, mag_labels, azi_labels, mag_labels[index], azi_labels[index], lam


# ============================================================================
# Focal Loss for Class Imbalance
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# ConvNeXt Multi-Task Model
# ============================================================================

class ConvNeXtMultiTask(nn.Module):
    """ConvNeXt model with multi-task heads for magnitude and azimuth"""
    
    def __init__(self, num_mag_classes=4, num_azi_classes=9, dropout=0.5, pretrained=True):
        super().__init__()
        
        # Load pretrained ConvNeXt-Tiny
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        num_features = 768
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Magnitude classification head
        self.mag_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_mag_classes)
        )
        
        # Azimuth classification head
        self.azi_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_azi_classes)
        )
        
        self.num_features = num_features
        
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.flatten(1)
        mag_out = self.mag_head(features)
        azi_out = self.azi_head(features)
        return mag_out, azi_out


# ============================================================================
# Dataset Class
# ============================================================================

class LOEODataset(Dataset):
    """Dataset for LOEO validation with augmentation support"""
    
    def __init__(self, metadata_df, dataset_dir, transform=None, image_size=224, is_training=True):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.image_size = image_size
        self.is_training = is_training
        
        self.magnitude_classes = sorted(self.metadata['magnitude_class'].dropna().unique())
        self.azimuth_classes = sorted(self.metadata['azimuth_class'].dropna().unique())
        
        self.magnitude_to_idx = {cls: idx for idx, cls in enumerate(self.magnitude_classes)}
        self.azimuth_to_idx = {cls: idx for idx, cls in enumerate(self.azimuth_classes)}
        
        # Calculate class weights for weighted sampling
        self.class_counts = self.metadata['magnitude_class'].value_counts().to_dict()
        self.sample_weights = self._calculate_sample_weights()
        
    def _calculate_sample_weights(self):
        """Calculate sample weights for balanced sampling."""
        weights = []
        for idx in range(len(self.metadata)):
            mag_class = self.metadata.iloc[idx]['magnitude_class']
            # Inverse frequency weighting
            weight = 1.0 / self.class_counts.get(mag_class, 1)
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        
        # Handle unified_path - check for NaN
        unified_path = sample.get('unified_path', None)
        if pd.isna(unified_path):
            # Fallback to spectrogram_file
            unified_path = f"spectrograms/original/{sample['spectrogram_file']}"
        
        image_path = self.dataset_dir / unified_path
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            image = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(image)
        
        magnitude_label = self.magnitude_to_idx.get(sample['magnitude_class'], 0)
        azimuth_label = self.azimuth_to_idx.get(sample['azimuth_class'], 0)
        
        return image, magnitude_label, azimuth_label


# ============================================================================
# Fold Creation
# ============================================================================

def create_event_based_folds(metadata_path, n_folds=10, random_state=42):
    """Create stratified folds based on earthquake events"""
    print(f"\n{'='*70}")
    print("CREATING EVENT-BASED FOLDS FOR CONVNEXT")
    print(f"{'='*70}\n")
    
    df = pd.read_csv(metadata_path)
    print(f"Total samples: {len(df)}")
    
    # Clean data - remove rows with NaN in critical columns
    df = df.dropna(subset=['magnitude_class', 'azimuth_class', 'unified_path'])
    df['magnitude_class'] = df['magnitude_class'].astype(str)
    df['azimuth_class'] = df['azimuth_class'].astype(str)
    df = df[df['magnitude_class'] != 'nan']
    df = df[df['unified_path'].notna()]
    print(f"After cleaning: {len(df)} samples")
    
    df['event_id'] = df['station'] + '_' + df['date'].astype(str)
    
    event_info = df.groupby('event_id').agg({
        'magnitude_class': 'first',
        'station': 'first',
        'date': 'first'
    }).reset_index()
    
    print(f"Unique events: {len(event_info)}")
    print(f"\nEvent distribution by magnitude:")
    print(event_info['magnitude_class'].value_counts())
    
    event_to_indices = {}
    for event_id in event_info['event_id']:
        event_to_indices[event_id] = df[df['event_id'] == event_id].index.tolist()
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_event_idx, test_event_idx) in enumerate(
        skf.split(event_info['event_id'], event_info['magnitude_class'])
    ):
        train_events = event_info.iloc[train_event_idx]['event_id'].values
        test_events = event_info.iloc[test_event_idx]['event_id'].values
        
        train_indices = []
        test_indices = []
        
        for event_id in train_events:
            train_indices.extend(event_to_indices[event_id])
        for event_id in test_events:
            test_indices.extend(event_to_indices[event_id])
        
        folds.append({
            'fold': fold_idx + 1,
            'train_events': train_events.tolist(),
            'test_events': test_events.tolist(),
            'train_indices': train_indices,
            'test_indices': test_indices,
            'n_train_events': len(train_events),
            'n_test_events': len(test_events),
            'n_train_samples': len(train_indices),
            'n_test_samples': len(test_indices)
        })
        
        print(f"Fold {fold_idx + 1}: Train {len(train_events)} events ({len(train_indices)} samples), Test {len(test_events)} events ({len(test_indices)} samples)")
    
    return folds, df


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, train_loader, criterion_mag, criterion_azi, optimizer, device, scaler=None, mixup=None):
    """Train for one epoch with mixed precision and optional MixUp"""
    model.train()
    total_loss = 0
    correct_mag = 0
    correct_azi = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, mag_labels, azi_labels in pbar:
        images = images.to(device)
        mag_labels = mag_labels.to(device)
        azi_labels = azi_labels.to(device)
        
        optimizer.zero_grad()
        
        # Apply MixUp if enabled
        use_mixup = mixup is not None and np.random.random() < 0.5
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                if use_mixup:
                    mixed_images, mag_a, azi_a, mag_b, azi_b, lam = mixup(images, mag_labels, azi_labels)
                    mag_out, azi_out = model(mixed_images)
                    loss_mag = lam * criterion_mag(mag_out, mag_a) + (1 - lam) * criterion_mag(mag_out, mag_b)
                    loss_azi = lam * criterion_azi(azi_out, azi_a) + (1 - lam) * criterion_azi(azi_out, azi_b)
                else:
                    mag_out, azi_out = model(images)
                    loss_mag = criterion_mag(mag_out, mag_labels)
                    loss_azi = criterion_azi(azi_out, azi_labels)
                loss = loss_mag + 0.5 * loss_azi
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if use_mixup:
                mixed_images, mag_a, azi_a, mag_b, azi_b, lam = mixup(images, mag_labels, azi_labels)
                mag_out, azi_out = model(mixed_images)
                loss_mag = lam * criterion_mag(mag_out, mag_a) + (1 - lam) * criterion_mag(mag_out, mag_b)
                loss_azi = lam * criterion_azi(azi_out, azi_a) + (1 - lam) * criterion_azi(azi_out, azi_b)
            else:
                mag_out, azi_out = model(images)
                loss_mag = criterion_mag(mag_out, mag_labels)
                loss_azi = criterion_azi(azi_out, azi_labels)
            loss = loss_mag + 0.5 * loss_azi
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        
        # For accuracy calculation, use original images (not mixed)
        if use_mixup:
            with torch.no_grad():
                mag_out_orig, azi_out_orig = model(images)
            _, mag_pred = torch.max(mag_out_orig, 1)
            _, azi_pred = torch.max(azi_out_orig, 1)
        else:
            _, mag_pred = torch.max(mag_out, 1)
            _, azi_pred = torch.max(azi_out, 1)
        
        correct_mag += (mag_pred == mag_labels).sum().item()
        correct_azi += (azi_pred == azi_labels).sum().item()
        total += mag_labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mag': f'{100*correct_mag/total:.1f}%', 'azi': f'{100*correct_azi/total:.1f}%'})
    
    return {'loss': total_loss / len(train_loader), 'mag_acc': 100 * correct_mag / total, 'azi_acc': 100 * correct_azi / total}


def evaluate(model, test_loader, criterion_mag, criterion_azi, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct_mag = 0
    correct_azi = 0
    total = 0
    
    with torch.no_grad():
        for images, mag_labels, azi_labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            mag_labels = mag_labels.to(device)
            azi_labels = azi_labels.to(device)
            
            mag_out, azi_out = model(images)
            loss_mag = criterion_mag(mag_out, mag_labels)
            loss_azi = criterion_azi(azi_out, azi_labels)
            loss = loss_mag + 0.5 * loss_azi
            
            total_loss += loss.item()
            _, mag_pred = torch.max(mag_out, 1)
            _, azi_pred = torch.max(azi_out, 1)
            correct_mag += (mag_pred == mag_labels).sum().item()
            correct_azi += (azi_pred == azi_labels).sum().item()
            total += mag_labels.size(0)
    
    return {
        'loss': total_loss / len(test_loader) if len(test_loader) > 0 else 0,
        'mag_acc': 100 * correct_mag / total if total > 0 else 0,
        'azi_acc': 100 * correct_azi / total if total > 0 else 0
    }


# ============================================================================
# Train and Evaluate Fold
# ============================================================================

def train_and_evaluate_fold(fold_info, full_dataset, metadata_df, config, device):
    """Train and evaluate one fold with augmentation"""
    print(f"\n{'='*70}")
    print(f"FOLD {fold_info['fold']} - ConvNeXt-Tiny with Augmentation")
    print(f"{'='*70}")
    print(f"Train: {fold_info['n_train_events']} events, {fold_info['n_train_samples']} samples")
    print(f"Test:  {fold_info['n_test_events']} events, {fold_info['n_test_samples']} samples")
    print(f"Augmentation: {config['use_augmentation']}, MixUp: {config['use_mixup']}, Focal Loss: {config['use_focal_loss']}")
    
    # Create train dataset with augmentation
    train_metadata = metadata_df.iloc[fold_info['train_indices']].reset_index(drop=True)
    test_metadata = metadata_df.iloc[fold_info['test_indices']].reset_index(drop=True)
    
    train_transform = get_train_transforms(config['image_size'], use_augmentation=config['use_augmentation'])
    val_transform = get_val_transforms(config['image_size'])
    
    train_dataset = LOEODataset(
        metadata_df=train_metadata,
        dataset_dir=config['dataset_dir'],
        transform=train_transform,
        image_size=config['image_size'],
        is_training=True
    )
    
    test_dataset = LOEODataset(
        metadata_df=test_metadata,
        dataset_dir=config['dataset_dir'],
        transform=val_transform,
        image_size=config['image_size'],
        is_training=False
    )
    
    # Create weighted sampler for imbalanced classes
    if config['use_weighted_sampling']:
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            sampler=sampler,
            num_workers=0, 
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    model = ConvNeXtMultiTask(
        num_mag_classes=config['num_magnitude_classes'],
        num_azi_classes=config['num_azimuth_classes'],
        dropout=config['dropout']
    ).to(device)
    
    # Use Focal Loss or CrossEntropy
    if config['use_focal_loss']:
        criterion_mag = FocalLoss(gamma=config['focal_gamma'])
        criterion_azi = FocalLoss(gamma=config['focal_gamma'])
        print("Using Focal Loss for class imbalance")
    else:
        criterion_mag = nn.CrossEntropyLoss()
        criterion_azi = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Initialize MixUp if enabled
    mixup = MixUp(alpha=config['mixup_alpha']) if config['use_mixup'] else None
    
    best_test_acc = 0
    best_results = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        train_results = train_one_epoch(model, train_loader, criterion_mag, criterion_azi, optimizer, device, scaler, mixup)
        test_results = evaluate(model, test_loader, criterion_mag, criterion_azi, device)
        scheduler.step()
        
        combined_acc = (test_results['mag_acc'] + test_results['azi_acc']) / 2
        
        print(f"Epoch {epoch+1}/{config['epochs']}: Train Mag={train_results['mag_acc']:.2f}%, Azi={train_results['azi_acc']:.2f}% | Test Mag={test_results['mag_acc']:.2f}%, Azi={test_results['azi_acc']:.2f}%")
        
        if combined_acc > best_test_acc:
            best_test_acc = combined_acc
            best_results = test_results.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return {
        'fold': fold_info['fold'],
        'magnitude_accuracy': best_results['mag_acc'],
        'azimuth_accuracy': best_results['azi_acc'],
        'combined_accuracy': (best_results['mag_acc'] + best_results['azi_acc']) / 2,
        'n_train_events': fold_info['n_train_events'],
        'n_test_events': fold_info['n_test_events'],
        'n_train_samples': fold_info['n_train_samples'],
        'n_test_samples': fold_info['n_test_samples'],
        'augmentation_used': config['use_augmentation'],
        'mixup_used': config['use_mixup'],
        'focal_loss_used': config['use_focal_loss']
    }


# ============================================================================
# Main LOEO Validation
# ============================================================================

def run_loeo_validation():
    """Run complete LOEO validation for ConvNeXt with augmentation"""
    print(f"\n{'='*70}")
    print("LEAVE-ONE-EVENT-OUT CROSS-VALIDATION - ConvNeXt-Tiny")
    print("WITH ON-THE-FLY AUGMENTATION")
    print(f"{'='*70}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Print augmentation settings
    print(f"\nAugmentation Settings:")
    print(f"  - On-the-fly augmentation: {CONFIG['use_augmentation']}")
    print(f"  - MixUp (alpha={CONFIG['mixup_alpha']}): {CONFIG['use_mixup']}")
    print(f"  - Weighted sampling: {CONFIG['use_weighted_sampling']}")
    print(f"  - Focal Loss (gamma={CONFIG['focal_gamma']}): {CONFIG['use_focal_loss']}")
    
    output_dir = Path('loeo_convnext_results')
    output_dir.mkdir(exist_ok=True)
    
    folds, metadata_df = create_event_based_folds(CONFIG['metadata_path'], n_folds=CONFIG['n_folds'])
    
    # Create a dummy dataset just to get class info
    dummy_transform = get_val_transforms(CONFIG['image_size'])
    dummy_dataset = LOEODataset(
        metadata_df=metadata_df, 
        dataset_dir=CONFIG['dataset_dir'], 
        transform=dummy_transform, 
        image_size=CONFIG['image_size']
    )
    
    CONFIG['num_magnitude_classes'] = len(dummy_dataset.magnitude_classes)
    CONFIG['num_azimuth_classes'] = len(dummy_dataset.azimuth_classes)
    
    print(f"\nDataset: {len(metadata_df)} samples")
    print(f"Magnitude classes: {dummy_dataset.magnitude_classes}")
    print(f"Azimuth classes: {dummy_dataset.azimuth_classes}")
    
    # Print class distribution
    print(f"\nClass distribution:")
    for cls, count in dummy_dataset.class_counts.items():
        print(f"  {cls}: {count} samples")
    
    all_results = []
    
    for fold in folds:
        result = train_and_evaluate_fold(fold, dummy_dataset, metadata_df, CONFIG, device)
        all_results.append(result)
        
        with open(output_dir / f'fold_{fold["fold"]}_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n[OK] Fold {fold['fold']} complete: Mag={result['magnitude_accuracy']:.2f}%, Azi={result['azimuth_accuracy']:.2f}%")
    
    # Aggregate results
    mag_accs = [r['magnitude_accuracy'] for r in all_results]
    azi_accs = [r['azimuth_accuracy'] for r in all_results]
    
    final_results = {
        'model': 'ConvNeXt-Tiny',
        'n_folds': len(all_results),
        'augmentation_settings': {
            'use_augmentation': CONFIG['use_augmentation'],
            'use_mixup': CONFIG['use_mixup'],
            'mixup_alpha': CONFIG['mixup_alpha'],
            'use_weighted_sampling': CONFIG['use_weighted_sampling'],
            'use_focal_loss': CONFIG['use_focal_loss'],
            'focal_gamma': CONFIG['focal_gamma'],
        },
        'magnitude_accuracy': {
            'mean': float(np.mean(mag_accs)), 'std': float(np.std(mag_accs)),
            'min': float(np.min(mag_accs)), 'max': float(np.max(mag_accs)),
        },
        'azimuth_accuracy': {
            'mean': float(np.mean(azi_accs)), 'std': float(np.std(azi_accs)),
            'min': float(np.min(azi_accs)), 'max': float(np.max(azi_accs)),
        },
        'per_fold_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'loeo_convnext_final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("CONVNEXT LOEO VALIDATION RESULTS (WITH AUGMENTATION)")
    print(f"{'='*70}")
    print(f"\nMagnitude Accuracy: {np.mean(mag_accs):.2f}% ± {np.std(mag_accs):.2f}%")
    print(f"Azimuth Accuracy:   {np.mean(azi_accs):.2f}% ± {np.std(azi_accs):.2f}%")
    print(f"\nResults saved to: {output_dir}/")
    
    # Generate report
    generate_report(final_results, output_dir)
    
    return final_results


def generate_report(results, output_dir):
    """Generate markdown report"""
    mag = results['magnitude_accuracy']
    azi = results['azimuth_accuracy']
    aug = results.get('augmentation_settings', {})
    
    report = f"""# ConvNeXt LOEO Cross-Validation Report

**Model**: ConvNeXt-Tiny  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Validation**: Leave-One-Event-Out (LOEO) 10-Fold

---

## Augmentation Settings

| Setting | Value |
|---------|-------|
| On-the-fly Augmentation | {aug.get('use_augmentation', True)} |
| MixUp | {aug.get('use_mixup', True)} (alpha={aug.get('mixup_alpha', 0.4)}) |
| Weighted Sampling | {aug.get('use_weighted_sampling', True)} |
| Focal Loss | {aug.get('use_focal_loss', True)} (gamma={aug.get('focal_gamma', 2.0)}) |

### Augmentation Techniques Applied:
- **RandomHorizontalFlip**: 50% probability
- **RandomRotation**: ±15 degrees
- **ColorJitter**: brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
- **RandomAffine**: translate=10%, scale=90-110%
- **RandomErasing**: 10% probability

---

## Summary Results

| Task | Mean Accuracy | Std Dev | Min | Max |
|------|---------------|---------|-----|-----|
| Magnitude | **{mag['mean']:.2f}%** | ±{mag['std']:.2f}% | {mag['min']:.2f}% | {mag['max']:.2f}% |
| Azimuth | **{azi['mean']:.2f}%** | ±{azi['std']:.2f}% | {azi['min']:.2f}% | {azi['max']:.2f}% |

## Per-Fold Results

| Fold | Magnitude | Azimuth | Train Events | Test Events |
|------|-----------|---------|--------------|-------------|
"""
    
    for r in results['per_fold_results']:
        report += f"| {r['fold']} | {r['magnitude_accuracy']:.2f}% | {r['azimuth_accuracy']:.2f}% | {r['n_train_events']} | {r['n_test_events']} |\n"
    
    report += f"""
## Comparison: With vs Without Augmentation

| Model | Augmentation | Magnitude | Azimuth |
|-------|--------------|-----------|---------|
| ConvNeXt-Tiny | Without | 97.53% ± 0.96% | 69.30% ± 5.74% |
| ConvNeXt-Tiny | **With** | **{mag['mean']:.2f}% ± {mag['std']:.2f}%** | **{azi['mean']:.2f}% ± {azi['std']:.2f}%** |

## Model Configuration

- **Architecture**: ConvNeXt-Tiny (28.6M parameters)
- **Pretrained**: ImageNet-1K
- **Optimizer**: AdamW (lr={CONFIG['learning_rate']}, weight_decay={CONFIG['weight_decay']})
- **Scheduler**: Cosine Annealing
- **Epochs**: {CONFIG['epochs']} (with early stopping)
- **Batch Size**: {CONFIG['batch_size']}
- **Dropout**: {CONFIG['dropout']}

---

Generated: {datetime.now().isoformat()}
"""
    
    with open(output_dir / 'LOEO_CONVNEXT_REPORT.md', 'w') as f:
        f.write(report)
    
    print(f"\n[REPORT] Report saved to: {output_dir / 'LOEO_CONVNEXT_REPORT.md'}")


if __name__ == '__main__':
    print("="*70)
    print("CONVNEXT LOEO CROSS-VALIDATION WITH ON-THE-FLY AUGMENTATION")
    print("="*70)
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    print(f"\n{'='*70}")
    print("AUGMENTATION TECHNIQUES:")
    print("  1. RandomHorizontalFlip (p=0.5)")
    print("  2. RandomRotation (±15°)")
    print("  3. ColorJitter (brightness=0.2, contrast=0.2, saturation=0.1)")
    print("  4. RandomAffine (translate=10%, scale=90-110%)")
    print("  5. RandomErasing (p=0.1)")
    print("  6. MixUp (alpha=0.4) - batch-level mixing")
    print("  7. Focal Loss (gamma=2.0) - for class imbalance")
    print("  8. Weighted Sampling - oversample minority classes")
    print("="*70)
    
    results = run_loeo_validation()
    
    print("\n" + "="*70)
    print("[OK] CONVNEXT LOEO VALIDATION WITH AUGMENTATION COMPLETE!")
    print("="*70)
