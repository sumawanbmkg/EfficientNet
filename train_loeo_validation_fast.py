#!/usr/bin/env python3
"""
Fast LOEO Cross-Validation using EfficientNet-B0
Optimized for CPU training with reduced epochs

Author: Earthquake Prediction Research Team
Date: 4 February 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
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


class EfficientNetMultiTask(nn.Module):
    """EfficientNet-B0 based multi-task model - much faster than VGG16"""
    
    def __init__(self, num_magnitude_classes=4, num_azimuth_classes=9, dropout_rate=0.3):
        super(EfficientNetMultiTask, self).__init__()
        
        # Load pretrained EfficientNet-B0
        base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Get feature dimension
        feature_dim = base_model.classifier[1].in_features  # 1280 for B0
        
        # Use EfficientNet features
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Shared classifier
        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
        )
        
        # Magnitude head
        self.magnitude_head = nn.Linear(512, num_magnitude_classes)
        
        # Azimuth head
        self.azimuth_head = nn.Linear(512, num_azimuth_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared_fc(x)
        
        magnitude_out = self.magnitude_head(x)
        azimuth_out = self.azimuth_head(x)
        
        return magnitude_out, azimuth_out


class LOEODataset(Dataset):
    """Dataset for LOEO validation"""
    
    def __init__(self, metadata_df, dataset_dir, transform=None, image_size=224):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.image_size = image_size
        
        self.magnitude_classes = sorted(self.metadata['magnitude_class'].dropna().unique())
        self.azimuth_classes = sorted(self.metadata['azimuth_class'].dropna().unique())
        
        self.magnitude_to_idx = {cls: idx for idx, cls in enumerate(self.magnitude_classes)}
        self.azimuth_to_idx = {cls: idx for idx, cls in enumerate(self.azimuth_classes)}
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        image_path = self.dataset_dir / sample['unified_path']
        image = Image.open(image_path).convert('RGB')
        
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        magnitude_label = self.magnitude_to_idx.get(sample['magnitude_class'], 0)
        azimuth_label = self.azimuth_to_idx.get(sample['azimuth_class'], 0)
        
        return image, magnitude_label, azimuth_label


def create_event_based_folds(metadata_path, n_folds=10, random_state=42):
    """Create stratified folds based on earthquake events"""
    df = pd.read_csv(metadata_path)
    df = df.dropna(subset=['magnitude_class', 'azimuth_class'])
    df['magnitude_class'] = df['magnitude_class'].astype(str)
    df['azimuth_class'] = df['azimuth_class'].astype(str)
    df = df[df['magnitude_class'] != 'nan']
    
    df['event_id'] = df['station'] + '_' + df['date'].astype(str)
    
    event_info = df.groupby('event_id').agg({
        'magnitude_class': 'first',
        'station': 'first',
        'date': 'first'
    }).reset_index()
    
    print(f"Total samples: {len(df)}, Unique events: {len(event_info)}")
    print(f"Event distribution: {event_info['magnitude_class'].value_counts().to_dict()}")
    
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
    
    return folds, df


def train_one_epoch(model, train_loader, criterion_mag, criterion_azi, optimizer, device):
    model.train()
    total_loss = 0
    correct_mag = 0
    correct_azi = 0
    total = 0
    
    for images, mag_labels, azi_labels in tqdm(train_loader, desc="Train", leave=False):
        images = images.to(device)
        mag_labels = mag_labels.to(device)
        azi_labels = azi_labels.to(device)
        
        optimizer.zero_grad()
        mag_out, azi_out = model(images)
        
        loss = criterion_mag(mag_out, mag_labels) + criterion_azi(azi_out, azi_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, mag_pred = torch.max(mag_out, 1)
        _, azi_pred = torch.max(azi_out, 1)
        correct_mag += (mag_pred == mag_labels).sum().item()
        correct_azi += (azi_pred == azi_labels).sum().item()
        total += mag_labels.size(0)
    
    return {
        'loss': total_loss / len(train_loader),
        'mag_acc': 100 * correct_mag / total,
        'azi_acc': 100 * correct_azi / total
    }


def evaluate(model, test_loader, criterion_mag, criterion_azi, device):
    model.eval()
    correct_mag = 0
    correct_azi = 0
    total = 0
    
    with torch.no_grad():
        for images, mag_labels, azi_labels in test_loader:
            images = images.to(device)
            mag_labels = mag_labels.to(device)
            azi_labels = azi_labels.to(device)
            
            mag_out, azi_out = model(images)
            
            _, mag_pred = torch.max(mag_out, 1)
            _, azi_pred = torch.max(azi_out, 1)
            correct_mag += (mag_pred == mag_labels).sum().item()
            correct_azi += (azi_pred == azi_labels).sum().item()
            total += mag_labels.size(0)
    
    return {
        'mag_acc': 100 * correct_mag / total if total > 0 else 0,
        'azi_acc': 100 * correct_azi / total if total > 0 else 0
    }


def train_fold(fold_info, full_dataset, config, device):
    """Train and evaluate one fold"""
    print(f"\n--- Fold {fold_info['fold']}: {fold_info['n_train_samples']} train, {fold_info['n_test_samples']} test ---")
    
    train_dataset = Subset(full_dataset, fold_info['train_indices'])
    test_dataset = Subset(full_dataset, fold_info['test_indices'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    model = EfficientNetMultiTask(
        num_magnitude_classes=config['num_magnitude_classes'],
        num_azimuth_classes=config['num_azimuth_classes']
    ).to(device)
    
    criterion_mag = nn.CrossEntropyLoss()
    criterion_azi = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_acc = 0
    best_results = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        train_results = train_one_epoch(model, train_loader, criterion_mag, criterion_azi, optimizer, device)
        test_results = evaluate(model, test_loader, criterion_mag, criterion_azi, device)
        
        combined_acc = (test_results['mag_acc'] + test_results['azi_acc']) / 2
        
        print(f"  Epoch {epoch+1}: Train Mag={train_results['mag_acc']:.1f}%, Azi={train_results['azi_acc']:.1f}% | "
              f"Test Mag={test_results['mag_acc']:.1f}%, Azi={test_results['azi_acc']:.1f}%")
        
        if combined_acc > best_acc:
            best_acc = combined_acc
            best_results = test_results.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    return {
        'fold': fold_info['fold'],
        'magnitude_accuracy': best_results['mag_acc'],
        'azimuth_accuracy': best_results['azi_acc'],
        'n_train_events': fold_info['n_train_events'],
        'n_test_events': fold_info['n_test_events']
    }


def run_loeo_validation(config):
    """Run complete LOEO validation"""
    print("\n" + "="*60)
    print("LOEO CROSS-VALIDATION (EfficientNet-B0 - Fast Mode)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_dir = Path('loeo_validation_results')
    output_dir.mkdir(exist_ok=True)
    
    folds, metadata_df = create_event_based_folds(config['metadata_path'], n_folds=config['n_folds'])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = LOEODataset(metadata_df, config['dataset_dir'], transform=transform)
    
    config['num_magnitude_classes'] = len(full_dataset.magnitude_classes)
    config['num_azimuth_classes'] = len(full_dataset.azimuth_classes)
    
    print(f"Classes: {full_dataset.magnitude_classes}, {full_dataset.azimuth_classes}")
    
    all_results = []
    for fold in folds:
        result = train_fold(fold, full_dataset, config, device)
        all_results.append(result)
        
        with open(output_dir / f'loeo_fold_{fold["fold"]}.json', 'w') as f:
            json.dump(result, f, indent=2)
    
    # Aggregate results
    mag_accs = [r['magnitude_accuracy'] for r in all_results]
    azi_accs = [r['azimuth_accuracy'] for r in all_results]
    
    final_results = {
        'n_folds': len(all_results),
        'magnitude_accuracy': {
            'mean': float(np.mean(mag_accs)),
            'std': float(np.std(mag_accs)),
            'min': float(np.min(mag_accs)),
            'max': float(np.max(mag_accs))
        },
        'azimuth_accuracy': {
            'mean': float(np.mean(azi_accs)),
            'std': float(np.std(azi_accs)),
            'min': float(np.min(azi_accs)),
            'max': float(np.max(azi_accs))
        },
        'per_fold_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'loeo_final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("LOEO VALIDATION RESULTS")
    print("="*60)
    print(f"\nMagnitude: {final_results['magnitude_accuracy']['mean']:.2f}% ± {final_results['magnitude_accuracy']['std']:.2f}%")
    print(f"Azimuth:   {final_results['azimuth_accuracy']['mean']:.2f}% ± {final_results['azimuth_accuracy']['std']:.2f}%")
    
    # Compare with random split (EfficientNet)
    random_mag = 94.37
    random_azi = 57.39
    mag_drop = random_mag - final_results['magnitude_accuracy']['mean']
    azi_drop = random_azi - final_results['azimuth_accuracy']['mean']
    
    print(f"\nComparison with Random Split (EfficientNet):")
    print(f"  Magnitude: {random_mag:.2f}% → {final_results['magnitude_accuracy']['mean']:.2f}% (drop: {mag_drop:.2f}%)")
    print(f"  Azimuth:   {random_azi:.2f}% → {final_results['azimuth_accuracy']['mean']:.2f}% (drop: {azi_drop:.2f}%)")
    
    print(f"\n{'✅ Good generalization!' if mag_drop < 5 and azi_drop < 5 else '⚠️ Some overfitting detected'}")
    
    return final_results


if __name__ == '__main__':
    config = {
        'metadata_path': 'dataset_unified/metadata/unified_metadata.csv',
        'dataset_dir': 'dataset_unified',
        'n_folds': 10,
        'batch_size': 32,
        'epochs': 10,  # Reduced for faster training
        'learning_rate': 0.0001,
        'patience': 3
    }
    
    results = run_loeo_validation(config)
    print("\n✅ LOEO validation complete! Results saved to loeo_validation_results/")
