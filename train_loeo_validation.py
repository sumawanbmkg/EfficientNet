#!/usr/bin/env python3
"""
Leave-One-Event-Out (LOEO) Cross-Validation
Stratified 10-Fold for VGG16 Model

This script implements event-based cross-validation to prove
true generalization to unseen earthquake events.

Author: Earthquake Prediction Research Team
Date: 4 February 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
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


# ============================================================================
# VGG16 Multi-Task Model (same as training)
# ============================================================================

class VGG16MultiTask(nn.Module):
    """VGG16-based multi-task model for magnitude and azimuth classification"""
    
    def __init__(self, num_magnitude_classes=4, num_azimuth_classes=9, dropout_rate=0.5):
        super(VGG16MultiTask, self).__init__()
        
        # Load pretrained VGG16
        base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Use VGG16 features (convolutional layers)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # Shared classifier layers
        self.shared_fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
        )
        
        # Magnitude classification head
        self.magnitude_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, num_magnitude_classes)
        )
        
        # Azimuth classification head
        self.azimuth_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, num_azimuth_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared_fc(x)
        
        magnitude_out = self.magnitude_head(x)
        azimuth_out = self.azimuth_head(x)
        
        return magnitude_out, azimuth_out


# ============================================================================
# Dataset Class for LOEO
# ============================================================================

class LOEODataset(Dataset):
    """Dataset for LOEO validation with index-based access"""
    
    def __init__(self, metadata_df, dataset_dir, transform=None, image_size=224):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Create label mappings
        self.magnitude_classes = sorted(self.metadata['magnitude_class'].dropna().unique())
        self.azimuth_classes = sorted(self.metadata['azimuth_class'].dropna().unique())
        
        self.magnitude_to_idx = {cls: idx for idx, cls in enumerate(self.magnitude_classes)}
        self.azimuth_to_idx = {cls: idx for idx, cls in enumerate(self.azimuth_classes)}
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        
        # Load image
        image_path = self.dataset_dir / sample['unified_path']
        image = Image.open(image_path).convert('RGB')
        
        # Resize
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Get labels
        magnitude_label = self.magnitude_to_idx.get(sample['magnitude_class'], 0)
        azimuth_label = self.azimuth_to_idx.get(sample['azimuth_class'], 0)
        
        return image, magnitude_label, azimuth_label


# ============================================================================
# LOEO Fold Creation
# ============================================================================

def create_event_based_folds(metadata_path, n_folds=10, random_state=42):
    """Create stratified folds based on earthquake events"""
    print(f"\n{'='*70}")
    print("CREATING EVENT-BASED FOLDS")
    print(f"{'='*70}\n")
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    print(f"Total samples: {len(df)}")
    
    # Clean data
    df = df.dropna(subset=['magnitude_class', 'azimuth_class'])
    df['magnitude_class'] = df['magnitude_class'].astype(str)
    df['azimuth_class'] = df['azimuth_class'].astype(str)
    df = df[df['magnitude_class'] != 'nan']
    print(f"After cleaning: {len(df)} samples")
    
    # Create event ID from date + station (unique earthquake event)
    df['event_id'] = df['station'] + '_' + df['date'].astype(str)
    
    # Get unique events with their magnitude class
    event_info = df.groupby('event_id').agg({
        'magnitude_class': 'first',
        'station': 'first',
        'date': 'first'
    }).reset_index()
    
    print(f"Unique events: {len(event_info)}")
    print(f"\nEvent distribution by magnitude:")
    print(event_info['magnitude_class'].value_counts())
    
    # Create event to indices mapping
    event_to_indices = {}
    for event_id in event_info['event_id']:
        event_to_indices[event_id] = df[df['event_id'] == event_id].index.tolist()
    
    # Stratified K-Fold by magnitude class
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    for fold_idx, (train_event_idx, test_event_idx) in enumerate(
        skf.split(event_info['event_id'], event_info['magnitude_class'])
    ):
        train_events = event_info.iloc[train_event_idx]['event_id'].values
        test_events = event_info.iloc[test_event_idx]['event_id'].values
        
        # Get sample indices for each event
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
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train: {len(train_events)} events, {len(train_indices)} samples")
        print(f"  Test:  {len(test_events)} events, {len(test_indices)} samples")
    
    return folds, df


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, train_loader, criterion_mag, criterion_azi, optimizer, device):
    """Train for one epoch"""
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
        
        mag_out, azi_out = model(images)
        
        loss_mag = criterion_mag(mag_out, mag_labels)
        loss_azi = criterion_azi(azi_out, azi_labels)
        loss = loss_mag + loss_azi
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, mag_pred = torch.max(mag_out, 1)
        _, azi_pred = torch.max(azi_out, 1)
        
        correct_mag += (mag_pred == mag_labels).sum().item()
        correct_azi += (azi_pred == azi_labels).sum().item()
        total += mag_labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mag': f'{100*correct_mag/total:.1f}%',
            'azi': f'{100*correct_azi/total:.1f}%'
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'mag_acc': 100 * correct_mag / total,
        'azi_acc': 100 * correct_azi / total
    }


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
            loss = loss_mag + loss_azi
            
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


def train_and_evaluate_fold(fold_info, full_dataset, config, device):
    """Train and evaluate one fold"""
    print(f"\n{'='*70}")
    print(f"FOLD {fold_info['fold']}")
    print(f"{'='*70}")
    print(f"Train: {fold_info['n_train_events']} events, {fold_info['n_train_samples']} samples")
    print(f"Test:  {fold_info['n_test_events']} events, {fold_info['n_test_samples']} samples")
    
    # Create data loaders using Subset
    train_dataset = Subset(full_dataset, fold_info['train_indices'])
    test_dataset = Subset(full_dataset, fold_info['test_indices'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize model
    model = VGG16MultiTask(
        num_magnitude_classes=config['num_magnitude_classes'],
        num_azimuth_classes=config['num_azimuth_classes']
    ).to(device)
    
    # Loss functions
    criterion_mag = nn.CrossEntropyLoss()
    criterion_azi = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training loop
    best_test_acc = 0
    best_results = None
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        train_results = train_one_epoch(
            model, train_loader, criterion_mag, criterion_azi, optimizer, device
        )
        
        test_results = evaluate(
            model, test_loader, criterion_mag, criterion_azi, device
        )
        
        combined_acc = (test_results['mag_acc'] + test_results['azi_acc']) / 2
        
        print(f"Epoch {epoch+1}/{config['epochs']}: "
              f"Train Mag={train_results['mag_acc']:.2f}%, Azi={train_results['azi_acc']:.2f}% | "
              f"Test Mag={test_results['mag_acc']:.2f}%, Azi={test_results['azi_acc']:.2f}%")
        
        # Update scheduler
        scheduler.step(combined_acc)
        
        # Early stopping
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
        'n_test_samples': fold_info['n_test_samples']
    }


# ============================================================================
# Main LOEO Validation
# ============================================================================

def run_loeo_validation(config):
    """Run complete LOEO validation"""
    print(f"\n{'='*70}")
    print("LEAVE-ONE-EVENT-OUT CROSS-VALIDATION")
    print(f"{'='*70}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directory
    output_dir = Path('loeo_validation_results')
    output_dir.mkdir(exist_ok=True)
    
    # Create folds
    folds, metadata_df = create_event_based_folds(
        config['metadata_path'],
        n_folds=config['n_folds']
    )
    
    # Save fold information
    folds_serializable = []
    for fold in folds:
        fold_copy = {k: v for k, v in fold.items() if k not in ['train_indices', 'test_indices']}
        folds_serializable.append(fold_copy)
    
    with open(output_dir / 'loeo_folds.json', 'w') as f:
        json.dump(folds_serializable, f, indent=2)
    
    print(f"\n✅ Fold information saved to: {output_dir / 'loeo_folds.json'}")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset
    full_dataset = LOEODataset(
        metadata_df=metadata_df,
        dataset_dir=config['dataset_dir'],
        transform=transform,
        image_size=224
    )
    
    # Update config with actual class counts
    config['num_magnitude_classes'] = len(full_dataset.magnitude_classes)
    config['num_azimuth_classes'] = len(full_dataset.azimuth_classes)
    
    print(f"\nDataset info:")
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Magnitude classes: {full_dataset.magnitude_classes}")
    print(f"  Azimuth classes: {full_dataset.azimuth_classes}")
    
    # Run training for each fold
    all_results = []
    
    for fold in folds:
        result = train_and_evaluate_fold(fold, full_dataset, config, device)
        all_results.append(result)
        
        # Save intermediate results
        with open(output_dir / f'loeo_results_fold_{fold["fold"]}.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✅ Fold {fold['fold']} complete: Mag={result['magnitude_accuracy']:.2f}%, Azi={result['azimuth_accuracy']:.2f}%")
    
    # Aggregate results
    final_results = aggregate_results(all_results)
    
    # Save final results
    with open(output_dir / 'loeo_final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Generate report
    generate_report(final_results, output_dir)
    
    # Print summary
    print_results_summary(final_results)
    
    return final_results


def aggregate_results(all_results):
    """Aggregate results from all folds"""
    mag_accs = [r['magnitude_accuracy'] for r in all_results]
    azi_accs = [r['azimuth_accuracy'] for r in all_results]
    
    return {
        'n_folds': len(all_results),
        'magnitude_accuracy': {
            'mean': float(np.mean(mag_accs)),
            'std': float(np.std(mag_accs)),
            'min': float(np.min(mag_accs)),
            'max': float(np.max(mag_accs)),
            'ci_95_lower': float(np.mean(mag_accs) - 1.96 * np.std(mag_accs) / np.sqrt(len(mag_accs))),
            'ci_95_upper': float(np.mean(mag_accs) + 1.96 * np.std(mag_accs) / np.sqrt(len(mag_accs)))
        },
        'azimuth_accuracy': {
            'mean': float(np.mean(azi_accs)),
            'std': float(np.std(azi_accs)),
            'min': float(np.min(azi_accs)),
            'max': float(np.max(azi_accs)),
            'ci_95_lower': float(np.mean(azi_accs) - 1.96 * np.std(azi_accs) / np.sqrt(len(azi_accs))),
            'ci_95_upper': float(np.mean(azi_accs) + 1.96 * np.std(azi_accs) / np.sqrt(len(azi_accs)))
        },
        'per_fold_results': all_results,
        'timestamp': datetime.now().isoformat()
    }


def print_results_summary(results):
    """Print formatted results summary"""
    print("\n" + "="*70)
    print("LOEO VALIDATION RESULTS")
    print("="*70)
    
    mag = results['magnitude_accuracy']
    azi = results['azimuth_accuracy']
    
    print(f"\nMagnitude Accuracy:")
    print(f"  Mean: {mag['mean']:.2f}%")
    print(f"  Std:  ±{mag['std']:.2f}%")
    print(f"  Range: [{mag['min']:.2f}%, {mag['max']:.2f}%]")
    print(f"  95% CI: [{mag['ci_95_lower']:.2f}%, {mag['ci_95_upper']:.2f}%]")
    
    print(f"\nAzimuth Accuracy:")
    print(f"  Mean: {azi['mean']:.2f}%")
    print(f"  Std:  ±{azi['std']:.2f}%")
    print(f"  Range: [{azi['min']:.2f}%, {azi['max']:.2f}%]")
    print(f"  95% CI: [{azi['ci_95_lower']:.2f}%, {azi['ci_95_upper']:.2f}%]")
    
    # Compare with random split
    random_split_mag = 98.68
    random_split_azi = 54.93
    
    mag_drop = random_split_mag - mag['mean']
    azi_drop = random_split_azi - azi['mean']
    
    print(f"\nComparison with Random Split (VGG16):")
    print(f"  Magnitude: {random_split_mag:.2f}% → {mag['mean']:.2f}% (drop: {mag_drop:.2f}%)")
    print(f"  Azimuth:   {random_split_azi:.2f}% → {azi['mean']:.2f}% (drop: {azi_drop:.2f}%)")
    
    if mag_drop < 5:
        print(f"\n✅ Magnitude drop < 5%: ACCEPTABLE - Good generalization!")
    else:
        print(f"\n⚠️  Magnitude drop ≥ 5%: SIGNIFICANT - May indicate overfitting")
    
    if azi_drop < 5:
        print(f"✅ Azimuth drop < 5%: ACCEPTABLE - Good generalization!")
    else:
        print(f"⚠️  Azimuth drop ≥ 5%: SIGNIFICANT - May indicate overfitting")
    
    print("\n" + "="*70)


def generate_report(results, output_dir):
    """Generate markdown report"""
    mag = results['magnitude_accuracy']
    azi = results['azimuth_accuracy']
    
    random_split_mag = 98.68
    random_split_azi = 54.93
    mag_drop = random_split_mag - mag['mean']
    azi_drop = random_split_azi - azi['mean']
    
    report = f"""# LOEO Cross-Validation Report

## Summary

| Metric | LOEO Mean | LOEO Std | Random Split | Drop |
|--------|-----------|----------|--------------|------|
| Magnitude Accuracy | {mag['mean']:.2f}% | ±{mag['std']:.2f}% | {random_split_mag:.2f}% | {mag_drop:.2f}% |
| Azimuth Accuracy | {azi['mean']:.2f}% | ±{azi['std']:.2f}% | {random_split_azi:.2f}% | {azi_drop:.2f}% |

## Interpretation

- **Magnitude**: {'✅ Good generalization' if mag_drop < 5 else '⚠️ Significant drop'}
- **Azimuth**: {'✅ Good generalization' if azi_drop < 5 else '⚠️ Significant drop'}

## Per-Fold Results

| Fold | Magnitude | Azimuth | Train Events | Test Events |
|------|-----------|---------|--------------|-------------|
"""
    
    for r in results['per_fold_results']:
        report += f"| {r['fold']} | {r['magnitude_accuracy']:.2f}% | {r['azimuth_accuracy']:.2f}% | {r['n_train_events']} | {r['n_test_events']} |\n"
    
    report += f"""
## Statistical Analysis

### Magnitude Accuracy
- Mean: {mag['mean']:.2f}%
- Standard Deviation: {mag['std']:.2f}%
- 95% Confidence Interval: [{mag['ci_95_lower']:.2f}%, {mag['ci_95_upper']:.2f}%]
- Range: [{mag['min']:.2f}%, {mag['max']:.2f}%]

### Azimuth Accuracy
- Mean: {azi['mean']:.2f}%
- Standard Deviation: {azi['std']:.2f}%
- 95% Confidence Interval: [{azi['ci_95_lower']:.2f}%, {azi['ci_95_upper']:.2f}%]
- Range: [{azi['min']:.2f}%, {azi['max']:.2f}%]

## Conclusion

The LOEO validation demonstrates that the model {'generalizes well' if mag_drop < 5 and azi_drop < 5 else 'may have some overfitting'} to unseen earthquake events.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(output_dir / 'LOEO_VALIDATION_REPORT.md', 'w') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: {output_dir / 'LOEO_VALIDATION_REPORT.md'}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    # Configuration
    config = {
        'metadata_path': 'dataset_unified/metadata/unified_metadata.csv',
        'dataset_dir': 'dataset_unified',
        'n_folds': 10,
        'batch_size': 16,  # Smaller batch for memory efficiency
        'epochs': 15,      # Reduced for faster validation
        'learning_rate': 0.0001,
        'patience': 5,
        'num_magnitude_classes': 4,  # Will be updated from data
        'num_azimuth_classes': 9     # Will be updated from data
    }
    
    print("="*70)
    print("LOEO CROSS-VALIDATION FOR EARTHQUAKE PRECURSOR CNN")
    print("="*70)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Run LOEO validation
    results = run_loeo_validation(config)
    
    print("\n" + "="*70)
    print("✅ LOEO VALIDATION COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: loeo_validation_results/")
    print(f"  - loeo_folds.json")
    print(f"  - loeo_final_results.json")
    print(f"  - LOEO_VALIDATION_REPORT.md")
    print(f"  - loeo_results_fold_*.json (per-fold results)")
