#!/usr/bin/env python3
"""
Continue LOSO Validation from Fold 8
Completes the remaining folds and generates final report
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
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class EfficientNetMultiTask(nn.Module):
    """EfficientNet-B0 based multi-task model"""
    
    def __init__(self, num_magnitude_classes=4, num_azimuth_classes=9, dropout_rate=0.5):
        super(EfficientNetMultiTask, self).__init__()
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(1280, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_rate)
        )
        self.magnitude_head = nn.Linear(512, num_magnitude_classes)
        self.azimuth_head = nn.Linear(512, num_azimuth_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.shared(x)
        return self.magnitude_head(x), self.azimuth_head(x)


class LOSODataset(Dataset):
    def __init__(self, metadata_df, dataset_dir, transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
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
        if image.size != (224, 224):
            image = image.resize((224, 224), Image.LANCZOS)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, self.magnitude_to_idx.get(sample['magnitude_class'], 0), self.azimuth_to_idx.get(sample['azimuth_class'], 0)


def train_one_epoch(model, train_loader, criterion_mag, criterion_azi, optimizer, device):
    model.train()
    total_loss, correct_mag, correct_azi, total = 0, 0, 0, 0
    for images, mag_labels, azi_labels in tqdm(train_loader, desc="Train", leave=False):
        images, mag_labels, azi_labels = images.to(device), mag_labels.to(device), azi_labels.to(device)
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
    return {'loss': total_loss / len(train_loader), 'mag_acc': 100 * correct_mag / total, 'azi_acc': 100 * correct_azi / total}


def evaluate(model, test_loader, device):
    model.eval()
    correct_mag, correct_azi, total = 0, 0, 0
    with torch.no_grad():
        for images, mag_labels, azi_labels in test_loader:
            images, mag_labels, azi_labels = images.to(device), mag_labels.to(device), azi_labels.to(device)
            mag_out, azi_out = model(images)
            _, mag_pred = torch.max(mag_out, 1)
            _, azi_pred = torch.max(azi_out, 1)
            correct_mag += (mag_pred == mag_labels).sum().item()
            correct_azi += (azi_pred == azi_labels).sum().item()
            total += mag_labels.size(0)
    return {'mag_acc': 100 * correct_mag / total if total > 0 else 0, 'azi_acc': 100 * correct_azi / total if total > 0 else 0}


def train_fold(fold_info, full_dataset, config, device):
    print(f"\n--- Fold {fold_info['fold']}: Test on {fold_info['test_station']} ---")
    print(f"    Train: {fold_info['n_train_samples']} samples, Test: {fold_info['n_test_samples']} samples")
    
    train_dataset = Subset(full_dataset, fold_info['train_indices'])
    test_dataset = Subset(full_dataset, fold_info['test_indices'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    model = EfficientNetMultiTask(
        num_magnitude_classes=config['num_magnitude_classes'],
        num_azimuth_classes=config['num_azimuth_classes']
    ).to(device)
    
    criterion_mag, criterion_azi = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_acc, best_results, patience_counter = 0, None, 0
    
    for epoch in range(config['epochs']):
        train_results = train_one_epoch(model, train_loader, criterion_mag, criterion_azi, optimizer, device)
        test_results = evaluate(model, test_loader, device)
        combined_acc = (test_results['mag_acc'] + test_results['azi_acc']) / 2
        
        print(f"  Epoch {epoch+1}: Train Mag={train_results['mag_acc']:.1f}%, Azi={train_results['azi_acc']:.1f}% | "
              f"Test Mag={test_results['mag_acc']:.1f}%, Azi={test_results['azi_acc']:.1f}%")
        
        if combined_acc > best_acc:
            best_acc, best_results, patience_counter = combined_acc, test_results.copy(), 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    return {
        'fold': fold_info['fold'],
        'test_station': fold_info['test_station'],
        'magnitude_accuracy': best_results['mag_acc'],
        'azimuth_accuracy': best_results['azi_acc'],
        'n_train_samples': fold_info['n_train_samples'],
        'n_test_samples': fold_info['n_test_samples']
    }


def create_station_based_folds(metadata_path):
    df = pd.read_csv(metadata_path)
    df = df.dropna(subset=['magnitude_class', 'azimuth_class'])
    df['magnitude_class'] = df['magnitude_class'].astype(str)
    df['azimuth_class'] = df['azimuth_class'].astype(str)
    df = df[df['magnitude_class'] != 'nan']
    
    stations = df['station'].unique()
    station_stats = df.groupby('station').agg({'magnitude_class': 'count'}).rename(columns={'magnitude_class': 'n_samples'})
    
    large_stations = station_stats[station_stats['n_samples'] >= 50].index.tolist()
    small_stations = station_stats[station_stats['n_samples'] < 50].index.tolist()
    
    folds = []
    for station in large_stations:
        test_indices = df[df['station'] == station].index.tolist()
        train_indices = df[df['station'] != station].index.tolist()
        folds.append({
            'fold': len(folds) + 1,
            'test_station': station,
            'train_stations': [s for s in stations if s != station],
            'train_indices': train_indices,
            'test_indices': test_indices,
            'n_train_samples': len(train_indices),
            'n_test_samples': len(test_indices)
        })
    
    if small_stations:
        test_indices = df[df['station'].isin(small_stations)].index.tolist()
        train_indices = df[~df['station'].isin(small_stations)].index.tolist()
        folds.append({
            'fold': len(folds) + 1,
            'test_station': f"Small stations ({len(small_stations)})",
            'train_stations': large_stations,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'n_train_samples': len(train_indices),
            'n_test_samples': len(test_indices)
        })
    
    return folds, df


def continue_loso_validation():
    print("\n" + "="*70)
    print("CONTINUING LOSO VALIDATION FROM FOLD 8")
    print("="*70)
    
    config = {
        'metadata_path': 'dataset_unified/metadata/unified_metadata.csv',
        'dataset_dir': 'dataset_unified',
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.0001,
        'patience': 3
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_dir = Path('loso_validation_results')
    
    # Load existing results
    existing_results = []
    for i in range(1, 8):
        fold_path = output_dir / f'loso_fold_{i}.json'
        if fold_path.exists():
            with open(fold_path) as f:
                existing_results.append(json.load(f))
    
    print(f"Loaded {len(existing_results)} existing fold results")
    
    # Create folds
    folds, metadata_df = create_station_based_folds(config['metadata_path'])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = LOSODataset(metadata_df, config['dataset_dir'], transform=transform)
    config['num_magnitude_classes'] = len(full_dataset.magnitude_classes)
    config['num_azimuth_classes'] = len(full_dataset.azimuth_classes)
    
    print(f"Total folds: {len(folds)}, Starting from fold 8")
    
    # Continue from fold 8
    all_results = existing_results.copy()
    for fold in folds[7:]:  # Start from index 7 (fold 8)
        result = train_fold(fold, full_dataset, config, device)
        all_results.append(result)
        
        with open(output_dir / f'loso_fold_{fold["fold"]}.json', 'w') as f:
            json.dump(result, f, indent=2)
    
    # Aggregate final results
    mag_accs = [r['magnitude_accuracy'] for r in all_results]
    azi_accs = [r['azimuth_accuracy'] for r in all_results]
    
    total_test = sum(r['n_test_samples'] for r in all_results)
    weighted_mag = sum(r['magnitude_accuracy'] * r['n_test_samples'] for r in all_results) / total_test
    weighted_azi = sum(r['azimuth_accuracy'] * r['n_test_samples'] for r in all_results) / total_test
    
    final_results = {
        'n_folds': len(all_results),
        'magnitude_accuracy': {
            'mean': float(np.mean(mag_accs)),
            'std': float(np.std(mag_accs)),
            'min': float(np.min(mag_accs)),
            'max': float(np.max(mag_accs)),
            'weighted_mean': float(weighted_mag)
        },
        'azimuth_accuracy': {
            'mean': float(np.mean(azi_accs)),
            'std': float(np.std(azi_accs)),
            'min': float(np.min(azi_accs)),
            'max': float(np.max(azi_accs)),
            'weighted_mean': float(weighted_azi)
        },
        'per_fold_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Compare with LOEO
    loeo_mag, loeo_azi = 97.53, 69.51
    mag_drop = loeo_mag - final_results['magnitude_accuracy']['mean']
    azi_drop = loeo_azi - final_results['azimuth_accuracy']['mean']
    
    if mag_drop < 10 and azi_drop < 15:
        assessment = "✅ GOOD: Model shows reasonable spatial generalization"
    elif mag_drop < 20 and azi_drop < 25:
        assessment = "⚠️ MODERATE: Some spatial dependency, but still useful"
    else:
        assessment = "❌ CONCERNING: Strong spatial dependency - model may be learning station-specific patterns"
    
    final_results['assessment'] = assessment
    final_results['comparison_with_loeo'] = {
        'loeo_magnitude': loeo_mag,
        'loeo_azimuth': loeo_azi,
        'magnitude_drop': mag_drop,
        'azimuth_drop': azi_drop
    }
    
    with open(output_dir / 'loso_final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("LOSO VALIDATION FINAL RESULTS")
    print("="*70)
    
    print("\nPer-Station Results:")
    print("-" * 60)
    for r in all_results:
        print(f"  {r['test_station']:30s} | Mag: {r['magnitude_accuracy']:5.1f}% | Azi: {r['azimuth_accuracy']:5.1f}%")
    print("-" * 60)
    
    print(f"\nOverall (Simple Mean):")
    print(f"  Magnitude: {final_results['magnitude_accuracy']['mean']:.2f}% ± {final_results['magnitude_accuracy']['std']:.2f}%")
    print(f"  Azimuth:   {final_results['azimuth_accuracy']['mean']:.2f}% ± {final_results['azimuth_accuracy']['std']:.2f}%")
    
    print(f"\nOverall (Weighted by samples):")
    print(f"  Magnitude: {weighted_mag:.2f}%")
    print(f"  Azimuth:   {weighted_azi:.2f}%")
    
    print(f"\nComparison with LOEO:")
    print(f"  Magnitude: LOEO={loeo_mag:.2f}% → LOSO={final_results['magnitude_accuracy']['mean']:.2f}% (drop: {mag_drop:.2f}%)")
    print(f"  Azimuth:   LOEO={loeo_azi:.2f}% → LOSO={final_results['azimuth_accuracy']['mean']:.2f}% (drop: {azi_drop:.2f}%)")
    
    print(f"\n{assessment}")
    print(f"\nResults saved to: {output_dir}/loso_final_results.json")
    
    return final_results


if __name__ == '__main__':
    results = continue_loso_validation()
