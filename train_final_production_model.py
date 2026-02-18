#!/usr/bin/env python3
"""
Train Final Production Model with ALL Data
No train/test split - uses all available data for maximum performance

Author: Earthquake Prediction Research Team
Date: 4 February 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import shutil
import warnings
warnings.filterwarnings('ignore')


class EfficientNetMultiTask(nn.Module):
    """EfficientNet-B0 based multi-task model for production"""
    
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


class FullDataset(Dataset):
    """Dataset using ALL available data"""
    
    def __init__(self, metadata_df, dataset_dir, transform=None, image_size=224):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.image_size = image_size
        
        self.magnitude_classes = sorted(self.metadata['magnitude_class'].dropna().unique())
        self.azimuth_classes = sorted(self.metadata['azimuth_class'].dropna().unique())
        
        self.magnitude_to_idx = {cls: idx for idx, cls in enumerate(self.magnitude_classes)}
        self.azimuth_to_idx = {cls: idx for idx, cls in enumerate(self.azimuth_classes)}
        
        self.idx_to_magnitude = {idx: cls for cls, idx in self.magnitude_to_idx.items()}
        self.idx_to_azimuth = {idx: cls for cls, idx in self.azimuth_to_idx.items()}
        
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


def train_one_epoch(model, train_loader, criterion_mag, criterion_azi, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct_mag = 0
    correct_azi = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for images, mag_labels, azi_labels in pbar:
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
        
        pbar.set_postfix({
            'loss': f'{total_loss/len(train_loader):.4f}',
            'mag': f'{100*correct_mag/total:.1f}%',
            'azi': f'{100*correct_azi/total:.1f}%'
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'mag_acc': 100 * correct_mag / total,
        'azi_acc': 100 * correct_azi / total
    }


def train_final_model(config):
    """Train final production model with ALL data"""
    print("\n" + "="*60)
    print("TRAINING FINAL PRODUCTION MODEL")
    print("Using ALL data (no train/test split)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path('final_production_model')
    output_dir.mkdir(exist_ok=True)
    
    # Load metadata
    print("\nLoading dataset...")
    df = pd.read_csv(config['metadata_path'])
    df = df.dropna(subset=['magnitude_class', 'azimuth_class'])
    df['magnitude_class'] = df['magnitude_class'].astype(str)
    df['azimuth_class'] = df['azimuth_class'].astype(str)
    df = df[df['magnitude_class'] != 'nan']
    
    print(f"Total samples: {len(df)}")
    print(f"Magnitude distribution: {df['magnitude_class'].value_counts().to_dict()}")
    print(f"Azimuth distribution: {df['azimuth_class'].value_counts().to_dict()}")
    
    # Create transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    full_dataset = FullDataset(df, config['dataset_dir'], transform=train_transform)
    
    num_magnitude_classes = len(full_dataset.magnitude_classes)
    num_azimuth_classes = len(full_dataset.azimuth_classes)
    
    print(f"\nMagnitude classes ({num_magnitude_classes}): {full_dataset.magnitude_classes}")
    print(f"Azimuth classes ({num_azimuth_classes}): {full_dataset.azimuth_classes}")
    
    # Create data loader
    train_loader = DataLoader(
        full_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("\nInitializing model...")
    model = EfficientNetMultiTask(
        num_magnitude_classes=num_magnitude_classes,
        num_azimuth_classes=num_azimuth_classes,
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Calculate class weights for imbalanced data
    mag_counts = df['magnitude_class'].value_counts()
    azi_counts = df['azimuth_class'].value_counts()
    
    mag_weights = torch.tensor([1.0 / mag_counts.get(cls, 1) for cls in full_dataset.magnitude_classes], dtype=torch.float32)
    azi_weights = torch.tensor([1.0 / azi_counts.get(cls, 1) for cls in full_dataset.azimuth_classes], dtype=torch.float32)
    
    mag_weights = mag_weights / mag_weights.sum() * len(mag_weights)
    azi_weights = azi_weights / azi_weights.sum() * len(azi_weights)
    
    criterion_mag = nn.CrossEntropyLoss(weight=mag_weights.to(device))
    criterion_azi = nn.CrossEntropyLoss(weight=azi_weights.to(device))
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    best_loss = float('inf')
    training_history = []
    
    for epoch in range(config['epochs']):
        results = train_one_epoch(model, train_loader, criterion_mag, criterion_azi, optimizer, device, epoch)
        
        training_history.append({
            'epoch': epoch + 1,
            'loss': results['loss'],
            'mag_acc': results['mag_acc'],
            'azi_acc': results['azi_acc'],
            'lr': optimizer.param_groups[0]['lr']
        })
        
        scheduler.step(results['loss'])
        
        print(f"Epoch {epoch+1}/{config['epochs']}: Loss={results['loss']:.4f}, "
              f"Mag={results['mag_acc']:.2f}%, Azi={results['azi_acc']:.2f}%")
        
        # Save best model
        if results['loss'] < best_loss:
            best_loss = results['loss']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'mag_acc': results['mag_acc'],
                'azi_acc': results['azi_acc'],
                'magnitude_classes': full_dataset.magnitude_classes,
                'azimuth_classes': full_dataset.azimuth_classes,
                'magnitude_to_idx': full_dataset.magnitude_to_idx,
                'azimuth_to_idx': full_dataset.azimuth_to_idx,
                'idx_to_magnitude': full_dataset.idx_to_magnitude,
                'idx_to_azimuth': full_dataset.idx_to_azimuth,
            }, output_dir / 'best_final_model.pth')
            print(f"  ✓ Saved best model (loss: {best_loss:.4f})")
    
    # Save final model
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': results['loss'],
        'mag_acc': results['mag_acc'],
        'azi_acc': results['azi_acc'],
        'magnitude_classes': full_dataset.magnitude_classes,
        'azimuth_classes': full_dataset.azimuth_classes,
        'magnitude_to_idx': full_dataset.magnitude_to_idx,
        'azimuth_to_idx': full_dataset.azimuth_to_idx,
        'idx_to_magnitude': full_dataset.idx_to_magnitude,
        'idx_to_azimuth': full_dataset.idx_to_azimuth,
    }, output_dir / 'final_model.pth')
    
    # Save training history
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    
    # Save class mappings
    class_mappings = {
        'magnitude_classes': full_dataset.magnitude_classes,
        'azimuth_classes': full_dataset.azimuth_classes,
        'magnitude_to_idx': full_dataset.magnitude_to_idx,
        'azimuth_to_idx': full_dataset.azimuth_to_idx,
        'idx_to_magnitude': {str(k): v for k, v in full_dataset.idx_to_magnitude.items()},
        'idx_to_azimuth': {str(k): v for k, v in full_dataset.idx_to_azimuth.items()}
    }
    with open(output_dir / 'class_mappings.json', 'w') as f:
        json.dump(class_mappings, f, indent=2)
    
    # Save training summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'total_samples': len(df),
        'epochs_trained': config['epochs'],
        'final_loss': float(results['loss']),
        'final_mag_acc': float(results['mag_acc']),
        'final_azi_acc': float(results['azi_acc']),
        'best_loss': float(best_loss),
        'num_magnitude_classes': num_magnitude_classes,
        'num_azimuth_classes': num_azimuth_classes,
        'validation_results': {
            'loeo_mag_acc': 97.53,
            'loeo_azi_acc': 69.51,
            'loso_mag_acc': 97.57,
            'loso_azi_acc': 69.73
        },
        'config': config
    }
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Magnitude Accuracy: {results['mag_acc']:.2f}%")
    print(f"  Azimuth Accuracy: {results['azi_acc']:.2f}%")
    print(f"\nModel saved to: {output_dir}")
    
    return summary


def deploy_to_production(source_dir='final_production_model', production_dir='production/models'):
    """Deploy the trained model to production"""
    print("\n" + "="*60)
    print("DEPLOYING TO PRODUCTION")
    print("="*60)
    
    source_path = Path(source_dir)
    prod_path = Path(production_dir)
    prod_path.mkdir(parents=True, exist_ok=True)
    
    # Backup existing model
    existing_model = prod_path / 'earthquake_model.pth'
    if existing_model.exists():
        backup_path = prod_path / f'earthquake_model_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        shutil.copy(existing_model, backup_path)
        print(f"✓ Backed up existing model to: {backup_path}")
    
    # Copy new model
    new_model = source_path / 'best_final_model.pth'
    if new_model.exists():
        shutil.copy(new_model, existing_model)
        print(f"✓ Deployed new model to: {existing_model}")
    
    # Copy class mappings
    class_mappings = source_path / 'class_mappings.json'
    if class_mappings.exists():
        shutil.copy(class_mappings, prod_path / 'class_mappings.json')
        print(f"✓ Copied class mappings")
    
    # Copy training summary
    summary = source_path / 'training_summary.json'
    if summary.exists():
        shutil.copy(summary, prod_path / 'training_summary.json')
        print(f"✓ Copied training summary")
    
    print("\n✅ Deployment complete!")
    return True


if __name__ == '__main__':
    config = {
        'metadata_path': 'dataset_unified/metadata/unified_metadata.csv',
        'dataset_dir': 'dataset_unified',
        'batch_size': 32,
        'epochs': 20,  # More epochs for final model
        'learning_rate': 0.0001,
        'dropout_rate': 0.3
    }
    
    # Train final model
    summary = train_final_model(config)
    
    # Ask for deployment
    print("\n" + "-"*60)
    deploy = input("Deploy to production? (y/n): ").strip().lower()
    if deploy == 'y':
        deploy_to_production()
    else:
        print("Skipping deployment. You can deploy later with:")
        print("  python -c \"from train_final_production_model import deploy_to_production; deploy_to_production()\"")
