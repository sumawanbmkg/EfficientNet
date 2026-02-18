#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'metadata_path': 'dataset_unified/metadata/unified_metadata.csv',
    'dataset_dir': 'dataset_unified',
    'output_dir': 'convnext_production_model',
    'batch_size': 16,
    'epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 0.05,
    'patience': 7,
    'dropout': 0.5,
    'image_size': 224,
    'val_split': 0.2,
    'random_seed': 42,
    'use_augmentation': True,
    'use_weighted_sampling': True,
    'use_focal_loss': True,
    'focal_gamma': 2.0,
}

def get_train_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),
    ])

def get_val_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class EarthquakeDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, mag_classes=None, azi_classes=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.mag_classes = mag_classes or sorted(self.df['magnitude_class'].unique())
        self.azi_classes = azi_classes or sorted(self.df['azimuth_class'].unique())
        self.mag_to_idx = {c: i for i, c in enumerate(self.mag_classes)}
        self.azi_to_idx = {c: i for i, c in enumerate(self.azi_classes)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / 'spectrograms' / row['spectrogram_file']
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            image = self.transform(image)
        mag_label = self.mag_to_idx.get(row['magnitude_class'], 0)
        azi_label = self.azi_to_idx.get(row['azimuth_class'], 0)
        return image, mag_label, azi_label

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()

class ConvNeXtMultiTask(nn.Module):
    def __init__(self, num_mag, num_azi, dropout=0.5):
        super().__init__()
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_f = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()
        self.shared = nn.Sequential(nn.LayerNorm(in_f), nn.Dropout(dropout), nn.Linear(in_f, 512), nn.GELU(), nn.Dropout(dropout/2))
        self.mag_head = nn.Linear(512, num_mag)
        self.azi_head = nn.Linear(512, num_azi)
    
    def forward(self, x):
        f = self.backbone(x)
        s = self.shared(f)
        return self.mag_head(s), self.azi_head(s)

def train_epoch(model, loader, optimizer, mag_crit, azi_crit, device):
    model.train()
    total_loss, mag_c, azi_c, total = 0, 0, 0, 0
    for images, mag_labels, azi_labels in tqdm(loader, desc='Training'):
        images, mag_labels, azi_labels = images.to(device), mag_labels.to(device), azi_labels.to(device)
        optimizer.zero_grad()
        mag_out, azi_out = model(images)
        loss = mag_crit(mag_out, mag_labels) + azi_crit(azi_out, azi_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        mag_c += (mag_out.argmax(1) == mag_labels).sum().item()
        azi_c += (azi_out.argmax(1) == azi_labels).sum().item()
        total += len(images)
    return total_loss/len(loader), mag_c/total, azi_c/total

def evaluate(model, loader, mag_crit, azi_crit, device):
    model.eval()
    total_loss, mag_c, azi_c, total = 0, 0, 0, 0
    with torch.no_grad():
        for images, mag_labels, azi_labels in tqdm(loader, desc='Evaluating'):
            images, mag_labels, azi_labels = images.to(device), mag_labels.to(device), azi_labels.to(device)
            mag_out, azi_out = model(images)
            loss = mag_crit(mag_out, mag_labels) + azi_crit(azi_out, azi_labels)
            total_loss += loss.item()
            mag_c += (mag_out.argmax(1) == mag_labels).sum().item()
            azi_c += (azi_out.argmax(1) == azi_labels).sum().item()
            total += len(images)
    return total_loss/len(loader), mag_c/total, azi_c/total

def main():
    print("="*70 + "\nCONVNEXT PRODUCTION TRAINING\n" + "="*70)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    df = pd.read_csv(CONFIG['metadata_path']).dropna(subset=['magnitude_class', 'azimuth_class'])
    print(f"Samples: {len(df)}")
    
    mag_cls = sorted(df['magnitude_class'].unique())
    azi_cls = sorted(df['azimuth_class'].unique())
    print(f"Mag classes: {mag_cls}\nAzi classes: {azi_cls}")
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['magnitude_class'], random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    train_ds = EarthquakeDataset(train_df, CONFIG['dataset_dir'], get_train_transforms(), mag_cls, azi_cls)
    val_ds = EarthquakeDataset(val_df, CONFIG['dataset_dir'], get_val_transforms(), mag_cls, azi_cls)
    
    mag_counts = train_df['magnitude_class'].value_counts()
    weights = train_df['magnitude_class'].map(lambda x: 1.0/mag_counts[x])
    sampler = WeightedRandomSampler(weights.values, len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    print("Creating ConvNeXt model...")
    model = ConvNeXtMultiTask(len(mag_cls), len(azi_cls)).to(device)
    mag_crit = FocalLoss(gamma=2.0)
    azi_crit = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    best_combined, patience = 0, 0
    for epoch in range(30):
        print(f"\nEpoch {epoch+1}/30")
        tr_loss, tr_mag, tr_azi = train_epoch(model, train_loader, optimizer, mag_crit, azi_crit, device)
        val_loss, val_mag, val_azi = evaluate(model, val_loader, mag_crit, azi_crit, device)
        scheduler.step()
        print(f"Train - Loss:{tr_loss:.4f} Mag:{100*tr_mag:.1f}% Azi:{100*tr_azi:.1f}%")
        print(f"Val   - Loss:{val_loss:.4f} Mag:{100*val_mag:.1f}% Azi:{100*val_azi:.1f}%")
        
        combined = (val_mag + val_azi) / 2
        if combined > best_combined:
            best_combined = combined
            patience = 0
            torch.save({'model_state_dict': model.state_dict(), 'mag_classes': mag_cls, 'azi_classes': azi_cls, 'val_mag': val_mag, 'val_azi': val_azi}, output_dir/'best_convnext_model.pth')
            print(f"Saved best model! Combined: {100*combined:.1f}%")
        else:
            patience += 1
            if patience >= 7:
                print("Early stopping!")
                break
    
    torch.save({'model_state_dict': model.state_dict(), 'mag_classes': mag_cls, 'azi_classes': azi_cls}, output_dir/'final_model.pth')
    print(f"\nDone! Best combined: {100*best_combined:.1f}%")

if __name__ == '__main__':
    main()
