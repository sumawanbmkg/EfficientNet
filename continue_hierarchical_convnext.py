"""
Continue Hierarchical ConvNeXt Training
Stage 1 sudah selesai (100%), lanjut Stage 2 dan 3
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import timm
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter
from datetime import datetime

# Configuration
BACKBONE = 'convnext'
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
DATASET_PATH = 'dataset_unified'
METADATA_FILE = os.path.join(DATASET_PATH, 'metadata', 'unified_metadata.csv')
STAGE1_MODEL_PATH = 'experiments_hierarchical/convnext_hierarchical_20260211_093622/stage1_binary_best.pth'

# Output directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f'experiments_hierarchical/convnext_hierarchical_continue_{timestamp}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CONTINUE CONVNEXT HIERARCHICAL TRAINING")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT_DIR}")
print(f"Stage 1 model: {STAGE1_MODEL_PATH}")


class HierarchicalDataset(Dataset):
    def __init__(self, metadata, dataset_path, stage, transform=None):
        self.metadata = metadata.reset_index(drop=True)
        self.dataset_path = dataset_path
        self.stage = stage
        self.transform = transform
        
        # Class mappings
        self.binary_map = {'Normal': 0, 'Precursor': 1}
        self.magnitude_map = {'Moderate': 0, 'Medium': 1, 'Large': 2}
        self.azimuth_map = {'E': 0, 'N': 1, 'NE': 2, 'NW': 3, 'S': 4, 'SE': 5, 'SW': 6, 'W': 7}
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.dataset_path, row['unified_path'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get label based on stage
        if self.stage == 'binary':
            label = self.binary_map.get(row['label'], 1)
        elif self.stage == 'magnitude':
            label = self.magnitude_map.get(row['magnitude_class'], 1)
        elif self.stage == 'azimuth':
            label = self.azimuth_map.get(row['azimuth_class'], 0)
        else:
            label = 0
            
        return image, label


def create_model(num_classes, backbone='convnext'):
    """Create ConvNeXt model"""
    model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100.*correct/total:.2f}%')
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Evaluating')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def train_stage(stage_name, num_classes, train_data, val_data, class_weights=None):
    """Train a single stage"""
    print(f"\n{'='*70}")
    print(f"STAGE: {stage_name.upper()} ({num_classes} classes)")
    print(f"{'='*70}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = HierarchicalDataset(train_data, DATASET_PATH, stage_name.split('_')[0], train_transform)
    val_dataset = HierarchicalDataset(val_data, DATASET_PATH, stage_name.split('_')[0], val_transform)
    
    # Weighted sampler for imbalanced classes
    if class_weights is not None:
        if stage_name == 'magnitude':
            labels = [train_dataset.magnitude_map.get(row['magnitude_class'], 1) for _, row in train_data.iterrows()]
        else:
            labels = [train_dataset.azimuth_map.get(row['azimuth_class'], 0) for _, row in train_data.iterrows()]
        
        sample_weights = [class_weights[l] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model
    model = create_model(num_classes, BACKBONE).to(DEVICE)
    
    # Loss with class weights
    if class_weights is not None:
        weight_tensor = torch.tensor([class_weights.get(i, 1.0) for i in range(num_classes)], dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    best_acc = 0
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{stage_name} - Epoch {epoch}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'{stage_name}_best.pth'))
            print(f"[OK] New best model saved! Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"\n[OK] {stage_name} Complete: Best Acc = {best_acc:.2f}% (epoch {best_epoch})")
    return best_acc, best_epoch


def main():
    # Load metadata from split files
    print("\nLoading metadata...")
    train_data = pd.read_csv(os.path.join(DATASET_PATH, 'metadata', 'train_split.csv'))
    val_data = pd.read_csv(os.path.join(DATASET_PATH, 'metadata', 'val_split.csv'))
    
    print(f"Train samples (all): {len(train_data)}")
    print(f"Val samples (all): {len(val_data)}")
    
    # Filter only Precursor samples (exclude Normal)
    # Precursor samples have magnitude_class in ['Moderate', 'Medium', 'Large']
    valid_mag_classes = ['Moderate', 'Medium', 'Large']
    train_precursor = train_data[train_data['magnitude_class'].isin(valid_mag_classes)].copy()
    val_precursor = val_data[val_data['magnitude_class'].isin(valid_mag_classes)].copy()
    
    print(f"\nPrecursor train: {len(train_precursor)}")
    print(f"Precursor val: {len(val_precursor)}")
    
    results = {}
    
    # Copy Stage 1 model
    import shutil
    if os.path.exists(STAGE1_MODEL_PATH):
        shutil.copy(STAGE1_MODEL_PATH, os.path.join(OUTPUT_DIR, 'stage1_binary_best.pth'))
        print(f"\n[OK] Stage 1 model copied from previous training")
        results['stage1_binary'] = {'accuracy': 100.0, 'best_epoch': 1}
    
    # ==================== STAGE 2: Magnitude ====================
    print("\n" + "="*70)
    print("STAGE 2: Magnitude Classification (Moderate/Medium/Large)")
    print("="*70)
    
    # Check class distribution
    mag_dist = Counter(train_precursor['magnitude_class'])
    print(f"Class distribution: {mag_dist}")
    
    # Calculate class weights (inverse frequency with boost for minority)
    total = sum(mag_dist.values())
    mag_weights = {}
    for cls, count in mag_dist.items():
        cls_idx = {'Moderate': 0, 'Medium': 1, 'Large': 2}[cls]
        # Boost minority classes significantly
        if count < 50:
            mag_weights[cls_idx] = (total / count) * 5  # Extra boost
        else:
            mag_weights[cls_idx] = total / (count * len(mag_dist))
    
    print(f"Class weights: {mag_weights}")
    
    mag_acc, mag_epoch = train_stage(
        'magnitude', 3, train_precursor, val_precursor, mag_weights
    )
    results['stage2_magnitude'] = {'accuracy': mag_acc, 'best_epoch': mag_epoch}
    
    # ==================== STAGE 3: Azimuth ====================
    print("\n" + "="*70)
    print("STAGE 3: Azimuth Classification (8 directions)")
    print("="*70)
    
    # Check class distribution
    azi_dist = Counter(train_precursor['azimuth_class'])
    print(f"Class distribution: {azi_dist}")
    
    # Calculate class weights
    azi_weights = {}
    azi_map = {'E': 0, 'N': 1, 'NE': 2, 'NW': 3, 'S': 4, 'SE': 5, 'SW': 6, 'W': 7}
    for cls, count in azi_dist.items():
        cls_idx = azi_map[cls]
        if count < 50:
            azi_weights[cls_idx] = (total / count) * 3
        else:
            azi_weights[cls_idx] = total / (count * len(azi_dist))
    
    print(f"Class weights: {azi_weights}")
    
    azi_acc, azi_epoch = train_stage(
        'azimuth', 8, train_precursor, val_precursor, azi_weights
    )
    results['stage3_azimuth'] = {'accuracy': azi_acc, 'best_epoch': azi_epoch}
    
    # Save results
    results['backbone'] = BACKBONE
    with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save class mappings
    class_mappings = {
        'binary': {'Normal': 0, 'Precursor': 1},
        'magnitude': {'Moderate': 0, 'Medium': 1, 'Large': 2},
        'azimuth': {'E': 0, 'N': 1, 'NE': 2, 'NW': 3, 'S': 4, 'SE': 5, 'SW': 6, 'W': 7}
    }
    with open(os.path.join(OUTPUT_DIR, 'class_mappings.json'), 'w') as f:
        json.dump(class_mappings, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - CONVNEXT HIERARCHICAL")
    print("="*70)
    print(f"Stage 1 (Binary):    100.00% (from previous)")
    print(f"Stage 2 (Magnitude): {mag_acc:.2f}% (epoch {mag_epoch})")
    print(f"Stage 3 (Azimuth):   {azi_acc:.2f}% (epoch {azi_epoch})")
    print(f"\nModels saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
