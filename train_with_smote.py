"""
Train Hierarchical Model with SMOTE-balanced Dataset
Supports both EfficientNet and ConvNeXt backbones
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import argparse

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
SMOTE_DATASET = 'dataset_smote'


class SMOTEDataset(Dataset):
    def __init__(self, metadata_path, dataset_path, task='magnitude', transform=None):
        self.metadata = pd.read_csv(metadata_path)
        self.dataset_path = dataset_path
        self.task = task
        self.transform = transform
        
        # Class mappings
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
        
        # Get label
        if self.task == 'magnitude':
            label = self.magnitude_map.get(row['magnitude_class'], 1)
        else:
            label = self.azimuth_map.get(row['azimuth_class'], 0)
        
        return image, label


def create_model(num_classes, backbone='efficientnet'):
    """Create model based on backbone"""
    if backbone == 'efficientnet':
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    else:
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
    all_preds = []
    all_labels = []
    
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
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), 100. * correct / total, all_preds, all_labels


def train_task(task_name, num_classes, backbone, output_dir):
    """Train a single task (magnitude or azimuth)"""
    print(f"\n{'='*70}")
    print(f"TRAINING: {task_name.upper()} with {backbone.upper()}")
    print(f"{'='*70}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = SMOTEDataset(
        os.path.join(SMOTE_DATASET, 'metadata', 'smote_train.csv'),
        SMOTE_DATASET, task_name, train_transform
    )
    val_dataset = SMOTEDataset(
        os.path.join(SMOTE_DATASET, 'metadata', 'smote_val.csv'),
        SMOTE_DATASET, task_name, val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Check class distribution
    if task_name == 'magnitude':
        dist = Counter(train_dataset.metadata['magnitude_class'])
    else:
        dist = Counter(train_dataset.metadata['azimuth_class'])
    print(f"Class distribution: {dict(dist)}")
    
    # Model
    model = create_model(num_classes, backbone).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    best_acc = 0
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            model_path = os.path.join(output_dir, f'{backbone}_{task_name}_smote_best.pth')
            torch.save(model.state_dict(), model_path)
            print(f"[OK] New best model! Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"\n[OK] {task_name} Complete: Best Acc = {best_acc:.2f}% (epoch {best_epoch})")
    return best_acc, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='efficientnet', 
                        choices=['efficientnet', 'convnext', 'both'])
    parser.add_argument('--task', type=str, default='both',
                        choices=['magnitude', 'azimuth', 'both'])
    args = parser.parse_args()
    
    print("=" * 70)
    print("SMOTE-BALANCED HIERARCHICAL TRAINING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Backbone: {args.backbone}")
    print(f"Task: {args.task}")
    
    # Check SMOTE dataset exists
    if not os.path.exists(os.path.join(SMOTE_DATASET, 'metadata', 'smote_train.csv')):
        print("\n[ERROR] SMOTE dataset not found!")
        print("Please run: python generate_smote_dataset.py")
        sys.exit(1)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'experiments_smote/smote_training_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    backbones = ['efficientnet', 'convnext'] if args.backbone == 'both' else [args.backbone]
    tasks = ['magnitude', 'azimuth'] if args.task == 'both' else [args.task]
    
    for backbone in backbones:
        results[backbone] = {}
        for task in tasks:
            num_classes = 3 if task == 'magnitude' else 8
            acc, epoch = train_task(task, num_classes, backbone, output_dir)
            results[backbone][task] = {'accuracy': acc, 'best_epoch': epoch}
    
    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    for backbone, tasks in results.items():
        print(f"\n{backbone.upper()}:")
        for task, res in tasks.items():
            print(f"  {task}: {res['accuracy']:.2f}% (epoch {res['best_epoch']})")
    
    print(f"\nModels saved to: {output_dir}")


if __name__ == '__main__':
    main()
