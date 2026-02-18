"""
Hierarchical Classification Model Training
==========================================
Stage 1: Binary Classification (Normal vs Precursor)
Stage 2: Magnitude Classification (Moderate/Medium/Large) - only for Precursor
Stage 3: Azimuth Classification (8 directions) - only for Precursor

Supports: EfficientNet-B0 and ConvNeXt-Tiny
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    metadata_path = 'dataset_unified/metadata/unified_metadata.csv'
    dataset_dir = 'dataset_unified'
    
    # Model
    backbone = 'efficientnet'  # 'efficientnet' or 'convnext'
    img_size = 224
    
    # Training
    batch_size = 32
    epochs_stage1 = 30  # Binary classification
    epochs_stage2 = 30  # Magnitude classification
    epochs_stage3 = 30  # Azimuth classification
    learning_rate = 1e-4
    weight_decay = 0.01
    patience = 10
    
    # Output
    output_dir = 'experiments_hierarchical'

# ============================================================================
# DATASET
# ============================================================================

class HierarchicalDataset(Dataset):
    """Dataset for hierarchical classification"""
    
    def __init__(self, metadata_df, dataset_dir, transform=None, stage='stage1'):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.stage = stage
        
        # Stage 1: Binary (Normal=0, Precursor=1)
        # Stage 2: Magnitude (Moderate=0, Medium=1, Large=2)
        # Stage 3: Azimuth (E=0, N=1, NE=2, NW=3, S=4, SE=5, SW=6, W=7)
        
        self.binary_map = {'Normal': 0, 'Moderate': 1, 'Medium': 1, 'Large': 1}
        self.magnitude_map = {'Moderate': 0, 'Medium': 1, 'Large': 2}
        self.azimuth_map = {'E': 0, 'N': 1, 'NE': 2, 'NW': 3, 'S': 4, 'SE': 5, 'SW': 6, 'W': 7}
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load image (try unified_path first, then spectrogram_path)
        rel_path = row.get('unified_path') or row.get('spectrogram_path') or row.get('relative_path')
        img_path = os.path.join(self.dataset_dir, rel_path)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels based on stage
        mag_class = row['magnitude_class']
        azi_class = row.get('azimuth_class', 'Normal')
        
        binary_label = self.binary_map.get(mag_class, 0)
        
        if self.stage == 'stage1':
            return image, binary_label
        elif self.stage == 'stage2':
            mag_label = self.magnitude_map.get(mag_class, 1)
            return image, mag_label
        else:  # stage3
            azi_label = self.azimuth_map.get(azi_class, 0)
            return image, azi_label

# ============================================================================
# MODELS
# ============================================================================

class HierarchicalEfficientNet(nn.Module):
    """EfficientNet-B0 for hierarchical classification"""
    
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class HierarchicalConvNeXt(nn.Module):
    """ConvNeXt-Tiny for hierarchical classification"""
    
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=True)
        num_features = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def create_model(backbone, num_classes, dropout=0.3):
    """Create model based on backbone type"""
    if backbone == 'efficientnet':
        return HierarchicalEfficientNet(num_classes, dropout)
    else:
        return HierarchicalConvNeXt(num_classes, dropout)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def get_class_weights(labels):
    """Calculate class weights for imbalanced data"""
    counter = Counter(labels)
    total = len(labels)
    weights = {cls: total / count for cls, count in counter.items()}
    max_weight = max(weights.values())
    weights = {cls: w / max_weight for cls, w in weights.items()}
    return weights

def get_sample_weights(labels, class_weights):
    """Get sample weights for WeightedRandomSampler"""
    return [class_weights[label] for label in labels]

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return total_loss / len(dataloader), 100. * correct / total, all_preds, all_labels, all_probs

def train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, epochs, patience, stage_name, save_dir):
    """Train a single stage with early stopping"""
    
    best_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    for epoch in range(epochs):
        print(f"\n{stage_name} - Epoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f'{stage_name}_best.pth'))
            print(f"[OK] New best model saved! Acc: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, f'{stage_name}_best.pth')))
    
    return model, best_acc, best_epoch, history

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main(backbone='efficientnet'):
    """Main training function"""
    
    print("=" * 70)
    print(f"HIERARCHICAL CLASSIFICATION TRAINING - {backbone.upper()}")
    print("=" * 70)
    
    config = Config()
    config.backbone = backbone
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(config.output_dir, f'{backbone}_hierarchical_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output: {save_dir}")
    
    # Load metadata
    print("\nLoading metadata...")
    metadata = pd.read_csv(config.metadata_path)
    print(f"Total samples: {len(metadata)}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Split data by event (use 'date' column)
    events = metadata.groupby(['station', 'date']).first().reset_index()
    train_events, val_events = train_test_split(
        events[['station', 'date']], 
        test_size=0.2, 
        random_state=42
    )
    
    train_keys = set(zip(train_events['station'], train_events['date']))
    val_keys = set(zip(val_events['station'], val_events['date']))
    
    train_mask = metadata.apply(lambda x: (x['station'], x['date']) in train_keys, axis=1)
    val_mask = metadata.apply(lambda x: (x['station'], x['date']) in val_keys, axis=1)
    
    train_df = metadata[train_mask].reset_index(drop=True)
    val_df = metadata[val_mask].reset_index(drop=True)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    results = {}
    
    # ========================================================================
    # STAGE 1: Binary Classification (Normal vs Precursor)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: Binary Classification (Normal vs Precursor)")
    print("=" * 70)
    
    # Create datasets
    train_dataset_s1 = HierarchicalDataset(train_df, config.dataset_dir, train_transform, 'stage1')
    val_dataset_s1 = HierarchicalDataset(val_df, config.dataset_dir, val_transform, 'stage1')
    
    # Get labels for weighted sampling
    train_labels_s1 = [train_dataset_s1.binary_map.get(row['magnitude_class'], 0) 
                       for _, row in train_df.iterrows()]
    
    class_weights_s1 = get_class_weights(train_labels_s1)
    sample_weights_s1 = get_sample_weights(train_labels_s1, class_weights_s1)
    sampler_s1 = WeightedRandomSampler(sample_weights_s1, len(sample_weights_s1))
    
    print(f"Class distribution: {Counter(train_labels_s1)}")
    print(f"Class weights: {class_weights_s1}")
    
    train_loader_s1 = DataLoader(train_dataset_s1, batch_size=config.batch_size, sampler=sampler_s1)
    val_loader_s1 = DataLoader(val_dataset_s1, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    model_s1 = create_model(backbone, num_classes=2).to(device)
    criterion_s1 = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device))
    optimizer_s1 = optim.AdamW(model_s1.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler_s1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s1, patience=3, factor=0.5)
    
    # Train
    model_s1, best_acc_s1, best_epoch_s1, history_s1 = train_stage(
        model_s1, train_loader_s1, val_loader_s1, criterion_s1, optimizer_s1, scheduler_s1,
        device, config.epochs_stage1, config.patience, 'stage1_binary', save_dir
    )
    
    results['stage1'] = {
        'task': 'Binary (Normal vs Precursor)',
        'best_acc': best_acc_s1,
        'best_epoch': best_epoch_s1,
        'history': history_s1
    }
    
    print(f"\n[OK] Stage 1 Complete: Best Acc = {best_acc_s1:.2f}%")
    
    # ========================================================================
    # STAGE 2: Magnitude Classification (only Precursor samples)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: Magnitude Classification (Moderate/Medium/Large)")
    print("=" * 70)
    
    # Filter only precursor samples
    train_precursor = train_df[train_df['magnitude_class'] != 'Normal'].reset_index(drop=True)
    val_precursor = val_df[val_df['magnitude_class'] != 'Normal'].reset_index(drop=True)
    
    print(f"Train precursor samples: {len(train_precursor)}")
    print(f"Val precursor samples: {len(val_precursor)}")
    
    train_dataset_s2 = HierarchicalDataset(train_precursor, config.dataset_dir, train_transform, 'stage2')
    val_dataset_s2 = HierarchicalDataset(val_precursor, config.dataset_dir, val_transform, 'stage2')
    
    # Get labels for weighted sampling
    train_labels_s2 = [train_dataset_s2.magnitude_map.get(row['magnitude_class'], 1) 
                       for _, row in train_precursor.iterrows()]
    
    class_weights_s2 = get_class_weights(train_labels_s2)
    # Boost Large class weight
    if 2 in class_weights_s2:
        class_weights_s2[2] *= 10  # Extra boost for Large
    sample_weights_s2 = get_sample_weights(train_labels_s2, class_weights_s2)
    sampler_s2 = WeightedRandomSampler(sample_weights_s2, len(sample_weights_s2))
    
    print(f"Class distribution: {Counter(train_labels_s2)}")
    print(f"Class weights: {class_weights_s2}")
    
    train_loader_s2 = DataLoader(train_dataset_s2, batch_size=config.batch_size, sampler=sampler_s2)
    val_loader_s2 = DataLoader(val_dataset_s2, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    model_s2 = create_model(backbone, num_classes=3).to(device)
    
    # Class weights for loss
    weight_tensor_s2 = torch.tensor([
        class_weights_s2.get(0, 1.0),
        class_weights_s2.get(1, 1.0),
        class_weights_s2.get(2, 1.0) * 5  # Extra weight for Large
    ]).to(device)
    
    criterion_s2 = nn.CrossEntropyLoss(weight=weight_tensor_s2)
    optimizer_s2 = optim.AdamW(model_s2.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler_s2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s2, patience=3, factor=0.5)
    
    # Train
    model_s2, best_acc_s2, best_epoch_s2, history_s2 = train_stage(
        model_s2, train_loader_s2, val_loader_s2, criterion_s2, optimizer_s2, scheduler_s2,
        device, config.epochs_stage2, config.patience, 'stage2_magnitude', save_dir
    )
    
    results['stage2'] = {
        'task': 'Magnitude (Moderate/Medium/Large)',
        'best_acc': best_acc_s2,
        'best_epoch': best_epoch_s2,
        'history': history_s2
    }
    
    print(f"\n[OK] Stage 2 Complete: Best Acc = {best_acc_s2:.2f}%")
    
    # ========================================================================
    # STAGE 3: Azimuth Classification (only Precursor samples)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: Azimuth Classification (8 directions)")
    print("=" * 70)
    
    train_dataset_s3 = HierarchicalDataset(train_precursor, config.dataset_dir, train_transform, 'stage3')
    val_dataset_s3 = HierarchicalDataset(val_precursor, config.dataset_dir, val_transform, 'stage3')
    
    # Get labels for weighted sampling
    train_labels_s3 = [train_dataset_s3.azimuth_map.get(row.get('azimuth_class', 'N'), 1) 
                       for _, row in train_precursor.iterrows()]
    
    class_weights_s3 = get_class_weights(train_labels_s3)
    sample_weights_s3 = get_sample_weights(train_labels_s3, class_weights_s3)
    sampler_s3 = WeightedRandomSampler(sample_weights_s3, len(sample_weights_s3))
    
    print(f"Class distribution: {Counter(train_labels_s3)}")
    
    train_loader_s3 = DataLoader(train_dataset_s3, batch_size=config.batch_size, sampler=sampler_s3)
    val_loader_s3 = DataLoader(val_dataset_s3, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    model_s3 = create_model(backbone, num_classes=8).to(device)
    criterion_s3 = nn.CrossEntropyLoss()
    optimizer_s3 = optim.AdamW(model_s3.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler_s3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s3, patience=3, factor=0.5)
    
    # Train
    model_s3, best_acc_s3, best_epoch_s3, history_s3 = train_stage(
        model_s3, train_loader_s3, val_loader_s3, criterion_s3, optimizer_s3, scheduler_s3,
        device, config.epochs_stage3, config.patience, 'stage3_azimuth', save_dir
    )
    
    results['stage3'] = {
        'task': 'Azimuth (8 directions)',
        'best_acc': best_acc_s3,
        'best_epoch': best_epoch_s3,
        'history': history_s3
    }
    
    print(f"\n[OK] Stage 3 Complete: Best Acc = {best_acc_s3:.2f}%")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save config
    config_dict = {
        'backbone': backbone,
        'img_size': config.img_size,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'timestamp': timestamp
    }
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save class mappings
    mappings = {
        'binary': {'Normal': 0, 'Precursor': 1},
        'magnitude': {'Moderate': 0, 'Medium': 1, 'Large': 2},
        'azimuth': {'E': 0, 'N': 1, 'NE': 2, 'NW': 3, 'S': 4, 'SE': 5, 'SW': 6, 'W': 7}
    }
    
    with open(os.path.join(save_dir, 'class_mappings.json'), 'w') as f:
        json.dump(mappings, f, indent=2)
    
    # Save results summary
    summary = {
        'backbone': backbone,
        'stage1_binary': {'accuracy': best_acc_s1, 'best_epoch': best_epoch_s1},
        'stage2_magnitude': {'accuracy': best_acc_s2, 'best_epoch': best_epoch_s2},
        'stage3_azimuth': {'accuracy': best_acc_s3, 'best_epoch': best_epoch_s3}
    }
    
    with open(os.path.join(save_dir, 'results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"Backbone: {backbone.upper()}")
    print(f"Stage 1 (Binary):    {best_acc_s1:.2f}% (epoch {best_epoch_s1})")
    print(f"Stage 2 (Magnitude): {best_acc_s2:.2f}% (epoch {best_epoch_s2})")
    print(f"Stage 3 (Azimuth):   {best_acc_s3:.2f}% (epoch {best_epoch_s3})")
    print(f"\nModels saved to: {save_dir}")
    print("=" * 70)
    
    return results, save_dir

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='efficientnet', 
                        choices=['efficientnet', 'convnext'])
    args = parser.parse_args()
    
    main(args.backbone)
