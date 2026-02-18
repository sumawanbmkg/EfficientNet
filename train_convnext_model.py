#!/usr/bin/env python3
"""
ConvNeXt Model Training for Earthquake Precursor Detection

ConvNeXt is a pure convolutional model that incorporates design choices from
Vision Transformers (ViT) while maintaining the efficiency of CNNs.

Key features:
- Patchify stem (4x4 non-overlapping convolution)
- Inverted bottleneck design
- Large kernel sizes (7x7)
- Layer normalization instead of batch normalization
- GELU activation
- Fewer activation functions and normalization layers

Reference: Liu et al., "A ConvNet for the 2020s" (CVPR 2022)

Author: Earthquake Prediction Research Team
Date: 5 February 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny, convnext_small, convnext_base
from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CONFIG = {
    # Model
    "model_variant": "tiny",  # Options: "tiny", "small", "base"
    "pretrained": True,
    "num_mag_classes": 4,
    "num_azi_classes": 9,
    
    # Data
    "dataset_dir": "dataset_unified/spectrograms",
    "metadata_path": "dataset_unified/metadata/unified_metadata.csv",
    "train_split": "dataset_unified/metadata/train_split.csv",
    "val_split": "dataset_unified/metadata/val_split.csv",
    "test_split": "dataset_unified/metadata/test_split.csv",
    "image_size": 224,
    
    # Training
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 0.05,  # ConvNeXt uses higher weight decay
    "lr_scheduler": "cosine",
    "warmup_epochs": 5,
    "early_stopping_patience": 10,
    
    # Augmentation
    "use_augmentation": True,
    "mixup_alpha": 0.8,
    "cutmix_alpha": 1.0,
    
    # Output
    "output_dir": "experiments_convnext",
    "save_best_only": True,
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(CONFIG["output_dir"]) / f"convnext_{CONFIG['model_variant']}_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save config
with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

print("=" * 70)
print("CONVNEXT MODEL TRAINING - Earthquake Precursor Detection")
print("=" * 70)
print(f"Model variant: ConvNeXt-{CONFIG['model_variant'].upper()}")
print(f"Output directory: {OUTPUT_DIR}")
print()


class EarthquakeDataset(Dataset):
    """Dataset for earthquake precursor spectrograms"""
    
    def __init__(self, metadata_df, img_dir, transform=None, 
                 mag_mapping=None, azi_mapping=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Create class mappings
        if mag_mapping is None:
            mag_classes = sorted(self.metadata['magnitude_class'].unique())
            self.mag_mapping = {c: i for i, c in enumerate(mag_classes)}
        else:
            self.mag_mapping = mag_mapping
            
        if azi_mapping is None:
            azi_classes = sorted(self.metadata['azimuth_class'].unique())
            self.azi_mapping = {c: i for i, c in enumerate(azi_classes)}
        else:
            self.azi_mapping = azi_mapping
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load image - support multiple column names
        if 'unified_path' in row.index:
            # unified_path is relative to dataset_unified folder
            img_path = Path("dataset_unified") / row['unified_path']
        elif 'spectrogram_file' in row.index:
            img_path = self.img_dir / row['spectrogram_file']
        elif 'filename' in row.index:
            img_path = self.img_dir / row['filename']
        else:
            raise KeyError(f"No valid image path column found. Available: {row.index.tolist()}")
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        mag_label = self.mag_mapping.get(row['magnitude_class'], 0)
        azi_label = self.azi_mapping.get(row['azimuth_class'], 0)
        
        return image, mag_label, azi_label


class ConvNeXtMultiTask(nn.Module):
    """ConvNeXt model with multi-task heads for magnitude and azimuth"""
    
    def __init__(self, variant="tiny", pretrained=True, 
                 num_mag_classes=4, num_azi_classes=9, dropout=0.5):
        super().__init__()
        
        # Load pretrained ConvNeXt
        if variant == "tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = convnext_tiny(weights=weights)
            num_features = 768
        elif variant == "small":
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = convnext_small(weights=weights)
            num_features = 768
        elif variant == "base":
            # ConvNeXt-Base doesn't have pretrained weights in older torchvision
            self.backbone = convnext_base(weights=None)
            num_features = 1024
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
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
        # Extract features
        features = self.backbone(x)
        
        # Flatten if needed (ConvNeXt outputs [B, C, 1, 1] without classifier)
        if features.dim() == 4:
            features = features.flatten(1)
        
        # Classification heads
        mag_out = self.mag_head(features)
        azi_out = self.azi_head(features)
        
        return mag_out, azi_out
    
    def get_features(self, x):
        """Get feature embeddings (for visualization)"""
        return self.backbone(x)


def get_transforms(is_training=True):
    """Get data transforms"""
    
    if is_training and CONFIG["use_augmentation"]:
        transform = transforms.Compose([
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    
    return transform


def compute_class_weights(metadata_df, column):
    """Compute inverse frequency class weights"""
    class_counts = metadata_df[column].value_counts()
    total = len(metadata_df)
    weights = {cls: total / (len(class_counts) * count) 
               for cls, count in class_counts.items()}
    return weights


def train_epoch(model, dataloader, optimizer, criterion_mag, criterion_azi, 
                device, scaler=None):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    mag_correct = 0
    azi_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for images, mag_labels, azi_labels in pbar:
        images = images.to(device)
        mag_labels = mag_labels.to(device)
        azi_labels = azi_labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                mag_out, azi_out = model(images)
                loss_mag = criterion_mag(mag_out, mag_labels)
                loss_azi = criterion_azi(azi_out, azi_labels)
                loss = loss_mag + 0.5 * loss_azi  # Weight magnitude more
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            mag_out, azi_out = model(images)
            loss_mag = criterion_mag(mag_out, mag_labels)
            loss_azi = criterion_azi(azi_out, azi_labels)
            loss = loss_mag + 0.5 * loss_azi
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Statistics
        total_loss += loss.item() * images.size(0)
        mag_pred = torch.argmax(mag_out, dim=1)
        azi_pred = torch.argmax(azi_out, dim=1)
        mag_correct += (mag_pred == mag_labels).sum().item()
        azi_correct += (azi_pred == azi_labels).sum().item()
        total_samples += images.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mag_acc': f'{mag_correct/total_samples:.4f}',
            'azi_acc': f'{azi_correct/total_samples:.4f}'
        })
    
    return {
        'loss': total_loss / total_samples,
        'mag_acc': mag_correct / total_samples,
        'azi_acc': azi_correct / total_samples
    }


def validate(model, dataloader, criterion_mag, criterion_azi, device):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    mag_correct = 0
    azi_correct = 0
    total_samples = 0
    
    all_mag_preds = []
    all_mag_labels = []
    all_azi_preds = []
    all_azi_labels = []
    
    with torch.no_grad():
        for images, mag_labels, azi_labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            mag_labels = mag_labels.to(device)
            azi_labels = azi_labels.to(device)
            
            mag_out, azi_out = model(images)
            
            loss_mag = criterion_mag(mag_out, mag_labels)
            loss_azi = criterion_azi(azi_out, azi_labels)
            loss = loss_mag + 0.5 * loss_azi
            
            total_loss += loss.item() * images.size(0)
            
            mag_pred = torch.argmax(mag_out, dim=1)
            azi_pred = torch.argmax(azi_out, dim=1)
            
            mag_correct += (mag_pred == mag_labels).sum().item()
            azi_correct += (azi_pred == azi_labels).sum().item()
            total_samples += images.size(0)
            
            all_mag_preds.extend(mag_pred.cpu().numpy())
            all_mag_labels.extend(mag_labels.cpu().numpy())
            all_azi_preds.extend(azi_pred.cpu().numpy())
            all_azi_labels.extend(azi_labels.cpu().numpy())
    
    # Calculate F1 scores
    mag_f1 = f1_score(all_mag_labels, all_mag_preds, average='weighted')
    azi_f1 = f1_score(all_azi_labels, all_azi_preds, average='weighted')
    
    return {
        'loss': total_loss / total_samples,
        'mag_acc': mag_correct / total_samples,
        'azi_acc': azi_correct / total_samples,
        'mag_f1': mag_f1,
        'azi_f1': azi_f1,
        'mag_preds': all_mag_preds,
        'mag_labels': all_mag_labels,
        'azi_preds': all_azi_preds,
        'azi_labels': all_azi_labels
    }


def plot_training_curves(history, output_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Magnitude Accuracy
    axes[0, 1].plot(epochs, history['train_mag_acc'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_mag_acc'], 'r-', label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Magnitude Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Azimuth Accuracy
    axes[1, 0].plot(epochs, history['train_azi_acc'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history['val_azi_acc'], 'r-', label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Azimuth Classification Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Scores
    axes[1, 1].plot(epochs, history['val_mag_f1'], 'g-', label='Magnitude F1')
    axes[1, 1].plot(epochs, history['val_azi_f1'], 'm-', label='Azimuth F1')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Validation F1 Scores')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()


def plot_confusion_matrices(val_results, mag_classes, azi_classes, output_dir):
    """Plot confusion matrices"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Magnitude confusion matrix
    mag_cm = confusion_matrix(val_results['mag_labels'], val_results['mag_preds'])
    sns.heatmap(mag_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=mag_classes, yticklabels=mag_classes, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Magnitude Classification Confusion Matrix')
    
    # Azimuth confusion matrix
    azi_cm = confusion_matrix(val_results['azi_labels'], val_results['azi_preds'])
    sns.heatmap(azi_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=azi_classes, yticklabels=azi_classes, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Azimuth Classification Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=150)
    plt.close()


def main():
    """Main training function"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    logger.info("Loading dataset...")
    
    # Check if split files exist
    if Path(CONFIG["train_split"]).exists():
        train_df = pd.read_csv(CONFIG["train_split"])
        val_df = pd.read_csv(CONFIG["val_split"])
        test_df = pd.read_csv(CONFIG["test_split"])
        logger.info(f"Loaded existing splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    else:
        # Load full metadata and create splits
        metadata_df = pd.read_csv(CONFIG["metadata_path"])
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        
        train_df, temp_df = train_test_split(
            metadata_df, test_size=0.3, 
            stratify=metadata_df['magnitude_class'],
            random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5,
            stratify=temp_df['magnitude_class'],
            random_state=42
        )
        logger.info(f"Created new splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Create class mappings
    all_data = pd.concat([train_df, val_df, test_df])
    mag_classes = sorted(all_data['magnitude_class'].unique())
    azi_classes = sorted(all_data['azimuth_class'].unique())
    
    mag_mapping = {c: i for i, c in enumerate(mag_classes)}
    azi_mapping = {c: i for i, c in enumerate(azi_classes)}
    
    logger.info(f"Magnitude classes: {mag_classes}")
    logger.info(f"Azimuth classes: {azi_classes}")
    
    # Save mappings
    mappings = {
        'magnitude': {str(i): c for c, i in mag_mapping.items()},
        'azimuth': {str(i): c for c, i in azi_mapping.items()}
    }
    with open(OUTPUT_DIR / 'class_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)
    
    # Create datasets
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    train_dataset = EarthquakeDataset(
        train_df, CONFIG["dataset_dir"], train_transform, mag_mapping, azi_mapping
    )
    val_dataset = EarthquakeDataset(
        val_df, CONFIG["dataset_dir"], val_transform, mag_mapping, azi_mapping
    )
    test_dataset = EarthquakeDataset(
        test_df, CONFIG["dataset_dir"], val_transform, mag_mapping, azi_mapping
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], 
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create model
    logger.info(f"Creating ConvNeXt-{CONFIG['model_variant'].upper()} model...")
    model = ConvNeXtMultiTask(
        variant=CONFIG["model_variant"],
        pretrained=CONFIG["pretrained"],
        num_mag_classes=len(mag_classes),
        num_azi_classes=len(azi_classes)
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Compute class weights
    mag_weights = compute_class_weights(train_df, 'magnitude_class')
    azi_weights = compute_class_weights(train_df, 'azimuth_class')
    
    mag_weight_tensor = torch.tensor([mag_weights[c] for c in mag_classes]).float().to(device)
    azi_weight_tensor = torch.tensor([azi_weights.get(c, 1.0) for c in azi_classes]).float().to(device)
    
    # Loss functions
    criterion_mag = nn.CrossEntropyLoss(weight=mag_weight_tensor)
    criterion_azi = nn.CrossEntropyLoss(weight=azi_weight_tensor)
    
    # Optimizer (AdamW with higher weight decay for ConvNeXt)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Learning rate scheduler (Cosine annealing with warmup)
    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = len(train_loader) * CONFIG["warmup_epochs"]
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mag_acc': [], 'val_mag_acc': [],
        'train_azi_acc': [], 'val_azi_acc': [],
        'val_mag_f1': [], 'val_azi_f1': [],
        'lr': []
    }
    
    best_val_mag_acc = 0
    patience_counter = 0
    
    logger.info("Starting training...")
    print()
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        print("-" * 50)
        
        # Train
        train_results = train_epoch(
            model, train_loader, optimizer, criterion_mag, criterion_azi,
            device, scaler
        )
        
        # Validate
        val_results = validate(
            model, val_loader, criterion_mag, criterion_azi, device
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_results['loss'])
        history['val_loss'].append(val_results['loss'])
        history['train_mag_acc'].append(train_results['mag_acc'])
        history['val_mag_acc'].append(val_results['mag_acc'])
        history['train_azi_acc'].append(train_results['azi_acc'])
        history['val_azi_acc'].append(val_results['azi_acc'])
        history['val_mag_f1'].append(val_results['mag_f1'])
        history['val_azi_f1'].append(val_results['azi_f1'])
        history['lr'].append(current_lr)
        
        # Print results
        print(f"\nTrain - Loss: {train_results['loss']:.4f}, "
              f"Mag Acc: {train_results['mag_acc']:.4f}, "
              f"Azi Acc: {train_results['azi_acc']:.4f}")
        print(f"Val   - Loss: {val_results['loss']:.4f}, "
              f"Mag Acc: {val_results['mag_acc']:.4f}, "
              f"Azi Acc: {val_results['azi_acc']:.4f}")
        print(f"Val F1 - Mag: {val_results['mag_f1']:.4f}, Azi: {val_results['azi_f1']:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_results['mag_acc'] > best_val_mag_acc:
            best_val_mag_acc = val_results['mag_acc']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mag_acc': val_results['mag_acc'],
                'val_azi_acc': val_results['azi_acc'],
                'config': CONFIG
            }, OUTPUT_DIR / 'best_model.pth')
            
            logger.info(f"✓ New best model saved! Mag Acc: {best_val_mag_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= CONFIG["early_stopping_patience"]:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': CONFIG
    }, OUTPUT_DIR / 'final_model.pth')
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / 'training_history.csv', index=False)
    
    # Plot training curves
    plot_training_curves(history, OUTPUT_DIR)
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    model.load_state_dict(torch.load(OUTPUT_DIR / 'best_model.pth')['model_state_dict'])
    test_results = validate(model, test_loader, criterion_mag, criterion_azi, device)
    
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"Magnitude Accuracy: {test_results['mag_acc']:.4f} ({test_results['mag_acc']*100:.2f}%)")
    print(f"Azimuth Accuracy:   {test_results['azi_acc']:.4f} ({test_results['azi_acc']*100:.2f}%)")
    print(f"Magnitude F1:       {test_results['mag_f1']:.4f}")
    print(f"Azimuth F1:         {test_results['azi_f1']:.4f}")
    
    # Plot confusion matrices
    plot_confusion_matrices(test_results, mag_classes, azi_classes, OUTPUT_DIR)
    
    # Classification reports
    print("\nMagnitude Classification Report:")
    print(classification_report(
        test_results['mag_labels'], test_results['mag_preds'],
        target_names=mag_classes
    ))
    
    print("\nAzimuth Classification Report:")
    print(classification_report(
        test_results['azi_labels'], test_results['azi_preds'],
        target_names=azi_classes
    ))
    
    # Save summary
    summary = {
        'model': f"ConvNeXt-{CONFIG['model_variant'].upper()}",
        'total_params': total_params,
        'trainable_params': trainable_params,
        'best_epoch': int(torch.load(OUTPUT_DIR / 'best_model.pth')['epoch']),
        'test_results': {
            'magnitude_accuracy': float(test_results['mag_acc']),
            'azimuth_accuracy': float(test_results['azi_acc']),
            'magnitude_f1': float(test_results['mag_f1']),
            'azimuth_f1': float(test_results['azi_f1'])
        },
        'training_samples': len(train_df),
        'validation_samples': len(val_df),
        'test_samples': len(test_df)
    }
    
    with open(OUTPUT_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Best model: {OUTPUT_DIR / 'best_model.pth'}")
    print(f"Training curves: {OUTPUT_DIR / 'training_curves.png'}")
    print(f"Confusion matrices: {OUTPUT_DIR / 'confusion_matrices.png'}")


if __name__ == "__main__":
    main()
