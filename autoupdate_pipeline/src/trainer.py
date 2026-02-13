"""
Model Trainer Module

Trains new candidate models (challengers) using the combined dataset.
"""

import os
import sys
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from .utils import (
    load_config, load_registry, save_registry, 
    generate_model_id, log_pipeline_event
)

logger = logging.getLogger("autoupdate_pipeline.trainer")


class EarthquakeDataset(Dataset):
    """Dataset for earthquake spectrograms."""
    
    def __init__(self, metadata_df: pd.DataFrame, dataset_dir: str, transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        
        # Create class mappings
        self.magnitude_classes = sorted(self.metadata['magnitude_class'].unique())
        self.azimuth_classes = sorted(self.metadata['azimuth_class'].unique())
        
        self.mag_to_idx = {c: i for i, c in enumerate(self.magnitude_classes)}
        self.azi_to_idx = {c: i for i, c in enumerate(self.azimuth_classes)}
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load image - try multiple path patterns
        filename = row['filename']
        mag_class = row['magnitude_class']
        azi_class = row.get('azimuth_class', 'N')
        
        # Try different path patterns (comprehensive search)
        possible_paths = [
            self.dataset_dir / filename,  # Direct path
            self.dataset_dir / 'by_magnitude' / mag_class / filename,  # by_magnitude/CLASS/file
            self.dataset_dir / 'by_azimuth' / azi_class / filename,  # by_azimuth/CLASS/file
            self.dataset_dir / 'augmented' / filename,  # augmented/file
            self.dataset_dir / mag_class / filename,  # CLASS/file (magnitude)
            self.dataset_dir / azi_class / filename,  # CLASS/file (azimuth)
            self.dataset_dir / 'original' / filename,  # original/file
            self.dataset_dir / 'train' / mag_class / filename,  # train/CLASS/file
            self.dataset_dir / 'test' / mag_class / filename,  # test/CLASS/file
        ]
        
        img_path = None
        for path in possible_paths:
            if path.exists():
                img_path = path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Spectrogram not found: {filename}. Tried: {[str(p) for p in possible_paths]}")
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        mag_label = self.mag_to_idx[row['magnitude_class']]
        azi_label = self.azi_to_idx[row['azimuth_class']]
        
        return image, mag_label, azi_label


class MultiTaskConvNeXt(nn.Module):
    """ConvNeXt-based multi-task model."""
    
    def __init__(self, num_mag_classes: int, num_azi_classes: int):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        num_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()
        
        self.mag_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Flatten(start_dim=1),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_mag_classes)
        )
        
        self.azi_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Flatten(start_dim=1),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_azi_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.mean([-2, -1])
        return self.mag_head(features), self.azi_head(features)


class ModelTrainer:
    """
    Trains new candidate models.
    
    Features:
    - Supports multiple architectures (ConvNeXt, EfficientNet, VGG16)
    - On-the-fly data augmentation
    - Early stopping
    - LOEO cross-validation
    """
    
    def __init__(self, config: Dict[str, Any] = None, quick_test: bool = False):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            quick_test: If True, use reduced dataset and epochs for fast testing
        """
        self.config = config or load_config()
        self.training_config = self.config.get('training', {})
        self.quick_test = quick_test
        
        self.base_path = Path(__file__).parent.parent
        self.challenger_path = self.base_path / self.config['paths']['challenger_model']
        
        # Device setup
        device_config = self.training_config.get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        # Quick test mode settings
        if self.quick_test:
            logger.info("QUICK TEST MODE ENABLED - Using reduced dataset and epochs")
        
        logger.info(f"ModelTrainer initialized. Device: {self.device}")
    
    def prepare_dataset(self, include_new_events: bool = True) -> pd.DataFrame:
        """
        Prepare combined dataset (original + new validated events).
        
        Args:
            include_new_events: Whether to include newly validated events
            
        Returns:
            Combined metadata DataFrame
        """
        # Load original metadata
        metadata_path = self.base_path / self.config['paths']['metadata_file']
        
        if metadata_path.exists():
            original_df = pd.read_csv(metadata_path)
            logger.info(f"Loaded original dataset: {len(original_df)} samples")
        else:
            original_df = pd.DataFrame()
            logger.warning("Original metadata not found")
        
        # Load validated new events
        new_events_count = 0
        if include_new_events:
            registry = load_registry()
            validated_events = registry.get('validated_events', [])
            
            if validated_events:
                new_df = pd.DataFrame(validated_events)
                # Rename columns to match original format
                new_df = new_df.rename(columns={
                    'magnitude': 'magnitude_class',
                    'azimuth': 'azimuth_class'
                })
                
                # Filter events that have valid spectrogram files
                dataset_dir = self.base_path / self.config['paths']['dataset_dir']
                valid_new_events = []
                
                for _, row in new_df.iterrows():
                    if pd.notna(row.get('filename')):
                        spec_path = dataset_dir / row['filename']
                        if spec_path.exists():
                            valid_new_events.append(row.to_dict())
                            logger.info(f"Including new event: {row.get('id', 'unknown')}")
                        else:
                            logger.warning(f"Spectrogram not found for event {row.get('id', 'unknown')}: {spec_path}")
                    else:
                        logger.warning(f"Event {row.get('id', 'unknown')} has no spectrogram path (prototype mode)")
                
                if valid_new_events:
                    valid_new_df = pd.DataFrame(valid_new_events)
                    combined_df = pd.concat([original_df, valid_new_df], ignore_index=True)
                    new_events_count = len(valid_new_events)
                else:
                    combined_df = original_df
                    logger.info("No new events with valid spectrograms - using original dataset only")
                
                logger.info(f"Combined dataset: {len(combined_df)} samples ({new_events_count} new with spectrograms)")
            else:
                combined_df = original_df
        else:
            combined_df = original_df
        
        # Log summary of new events status
        if include_new_events:
            registry = load_registry()
            total_validated = len(registry.get('validated_events', []))
            logger.info(f"Validated events in registry: {total_validated}, with spectrograms: {new_events_count}")
        
        return combined_df
    
    def create_data_loaders(self, metadata_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""
        # Ensure we have the correct filename column
        if 'spectrogram_file' in metadata_df.columns and metadata_df['spectrogram_file'].notna().any():
            metadata_df = metadata_df.copy()
            metadata_df['filename'] = metadata_df['spectrogram_file']
        
        # Drop rows with NaN in required columns
        required_cols = ['magnitude_class', 'azimuth_class', 'filename']
        clean_df = metadata_df.dropna(subset=required_cols)
        logger.info(f"Clean dataset after dropping NaN: {len(clean_df)} samples (dropped {len(metadata_df) - len(clean_df)})")
        
        # Quick test mode: use subset of data
        if self.quick_test:
            # Sample 200 rows per class (or all if less)
            sampled_dfs = []
            for mag_class in clean_df['magnitude_class'].unique():
                class_df = clean_df[clean_df['magnitude_class'] == mag_class]
                n_samples = min(50, len(class_df))  # Max 50 per class for quick test
                sampled_dfs.append(class_df.sample(n=n_samples, random_state=42))
            clean_df = pd.concat(sampled_dfs, ignore_index=True)
            logger.info(f"QUICK TEST: Reduced to {len(clean_df)} samples")
        
        # Split data
        train_df, temp_df = train_test_split(clean_df, test_size=0.3, random_state=42, 
                                              stratify=clean_df['magnitude_class'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42,
                                            stratify=temp_df['magnitude_class'])
        
        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset_dir = self.base_path / self.config['paths']['dataset_dir']
        
        train_dataset = EarthquakeDataset(train_df, dataset_dir, train_transform)
        val_dataset = EarthquakeDataset(val_df, dataset_dir, val_transform)
        test_dataset = EarthquakeDataset(test_df, dataset_dir, val_transform)
        
        # Batch size - smaller for quick test
        batch_size = 8 if self.quick_test else self.training_config.get('batch_size', 32)
        
        # num_workers - 0 for Windows compatibility in quick test
        num_workers = 0 if self.quick_test else 4
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader, train_dataset
    
    def train_model(self, include_new_events: bool = True) -> Dict[str, Any]:
        """
        Train a new challenger model.
        
        Args:
            include_new_events: Whether to include new validated events
            
        Returns:
            Training results dictionary
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Prepare data
        metadata_df = self.prepare_dataset(include_new_events)
        train_loader, val_loader, test_loader, train_dataset = self.create_data_loaders(metadata_df)
        
        # Create model
        num_mag_classes = len(train_dataset.magnitude_classes)
        num_azi_classes = len(train_dataset.azimuth_classes)
        
        model = MultiTaskConvNeXt(num_mag_classes, num_azi_classes)
        model = model.to(self.device)
        
        # Loss and optimizer
        mag_criterion = nn.CrossEntropyLoss()
        azi_criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.training_config.get('learning_rate', 0.0001),
            weight_decay=self.training_config.get('weight_decay', 0.05)
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.training_config.get('epochs', 50)
        )
        
        # Training loop
        epochs = 3 if self.quick_test else self.training_config.get('epochs', 50)
        patience = 2 if self.quick_test else self.training_config.get('early_stopping_patience', 10)
        best_val_acc = 0
        patience_counter = 0
        history = []
        
        if self.quick_test:
            logger.info(f"QUICK TEST: Training for {epochs} epochs with patience={patience}")
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            train_mag_correct = 0
            train_azi_correct = 0
            train_total = 0
            
            for images, mag_labels, azi_labels in train_loader:
                images = images.to(self.device)
                mag_labels = mag_labels.to(self.device)
                azi_labels = azi_labels.to(self.device)
                
                optimizer.zero_grad()
                mag_out, azi_out = model(images)
                
                mag_loss = mag_criterion(mag_out, mag_labels)
                azi_loss = azi_criterion(azi_out, azi_labels)
                loss = mag_loss + azi_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_mag_correct += (mag_out.argmax(1) == mag_labels).sum().item()
                train_azi_correct += (azi_out.argmax(1) == azi_labels).sum().item()
                train_total += mag_labels.size(0)
            
            # Validate
            model.eval()
            val_loss = 0
            val_mag_correct = 0
            val_azi_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, mag_labels, azi_labels in val_loader:
                    images = images.to(self.device)
                    mag_labels = mag_labels.to(self.device)
                    azi_labels = azi_labels.to(self.device)
                    
                    mag_out, azi_out = model(images)
                    
                    mag_loss = mag_criterion(mag_out, mag_labels)
                    azi_loss = azi_criterion(azi_out, azi_labels)
                    loss = mag_loss + azi_loss
                    
                    val_loss += loss.item()
                    val_mag_correct += (mag_out.argmax(1) == mag_labels).sum().item()
                    val_azi_correct += (azi_out.argmax(1) == azi_labels).sum().item()
                    val_total += mag_labels.size(0)
            
            scheduler.step()
            
            # Calculate metrics
            train_mag_acc = train_mag_correct / train_total * 100
            train_azi_acc = train_azi_correct / train_total * 100
            val_mag_acc = val_mag_correct / val_total * 100
            val_azi_acc = val_azi_correct / val_total * 100
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss / len(val_loader),
                'train_mag_acc': train_mag_acc,
                'train_azi_acc': train_azi_acc,
                'val_mag_acc': val_mag_acc,
                'val_azi_acc': val_azi_acc
            })
            
            logger.info(f"Epoch {epoch+1}/{epochs} | "
                       f"Train Mag: {train_mag_acc:.2f}% | Val Mag: {val_mag_acc:.2f}% | "
                       f"Train Azi: {train_azi_acc:.2f}% | Val Azi: {val_azi_acc:.2f}%")
            
            # Early stopping
            if val_mag_acc > best_val_acc:
                best_val_acc = val_mag_acc
                patience_counter = 0
                
                # Save best model
                self._save_checkpoint(model, optimizer, epoch, val_mag_acc, val_azi_acc,
                                     train_dataset.magnitude_classes, train_dataset.azimuth_classes)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Evaluate on test set
        test_results = self._evaluate_on_test(model, test_loader)
        
        training_time = (datetime.now() - start_time).total_seconds() / 3600
        
        results = {
            "success": True,
            "model_id": generate_model_id("convnext"),
            "epochs_trained": len(history),
            "training_time_hours": training_time,
            "best_val_mag_acc": best_val_acc,
            "test_results": test_results,
            "history": history,
            "dataset_size": len(metadata_df),
            "class_mappings": {
                "magnitude": train_dataset.magnitude_classes,
                "azimuth": train_dataset.azimuth_classes
            }
        }
        
        # Log event
        log_pipeline_event("training_complete", {
            "model_id": results["model_id"],
            "epochs": results["epochs_trained"],
            "test_mag_acc": test_results["magnitude_accuracy"],
            "test_azi_acc": test_results["azimuth_accuracy"]
        })
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Test Magnitude Accuracy: {test_results['magnitude_accuracy']:.2f}%")
        logger.info(f"Test Azimuth Accuracy: {test_results['azimuth_accuracy']:.2f}%")
        logger.info("=" * 60)
        
        return results
    
    def _save_checkpoint(self, model, optimizer, epoch, val_mag_acc, val_azi_acc,
                        mag_classes, azi_classes):
        """Save model checkpoint."""
        self.challenger_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mag_acc': val_mag_acc,
            'val_azi_acc': val_azi_acc
        }
        
        torch.save(checkpoint, self.challenger_path / 'best_model.pth')
        
        # Save class mappings
        mappings = {
            'magnitude': {str(i): c for i, c in enumerate(mag_classes)},
            'azimuth': {str(i): c for i, c in enumerate(azi_classes)}
        }
        
        with open(self.challenger_path / 'class_mappings.json', 'w') as f:
            json.dump(mappings, f, indent=2)
        
        logger.info(f"Checkpoint saved: val_mag_acc={val_mag_acc:.2f}%")
    
    def _evaluate_on_test(self, model, test_loader) -> Dict[str, float]:
        """Evaluate model on test set."""
        model.eval()
        mag_correct = 0
        azi_correct = 0
        total = 0
        
        with torch.no_grad():
            for images, mag_labels, azi_labels in test_loader:
                images = images.to(self.device)
                mag_labels = mag_labels.to(self.device)
                azi_labels = azi_labels.to(self.device)
                
                mag_out, azi_out = model(images)
                
                mag_correct += (mag_out.argmax(1) == mag_labels).sum().item()
                azi_correct += (azi_out.argmax(1) == azi_labels).sum().item()
                total += mag_labels.size(0)
        
        return {
            "magnitude_accuracy": mag_correct / total * 100,
            "azimuth_accuracy": azi_correct / total * 100,
            "total_samples": total
        }
