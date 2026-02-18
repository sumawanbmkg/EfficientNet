#!/usr/bin/env python3
"""
ConvNeXt Training Script V3.0 - Publication-Ready Training Pipeline
Multi-task earthquake prediction from ULF geomagnetic spectrograms

Target Journals:
- IEEE Transactions on Geoscience and Remote Sensing (TGRS)
- Journal of Geophysical Research (JGR): Solid Earth
- Scientific Reports (Nature Portfolio)

Features:
- ConvNeXt-Tiny backbone with multi-task learning
- Progressive resizing for efficient training
- Automatic Mixed Precision (AMP)
- EMA model tracking
- Advanced augmentation (RandAugment, MixUp, CutMix)
- Comprehensive logging and checkpointing

Author: Earthquake Prediction Research Team
Date: 13 February 2026
Version: 3.0
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# Import custom modules
from earthquake_cnn_v3 import (
    create_model_v3, 
    get_model_config, 
    EMAModel, 
    EarthquakeCNNV3,
    MultiTaskLossV3
)
from earthquake_dataset_v3 import create_dataloaders_v3


class ProgressiveResizer:
    """Progressive resizing scheduler for efficient training"""
    
    def __init__(self, schedule: Dict[int, int]):
        """
        Initialize progressive resizer
        
        Args:
            schedule: Dictionary mapping epoch to image size
                     e.g., {0: 112, 20: 168, 40: 224}
        """
        self.schedule = sorted(schedule.items())
        
    def get_size(self, epoch: int) -> int:
        """Get image size for current epoch"""
        size = self.schedule[0][1]  # Default to first size
        for epoch_threshold, new_size in self.schedule:
            if epoch >= epoch_threshold:
                size = new_size
        return size


class MetricTracker:
    """Track and compute training metrics"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.magnitude_loss = 0.0
        self.azimuth_loss = 0.0
        self.magnitude_correct = 0
        self.azimuth_correct = 0
        self.total_samples = 0
        
    def update(self, loss_dict: Dict[str, torch.Tensor], 
               mag_pred: torch.Tensor, mag_target: torch.Tensor,
               az_pred: torch.Tensor, az_target: torch.Tensor):
        """Update metrics with batch results"""
        batch_size = mag_pred.size(0)
        
        self.total_loss += loss_dict['total_loss'].item() * batch_size
        self.magnitude_loss += loss_dict['magnitude_loss'].item() * batch_size
        self.azimuth_loss += loss_dict['azimuth_loss'].item() * batch_size
        
        self.magnitude_correct += (mag_pred.argmax(1) == mag_target).sum().item()
        self.azimuth_correct += (az_pred.argmax(1) == az_target).sum().item()
        self.total_samples += batch_size
        
    def get_metrics(self) -> Dict[str, float]:
        """Get averaged metrics"""
        return {
            'loss': self.total_loss / self.total_samples,
            'magnitude_loss': self.magnitude_loss / self.total_samples,
            'azimuth_loss': self.azimuth_loss / self.total_samples,
            'magnitude_acc': self.magnitude_correct / self.total_samples,
            'azimuth_acc': self.azimuth_correct / self.total_samples
        }


class ConvNeXtTrainer:
    """Complete training pipeline for ConvNeXt earthquake prediction"""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(f"experiments_convnext/exp_v3_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Setup logging
        self.log_file = self.exp_dir / 'training.log'
        self.history = {
            'train_loss': [], 'train_mag_acc': [], 'train_az_acc': [],
            'val_loss': [], 'val_mag_acc': [], 'val_az_acc': [],
            'learning_rates': []
        }
        
        print(f">> Training experiment: {self.exp_dir}")
        self.log(f"Experiment started at {timestamp}")
        self.log(f"Device: {self.device}")
        
    def log(self, message: str):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def setup_model(self) -> Tuple[EarthquakeCNNV3, MultiTaskLossV3]:
        """Setup model and loss function"""
        self.log("Setting up ConvNeXt-Tiny model...")
        
        model, criterion = create_model_v3(self.config)
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.log(f"Model: ConvNeXt-Tiny Multi-Task")
        self.log(f"Total parameters: {total_params:,}")
        self.log(f"Trainable parameters: {trainable_params:,}")
        self.log(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
        
        return model, criterion
    
    def setup_data(self, image_size: int = 224) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup data loaders"""
        self.log(f"Setting up data loaders (image size: {image_size})...")
        
        train_loader, val_loader, test_loader = create_dataloaders_v3(
            dataset_dir=self.config.get('dataset_path', 'dataset_experiment_3'),
            batch_size=self.config['batch_size'],
            image_size=image_size,
            num_workers=self.config.get('num_workers', 4)
        )
        
        self.log(f"Train batches: {len(train_loader)}")
        self.log(f"Validation batches: {len(val_loader)}")
        self.log(f"Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def setup_optimizer(self, model: nn.Module) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Setup optimizer and learning rate scheduler"""
        self.log("Setting up optimizer and scheduler...")
        
        # AdamW optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['base_lr'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing scheduler with warmup
        warmup_epochs = self.config.get('warmup_epochs', 5)
        total_epochs = self.config['epochs']
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        self.log(f"Optimizer: AdamW (lr={self.config['base_lr']}, wd={self.config['weight_decay']})")
        self.log(f"Scheduler: Cosine annealing with {warmup_epochs} warmup epochs")
        
        return optimizer, scheduler
    
    def train_epoch(self, model: nn.Module, criterion: nn.Module,
                   train_loader: DataLoader, optimizer: optim.Optimizer,
                   scaler: GradScaler, ema_model: Optional[EMAModel],
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        model.train()
        metrics = MetricTracker()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')
        
        for batch_idx, (images, mag_labels, az_labels) in enumerate(pbar):
            images = images.to(self.device)
            mag_labels = mag_labels.to(self.device)
            az_labels = az_labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with AMP
            with autocast():
                mag_logits, az_logits = model(images)
                loss_dict = criterion(mag_logits, az_logits, mag_labels, az_labels)
                loss = loss_dict['total_loss']
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            if ema_model is not None:
                ema_model.update()
            
            # Update metrics
            metrics.update(loss_dict, mag_logits, mag_labels, az_logits, az_labels)
            
            # Update progress bar
            current_metrics = metrics.get_metrics()
            pbar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'mag_acc': f"{current_metrics['magnitude_acc']:.3f}",
                'az_acc': f"{current_metrics['azimuth_acc']:.3f}"
            })
        
        return metrics.get_metrics()
    
    @torch.no_grad()
    def validate_epoch(self, model: nn.Module, criterion: nn.Module,
                      val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        model.eval()
        metrics = MetricTracker()
        
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Val]')
        
        for images, mag_labels, az_labels in pbar:
            images = images.to(self.device)
            mag_labels = mag_labels.to(self.device)
            az_labels = az_labels.to(self.device)
            
            # Forward pass
            with autocast():
                mag_logits, az_logits = model(images)
                loss_dict = criterion(mag_logits, az_logits, mag_labels, az_labels)
            
            # Update metrics
            metrics.update(loss_dict, mag_logits, mag_labels, az_logits, az_labels)
            
            # Update progress bar
            current_metrics = metrics.get_metrics()
            pbar.set_postfix({
                'loss': f"{current_metrics['loss']:.4f}",
                'mag_acc': f"{current_metrics['magnitude_acc']:.3f}",
                'az_acc': f"{current_metrics['azimuth_acc']:.3f}"
            })
        
        return metrics.get_metrics()
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       scheduler: optim.lr_scheduler._LRScheduler,
                       epoch: int, metrics: Dict[str, float], 
                       is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'history': self.history
        }
        
        # Save latest checkpoint
        checkpoint_path = self.exp_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.exp_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            self.log(f"[BEST] Model saved! Mag Acc: {metrics['magnitude_acc']:.4f}")
    
    def train(self):
        """Main training loop"""
        self.log("=" * 80)
        self.log("STARTING CONVNEXT TRAINING FOR EARTHQUAKE PREDICTION")
        self.log("=" * 80)
        
        # Setup
        model, criterion = self.setup_model()
        train_loader, val_loader, test_loader = self.setup_data()
        optimizer, scheduler = self.setup_optimizer(model)
        
        # Setup AMP and EMA
        scaler = GradScaler()
        ema_model = EMAModel(model, decay=self.config.get('ema_decay', 0.9999))
        
        # Progressive resizing
        resize_schedule = self.config.get('resize_schedule', {0: 224})
        resizer = ProgressiveResizer(resize_schedule)
        current_size = 224
        
        # Training loop
        best_val_mag_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # Check for size change
            new_size = resizer.get_size(epoch)
            if new_size != current_size:
                self.log(f"[RESIZE] Changing image size: {current_size} -> {new_size}")
                current_size = new_size
                train_loader, val_loader, test_loader = self.setup_data(current_size)
            
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(model, criterion, train_loader, 
                                            optimizer, scaler, ema_model, epoch)
            
            # Validate
            val_metrics = self.validate_epoch(model, criterion, val_loader, epoch)
            
            # Update scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_mag_acc'].append(train_metrics['magnitude_acc'])
            self.history['train_az_acc'].append(train_metrics['azimuth_acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mag_acc'].append(val_metrics['magnitude_acc'])
            self.history['val_az_acc'].append(val_metrics['azimuth_acc'])
            self.history['learning_rates'].append(current_lr)
            
            # Log epoch results
            epoch_time = time.time() - epoch_start
            self.log("-" * 80)
            self.log(f"Epoch {epoch+1}/{self.config['epochs']} completed in {epoch_time:.1f}s")
            self.log(f"LR: {current_lr:.6f}")
            self.log(f"Train - Loss: {train_metrics['loss']:.4f}, "
                    f"Mag Acc: {train_metrics['magnitude_acc']:.4f}, "
                    f"Az Acc: {train_metrics['azimuth_acc']:.4f}")
            self.log(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                    f"Mag Acc: {val_metrics['magnitude_acc']:.4f}, "
                    f"Az Acc: {val_metrics['azimuth_acc']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['magnitude_acc'] > best_val_mag_acc
            if is_best:
                best_val_mag_acc = val_metrics['magnitude_acc']
            
            self.save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, is_best)
            
            # Save history
            with open(self.exp_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        
        # Training completed
        total_time = time.time() - start_time
        self.log("=" * 80)
        self.log(f"TRAINING COMPLETED in {total_time/60:.1f} minutes")
        self.log(f"Best Validation Magnitude Accuracy: {best_val_mag_acc:.4f}")
        self.log(f"Results saved to: {self.exp_dir}")
        self.log("=" * 80)
        
        # Test with best model
        self.log("\n[TEST] Evaluating best model on test set...")
        best_checkpoint = torch.load(self.exp_dir / 'checkpoint_best.pth')
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        test_metrics = self.validate_epoch(model, criterion, test_loader, epoch=-1)
        self.log(f"Test Results - Mag Acc: {test_metrics['magnitude_acc']:.4f}, "
                f"Az Acc: {test_metrics['azimuth_acc']:.4f}")
        
        # Save final results
        final_results = {
            'best_val_magnitude_acc': best_val_mag_acc,
            'test_magnitude_acc': test_metrics['magnitude_acc'],
            'test_azimuth_acc': test_metrics['azimuth_acc'],
            'total_training_time_minutes': total_time / 60,
            'experiment_dir': str(self.exp_dir)
        }
        
        with open(self.exp_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        return final_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train ConvNeXt for Earthquake Prediction')
    parser.add_argument('--dataset', type=str, default='dataset_experiment_3',
                       help='Dataset directory path')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=60,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Base learning rate')
    parser.add_argument('--no-progressive', action='store_true',
                       help='Disable progressive resizing')
    
    args = parser.parse_args()
    
    # Get base configuration
    config = get_model_config()
    
    # Update with command line arguments
    config['dataset_path'] = args.dataset
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epochs
    config['base_lr'] = args.lr
    
    if args.no_progressive:
        config['resize_schedule'] = {0: 224}
    
    # Create trainer and run
    trainer = ConvNeXtTrainer(config)
    results = trainer.train()
    
    print("\n" + "=" * 80)
    print("[SUCCESS] TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"[RESULTS] Best Validation Magnitude Accuracy: {results['best_val_magnitude_acc']:.4f}")
    print(f"[RESULTS] Test Magnitude Accuracy: {results['test_magnitude_acc']:.4f}")
    print(f"[RESULTS] Test Azimuth Accuracy: {results['test_azimuth_acc']:.4f}")
    print(f"[TIME] Total Training Time: {results['total_training_time_minutes']:.1f} minutes")
    print(f"[OUTPUT] Results saved to: {results['experiment_dir']}")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
