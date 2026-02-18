#!/usr/bin/env python3
"""
Local Retraining Script (Safe for High RAM Usage)
Finetunes the best checkpoint on CPU with fixed image size and reduced batch size
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# Import from existing modules
from earthquake_cnn_v3 import create_model_v3, EMAModel
from train_earthquake_v3 import ConvNeXtTrainer, ProgressiveResizer, MetricTracker

class LocalFinetuneTrainer(ConvNeXtTrainer):
    """Trainer wrapper for local finetuning"""
    
    def __init__(self, checkpoint_path, config_override=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        print(f">> Loading BEST checkpoint: {self.checkpoint_path}")
        try:
            self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Restore configuration but apply overrides
        self.config = self.checkpoint['config']
        if config_override:
            self.config.update(config_override)
            
        # Create NEW experiment directory for local finetuning
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(f"experiments_convnext/local_finetune_v3_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save new configuration
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
            
        # Setup logging
        self.log_file = self.exp_dir / 'training.log'
        self.history = {
            'train_loss': [], 'train_mag_acc': [], 'train_az_acc': [],
            'val_loss': [], 'val_mag_acc': [], 'val_az_acc': [],
            'learning_rates': []
        }
        
        print(f">> Starting Local Finetuning Experiment: {self.exp_dir}")
        self.log(f"Finetuning from Best Checkpoint (Epoch {self.checkpoint['epoch'] + 1})")
        self.log(f"Fixed Image Size: {self.config['resize_schedule'][0]}")
        self.log(f"Device: {self.device}")
        self.log(f"Batch Size: {self.config['batch_size']}")
        self.log(f"Workers: {self.config['num_workers']}")

    def finetune(self):
        """Finetune training loop"""
        self.log("=" * 80)
        self.log(f"STARTING LOCAL FINETUNING (CPUSAFE)")
        self.log("=" * 80)
        
        # Setup model
        model, criterion = self.setup_model()
        model.load_state_dict(self.checkpoint['model_state_dict'])
        
        # Setup optimizer with lower learning rate for finetuning
        finetune_lr = self.config.get('base_lr', 1e-4) * 0.1  # Reduce LR for finetuning
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=finetune_lr,
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999), 
            eps=1e-8
        )
        
        # New scheduler
        total_epochs = self.config['epochs']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
        
        # Setup AMP: Disable on CPU to avoid "autocast is deprecated" warnings or issues
        use_amp = (self.device.type == 'cuda')
        scaler = GradScaler() if use_amp else None
        
        ema_model = EMAModel(model, decay=self.config.get('ema_decay', 0.9999))
        
        # Force Fixed Image Size
        current_size = 224
        
        # Setup data loaders for 224x224 - Using override config
        # We manually call create_dataloaders here to ensure overrides are respected if setup_data doesn't use self.config completely
        from earthquake_dataset_v3 import create_dataloaders_v3
        
        self.log(f"Setting up data loaders with size={current_size}, batch={self.config['batch_size']}, workers={self.config['num_workers']}")
        train_loader, val_loader, test_loader = create_dataloaders_v3(
            dataset_dir=self.config.get('dataset_path', 'dataset_experiment_3'),
            batch_size=self.config['batch_size'],
            image_size=current_size,
            num_workers=self.config['num_workers']
        )
        
        best_val_mag_acc = self.checkpoint['metrics']['magnitude_acc']
        self.log(f"Baseline validation accuracy: {best_val_mag_acc:.4f}")
            
        start_time = time.time()
        
        for epoch in range(total_epochs):
            epoch_start = time.time()
            
            # Train
            model.train()
            metrics = MetricTracker()
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
            
            for batch_idx, (images, mag_labels, az_labels) in enumerate(pbar):
                images = images.to(self.device)
                mag_labels = mag_labels.to(self.device)
                az_labels = az_labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass - No AMP on CPU usually safer
                if use_amp:
                    with autocast():
                        mag_logits, az_logits = model(images)
                        loss_dict = criterion(mag_logits, az_logits, mag_labels, az_labels)
                        loss = loss_dict['total_loss']
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    mag_logits, az_logits = model(images)
                    loss_dict = criterion(mag_logits, az_logits, mag_labels, az_labels)
                    loss = loss_dict['total_loss']
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Update EMA
                if ema_model is not None:
                    ema_model.update()
                
                metrics.update(loss_dict, mag_logits, mag_labels, az_logits, az_labels)
                
                current_metrics = metrics.get_metrics()
                pbar.set_postfix({
                    'loss': f"{current_metrics['loss']:.4f}",
                    'mag_acc': f"{current_metrics['magnitude_acc']:.3f}"
                })
            
            train_metrics = metrics.get_metrics()
            
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
            self.log(f"Epoch {epoch+1}/{total_epochs} completed in {epoch_time:.1f}s")
            self.log(f"LR: {current_lr:.6f}")
            self.log(f"Train - Loss: {train_metrics['loss']:.4f}, Mag Acc: {train_metrics['magnitude_acc']:.4f}")
            self.log(f"Val   - Loss: {val_metrics['loss']:.4f}, Mag Acc: {val_metrics['magnitude_acc']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['magnitude_acc'] > best_val_mag_acc
            if is_best:
                best_val_mag_acc = val_metrics['magnitude_acc']
                self.log(f"🔥 NEW BEST ACCURACY: {best_val_mag_acc:.4f}")
            
            self.save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, is_best)
            
            # Save history
            with open(self.exp_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        
        # Training completed
        total_time = time.time() - start_time
        self.log("=" * 80)
        self.log(f"LOCAL RETRAINING COMPLETED in {total_time/60:.1f} minutes")
        self.log(f"Best Validation Magnitude Accuracy: {best_val_mag_acc:.4f}")
        self.log(f"Results saved to: {self.exp_dir}")
        self.log("=" * 80)
        
        return str(self.exp_dir)

def main():
    parser = argparse.ArgumentParser(description='Local Retraining (CPU Safe)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to BEST checkpoint file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    
    args = parser.parse_args()
    
    # Safe CPU Configuration
    config_override = {
        'epochs': args.epochs,
        'resize_schedule': {0: 224},  # Force 224x224
        'base_lr': 1e-4,              
        'batch_size': 16,             # Reduce batch size for CPU/RAM safety
        'num_workers': 0              # 0 workers is safest for Windows multiprocessing
    }
    
    trainer = LocalFinetuneTrainer(args.checkpoint, config_override)
    trainer.finetune()
    
    print("\n[SUCCESS] Local retraining completed!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
