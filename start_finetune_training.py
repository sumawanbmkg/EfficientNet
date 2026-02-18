#!/usr/bin/env python3
"""
Finetune ConvNeXt Training Script
Finetunes the best checkpoint (Epoch 5) with fixed image size at 224x224
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

class FinetuneTrainer(ConvNeXtTrainer):
    """Trainer wrapper for finetuning from best checkpoint"""
    
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
            
        # Create NEW experiment directory for finetuning
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(f"experiments_convnext/finetune_v3_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save new configuration
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
            
        # Setup logging
        self.log_file = self.exp_dir / 'training.log'
        # Start fresh history for finetuning phase
        self.history = {
            'train_loss': [], 'train_mag_acc': [], 'train_az_acc': [],
            'val_loss': [], 'val_mag_acc': [], 'val_az_acc': [],
            'learning_rates': []
        }
        
        print(f">> Starting Finetuning Experiment: {self.exp_dir}")
        self.log(f"Finetuning from Best Checkpoint (Epoch {self.checkpoint['epoch'] + 1})")
        self.log(f"Fixed Image Size: {self.config['resize_schedule'][0]}")
        self.log(f"Device: {self.device}")

    def finetune(self):
        """Finetune training loop"""
        self.log("=" * 80)
        self.log(f"STARTING FINETUNING (Fixed 224x224)")
        self.log("=" * 80)
        
        # Setup model
        model, criterion = self.setup_model()
        model.load_state_dict(self.checkpoint['model_state_dict'])
        
        # Setup optimizer with lower learning rate for finetuning
        # We re-initialize optimizer to clear momentum from previous bad state
        # but usage lower LR to preserve good weights
        finetune_lr = self.config.get('base_lr', 1e-4) * 0.1  # 10x smaller learning rate
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=finetune_lr,
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # New scheduler for finetuning epochs
        total_epochs = self.config['epochs']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        
        # Setup AMP and EMA
        scaler = GradScaler()
        ema_model = EMAModel(model, decay=self.config.get('ema_decay', 0.9999))
        
        # Force Fixed Image Size
        current_size = 224
        self.log(f"Setting fixed image size: {current_size}")
        
        # Setup data loaders for 224x224
        train_loader, val_loader, test_loader = self.setup_data(current_size)
        
        best_val_mag_acc = self.checkpoint['metrics']['magnitude_acc']
        self.log(f"Baseline validation accuracy: {best_val_mag_acc:.4f}")
            
        start_time = time.time()
        
        for epoch in range(total_epochs):
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
            self.log(f"Epoch {epoch+1}/{total_epochs} completed in {epoch_time:.1f}s")
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
                self.log(f"🔥 NEW BEST ACCURACY: {best_val_mag_acc:.4f}")
            
            self.save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, is_best)
            
            # Save history
            with open(self.exp_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        
        # Training completed
        total_time = time.time() - start_time
        self.log("=" * 80)
        self.log(f"FINETUNING COMPLETED in {total_time/60:.1f} minutes")
        self.log(f"Best Validation Magnitude Accuracy: {best_val_mag_acc:.4f}")
        self.log(f"Results saved to: {self.exp_dir}")
        self.log("=" * 80)
        
        # Test with best model
        self.log("\n[TEST] Evaluating best model on test set...")
        best_checkpoint_path = self.exp_dir / 'checkpoint_best.pth'
        if best_checkpoint_path.exists():
            best_checkpoint = torch.load(best_checkpoint_path)
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
    parser = argparse.ArgumentParser(description='Finetune ConvNeXt Training')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to BEST checkpoint file (checkpoint_best.pth)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of finetuning epochs')
    
    args = parser.parse_args()
    
    # Override config to disable progressive resizing and set fixed size
    config_override = {
        'epochs': args.epochs,
        'resize_schedule': {0: 224},  # Force 224x224 from start
        'base_lr': 5e-4 # Reduce LR slightly for finetuning start
    }
    
    trainer = FinetuneTrainer(args.checkpoint, config_override)
    trainer.finetune()
    
    print("\n[SUCCESS] Finetuning completed!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        # traceback.print_exc() 
        sys.exit(1)
