#!/usr/bin/env python3
"""
Resume ConvNeXt Training Script
Resumes training from a specific checkpoint
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

class ResumeTrainer(ConvNeXtTrainer):
    """Trainer wrapper for resuming training"""
    
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        print(f">> Loading checkpoint: {self.checkpoint_path}")
        # weights_only=False is needed for numpy 2.0 scalars in newer torch versions
        try:
            self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for older torch versions involved
            self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Restore configuration
        self.config = self.checkpoint['config']
        self.exp_dir = self.checkpoint_path.parent
        
        # Setup logging
        self.log_file = self.exp_dir / 'training.log'
        self.history = self.checkpoint.get('history', {
            'train_loss': [], 'train_mag_acc': [], 'train_az_acc': [],
            'val_loss': [], 'val_mag_acc': [], 'val_az_acc': [],
            'learning_rates': []
        })
        
        print(f">> Resuming experiment: {self.exp_dir}")
        self.log(f"Resuming training from epoch {self.checkpoint['epoch'] + 2}...")
        self.log(f"Device: {self.device}")

    def resume(self):
        """Resume training loop"""
        self.log("=" * 80)
        self.log(f"RESUMING CONVNEXT TRAINING FROM EPOCH {self.checkpoint['epoch'] + 2}")
        self.log("=" * 80)
        
        # Setup model and optimizer
        model, criterion = self.setup_model()
        model.load_state_dict(self.checkpoint['model_state_dict'])
        
        optimizer, scheduler = self.setup_optimizer(model)
        optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
        
        # Setup AMP and EMA
        scaler = GradScaler()
        # Note: Scaler state dict might be missing in older checkpoints, but good to have
        if 'scaler_state_dict' in self.checkpoint:
            scaler.load_state_dict(self.checkpoint['scaler_state_dict'])
            
        ema_model = EMAModel(model, decay=self.config.get('ema_decay', 0.9999))
        # Note: EMA state isn't saved in the checkpoint in original script (simple version)
        # We'll just restart EMA or if it was part of model state (it wasn't, it wraps it)
        # The original script didn't save EMA state explicitly, which is a minor issue 
        # but acceptable for resuming. It will re-accumulate.
        
        # Progressive resizing setup
        resize_schedule = self.config.get('resize_schedule', {0: 224})
        resizer = ProgressiveResizer(resize_schedule)
        
        # Determine start parameters
        start_epoch = self.checkpoint['epoch'] + 1
        total_epochs = self.config['epochs']
        
        # Force initial size check for the starting epoch
        current_size = resizer.get_size(start_epoch)
        self.log(f"Starting with image size: {current_size}")
        
        # Setup data loaders for current size
        train_loader, val_loader, test_loader = self.setup_data(current_size)
        
        # Validation metric tracking
        if len(self.history['val_mag_acc']) > 0:
            best_val_mag_acc = max(self.history['val_mag_acc'])
        else:
            best_val_mag_acc = 0.0
            
        start_time = time.time()
        
        for epoch in range(start_epoch, total_epochs):
            # Check for size change
            new_size = resizer.get_size(epoch)
            if new_size != current_size:
                self.log(f"[RESIZE] Changing image size: {current_size} -> {new_size}")
                current_size = new_size
                # Clean up old loaders to free memory
                del train_loader, val_loader, test_loader
                torch.cuda.empty_cache()
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
    parser = argparse.ArgumentParser(description='Resume ConvNeXt Training')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (checkpoint_latest.pth)')
    
    args = parser.parse_args()
    
    trainer = ResumeTrainer(args.checkpoint)
    trainer.resume()
    
    print("\n[SUCCESS] Resumed training completed!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        # traceback.print_exc() 
        sys.exit(1)
