"""
Auto-Effi Model Trainer (V3)
Advanced Hierarchical Multi-Task Training with EfficientNet-B0.
Designed for the Auto-Update System with No-Harm Principle.

Author: Antigravity AI
Date: 2026-02-15
"""

import os
import sys
import json
import yaml
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, recall_score, accuracy_score

# Setup Logging
logger = logging.getLogger("auto_effi.trainer")
logger.setLevel(logging.INFO)

# Console Handler
c_handler = logging.StreamHandler()
c_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(c_handler)

# File Handler configuration will be added in Trainer init to save to specific model folder

# ============================================================================
# DATASET HANDLING
# ============================================================================

class EffiHierarchicalDataset(Dataset):
    """Dataset for Hierarchical Training (Binary + Magnitude + Azimuth)"""
    
    def __init__(self, metadata_df: pd.DataFrame, dataset_root: str, transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        
        # Standard Class Definitions (Immutable for Production Safety)
        self.mag_classes = ['Normal', 'Moderate', 'Medium', 'Large']
        self.azi_classes = ['Normal', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        self.mag_to_idx = {c: i for i, c in enumerate(self.mag_classes)}
        self.azi_to_idx = {c: i for i, c in enumerate(self.azi_classes)}
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Helper to safely get string from row
        def get_val(key):
            if key not in row: return None
            val = row[key]
            if pd.isna(val): return None
            return str(val)

        # Handle file paths from different sources (Original vs Pending)
        filename = get_val('spectrogram_file') or get_val('filename')
        
        # Comprehensive search for the file
        img_path = None
        
        # Pre-construct search paths
        search_paths = []
        if filename:
            search_paths.append(self.dataset_root / filename)
            search_paths.append(self.dataset_root / 'spectrograms' / filename)
            search_paths.append(self.dataset_root / 'data' / 'validated' / filename)
            
        direct_path = get_val('spectrogram_path')
        if direct_path:
            search_paths.append(Path(direct_path))
        
        for p in search_paths:
            if p and os.path.exists(p):
                img_path = p
                break
                
        if img_path is None or not img_path.exists():
            # Fallback to current directory for relative paths in pending data
            if filename:
                img_path = Path(filename)
                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {filename}")
            else:
                raise FileNotFoundError(f"Image filename missing in metadata for index {idx}")

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback to avoid crashing during long training
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
            
        # Labels Construction
        mag_cls = str(row.get('magnitude_class', 'Normal'))
        azi_cls = str(row.get('azimuth_class', 'Normal'))
        
        # 1. Binary Label (Gatekeeper)
        is_precursor = 1 if mag_cls != 'Normal' else 0
        
        # 2. Magnitude Label (Energy)
        mag_label = self.mag_to_idx.get(mag_cls, 0)
        
        # 3. Azimuth Label (Direction)
        azi_label = self.azi_to_idx.get(azi_cls, 0)
        
        return image, is_precursor, mag_label, azi_label

# ============================================================================
# ARCHITECTURE (EfficientNet-B0 Hierarchical)
# ============================================================================

class HierarchicalEfficientNetV3(nn.Module):
    """
    SOTA EfficientNet-B0 with Hierarchical Multi-Head Architecture.
    Optimized for Indonesian BMKG Geomagnetic Data.
    """
    def __init__(self, num_mag_classes=4, num_azi_classes=9, pretrained=True):
        super().__init__()
        # Load SOTA Backbone
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Input features for heads
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity() # Remove default head
        
        # Optimization Neck
        self.neck = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3)
        )
        
        # Stage 1: Binary Detection Head
        self.binary_head = nn.Linear(512, 2)
        
        # Stage 2: Magnitude Head
        self.mag_head = nn.Linear(512, num_mag_classes)
        
        # Stage 3: Azimuth Head
        self.azi_head = nn.Linear(512, num_azi_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.neck(features)
        
        logits_bin = self.binary_head(embeddings)
        logits_mag = self.mag_head(embeddings)
        logits_azi = self.azi_head(embeddings)
        
        return logits_bin, logits_mag, logits_azi

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Keep BatchNorm running stats updating or frozen? 
        # Usually for fine-tuning we keep them frozen if dataset is small, but here we might want to update.
        # Let's keep them in train mode but params frozen.
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

# ============================================================================
# MAIN TRAINER LOGIC
# ============================================================================

class AutoEffiTrainer:
    def __init__(self, config_path: str = "config/pipeline_config.yaml", output_dir: str = None):
        # Load Configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.training_config = self.config['training']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if output_dir:
            self.challenger_dir = Path(output_dir)
        else:
            self.base_dir = Path(__file__).parent.parent
            self.challenger_dir = self.base_dir / self.config['paths']['challenger_model']
            
        self.challenger_dir.mkdir(parents=True, exist_ok=True)
        
        # Add File Handler for this specific training run
        # Remove existing file handlers to prevent duplicates if instantiated multiple times
        for h in logger.handlers[:]:
            if isinstance(h, logging.FileHandler):
                logger.removeHandler(h)
                
        f_handler = logging.FileHandler(self.challenger_dir / "training_log.txt")
        f_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(f_handler)
        
        logger.info(f"Trainer V3 initialized. Device: {self.device}")
        logger.info(f"Logging training progress to {self.challenger_dir / 'training_log.txt'}")

    def compute_class_weights(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Balanced weights to handle natural earthquake scarcity."""
        # Magnitude Weights
        mag_counts = df['magnitude_class'].value_counts()
        total = len(df)
        
        # Focus on 'Large' (priority for safety)
        weights = []
        for cls in ['Normal', 'Moderate', 'Medium', 'Large']:
            count = mag_counts.get(cls, 0)
            w = total / (4 * count + 1e-6)
            if cls == 'Large': w *= 1.5 # Extra boost for safety
            weights.append(w)
        
        # Binary Weights (Normal vs Precursor)
        n_normal = mag_counts.get('Normal', 0)
        n_prec = total - n_normal
        bin_w = [total/(2*n_normal + 1e-6), total/(2*n_prec + 1e-6)]
        
        return torch.tensor(bin_w).float().to(self.device), torch.tensor(weights).float().to(self.device)

    def train(self, metadata_path: str, dataset_root: str, new_samples_count: int = 0, train_df=None, val_df=None):
        logger.info(f"Loading data from {metadata_path}...")
        
        if train_df is not None and val_df is not None:
            logger.info(f"Using provided data splits. Train: {len(train_df)}, Val: {len(val_df)}")
            df = pd.concat([train_df, val_df]) # Combine for class weights calc if needed
        else:
            df = pd.read_csv(metadata_path)
            # Split Data (80/20)
            train_df = df.sample(frac=0.8, random_state=42)
            val_df = df.drop(train_df.index)
            
        total_samples = len(df)
        
        # Transforms (Physics-Aware Augmentation)
        # ... (transforms code is fine, omitted for brevity if not changing) ...
        # (Wait, I need to keep the code I'm replacing or the tool will cut it out if I don't include it in ReplacementContent? 
        # Yes, replace_file_content replaces the block. I should include the transforms setup to be safe or target a smaller block.)
        
        # Let's target the Validatiom/Dataset setup which is lines 226-246 roughly.
        # Actually, let's just replace the beginning of the function and the logic block.
        
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Loaders
        train_ds = EffiHierarchicalDataset(train_df, dataset_root, train_tf)
        val_ds = EffiHierarchicalDataset(val_df, dataset_root, val_tf)
        
        train_loader = DataLoader(train_ds, batch_size=self.training_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.training_config['batch_size'], shuffle=False)
        
        # Weights & Criteria
        bin_weights, mag_weights = self.compute_class_weights(train_df)
        criterion_bin = nn.CrossEntropyLoss(weight=bin_weights)
        criterion_mag = nn.CrossEntropyLoss(weight=mag_weights)
        criterion_azi = nn.CrossEntropyLoss() # Balanced enough or less priority
        
        # Initialize Model
        model = HierarchicalEfficientNetV3().to(self.device)
        
        # Training Parameters
        epochs = int(self.training_config['epochs'])
        early_stop = int(self.training_config['early_stopping_patience'])
        best_composite = 0.0
        patience_counter = 0
        
        # DETERMINE TRAINING STRATEGY BASED ON *NEW* DATA VOLUME
        # We use the explicitly passed new_samples_count
        is_partial_mode = new_samples_count > 0 and new_samples_count < 50
        
        # Initialize Hierarchy Weights
        hw = self.training_config['hierarchy_weights'].copy()
        
        if is_partial_mode:
            logger.info(f"!!! PARTIAL UPDATE MODE ACTIVATED (New Samples: {new_samples_count} < 50) !!!")
            logger.info("Strategy: Freeze Backbone + Binary Head. Train ONLY Mag & Azi Heads.")
            
            # Freeze Backbone & Binary Head explicitly
            model.freeze_backbone()
            for param in model.binary_head.parameters():
                param.requires_grad = False
                
            # Optimizer only for Mag & Azi heads
            optimizer = optim.AdamW([
                {'params': model.mag_head.parameters(), 'lr': float(self.training_config['learning_rate'])},
                {'params': model.azi_head.parameters(), 'lr': float(self.training_config['learning_rate'])}
            ], weight_decay=float(self.training_config['weight_decay']))
            
            # Loss Weights Modification for Partial Mode
            hw['binary'] = 0.0 # Ignore binary loss
            hw['magnitude'] = 1.2
            hw['azimuth'] = 0.8
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
            
            # No Phase 2 for Partial Mode
            freeze_epochs = epochs + 1 # Never unfreeze
            
        else:
            logger.info(f"NORMAL FINE-TUNING MODE (Samples: {total_samples} >= 50)")
            
            # TRAINING PHASE 1: FROZEN BACKBONE (Heads Only)
            logger.info(">>> PHASE 1: Training Heads Only (Frozen Backbone)")
            model.freeze_backbone()
            
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                   lr=float(self.training_config['learning_rate']), 
                                   weight_decay=float(self.training_config['weight_decay']))
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
            
            # Phase 2 Trigger
            freeze_epochs = 5
        
        for epoch in range(epochs):
            # CHECK PHASE TRANSITION
            if epoch == freeze_epochs:
                logger.info(">>> PHASE 2: Unfreezing Backbone (Fine-Tuning)")
                model.unfreeze_backbone()
                # Re-initialize optimizer with lower LR for backbone
                optimizer = optim.AdamW([
                    {'params': model.backbone.parameters(), 'lr': float(self.training_config['learning_rate']) * 0.1},
                    {'params': model.neck.parameters(), 'lr': float(self.training_config['learning_rate'])},
                    {'params': model.binary_head.parameters(), 'lr': float(self.training_config['learning_rate'])},
                    {'params': model.mag_head.parameters(), 'lr': float(self.training_config['learning_rate'])},
                    {'params': model.azi_head.parameters(), 'lr': float(self.training_config['learning_rate'])}
                ], weight_decay=float(self.training_config['weight_decay']))
                
            model.train()
            total_loss = 0
            
            for imgs, labels_bin, labels_mag, labels_azi in train_loader:
                imgs = imgs.to(self.device)
                labels_bin, labels_mag, labels_azi = labels_bin.to(self.device), labels_mag.to(self.device), labels_azi.to(self.device)
                
                optimizer.zero_grad()
                out_bin, out_mag, out_azi = model(imgs)
                
                loss_bin = criterion_bin(out_bin, labels_bin)
                loss_mag = criterion_mag(out_mag, labels_mag)
                loss_azi = criterion_azi(out_azi, labels_azi)
                
                # Weighted Hierarchical Loss
                loss = (hw['binary'] * loss_bin) + (hw['magnitude'] * loss_mag) + (hw['azimuth'] * loss_azi)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            val_preds_mag = []
            val_targets_mag = []
            val_preds_azi = []
            val_targets_azi = []
            
            with torch.no_grad():
                for imgs, _, l_mag, l_azi in val_loader:
                    imgs = imgs.to(self.device)
                    _, out_mag, out_azi = model(imgs)
                    val_preds_mag.extend(out_mag.argmax(1).cpu().numpy())
                    val_targets_mag.extend(l_mag.cpu().numpy())
                    val_preds_azi.extend(out_azi.argmax(1).cpu().numpy())
                    val_targets_azi.extend(l_azi.cpu().numpy())
                    
            # Metrics Calculation
            acc_mag = accuracy_score(val_targets_mag, val_preds_mag)
            acc_azi = accuracy_score(val_targets_azi, val_preds_azi)
            
            # Recall Large (Critical for No-Harm Principle)
            # Index 3 is 'Large'
            recall_large = recall_score(val_targets_mag, val_preds_mag, labels=[3], average='macro', zero_division=0)
            
            # Composite Score for Early Stopping
            composite = (0.6 * acc_mag) + (0.4 * acc_azi)
            
            phase_str = "FROZEN" if epoch < freeze_epochs else "FINE-TUNE"
            logger.info(f"Epoch {epoch+1}/{epochs} [{phase_str}] | Loss: {total_loss/len(train_loader):.4f} | Mag Acc: {acc_mag:.4f} | Azi Acc: {acc_azi:.4f} | Large Recall: {recall_large:.4f}")
            
            scheduler.step(composite)
            
            # Save Best Challenger
            if composite > best_composite:
                best_composite = composite
                patience_counter = 0
                self._save_checkpoint(model, acc_mag, acc_azi, recall_large, epoch)
                logger.info(f"  --> [SAVED] Challenger Improved. Score: {composite:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stop:
                    logger.info("Early stopping triggered.")
        logger.info(f"Training Complete. Best Composite Score: {best_composite:.4f}")
        return self.challenger_dir / 'best_model.pth'

    def evaluate_on_test_set(self, test_csv_path: str, dataset_root: str) -> dict:
        """
        Evaluate the BEST SAVED MODEL on a held-out test set (Golden Set).
        Returns the definitive metrics dictionary.
        """
        logger.info(f"🧪 Evaluating BEST MODEL on Golden Set: {test_csv_path}")
        
        # Load Best Model
        best_model_path = self.challenger_dir / "best_model.pth"
        if not best_model_path.exists():
            logger.error(f"Best model not found at {best_model_path}")
            return {}
            
        model = HierarchicalEfficientNetV3().to(self.device)
        checkpoint = torch.load(best_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load Test Data
        df = pd.read_csv(test_csv_path)
        test_ds = EffiHierarchicalDataset(df, dataset_root, self.val_tf) # Use validation transform (no aug)
        test_loader = DataLoader(test_ds, batch_size=self.training_config['batch_size'], shuffle=False)
        
        # Run Inference
        all_preds_mag = []
        all_labels_mag = []
        all_preds_azi = []
        all_labels_azi = []
        
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch['image'].to(self.device)
                labels_bin = batch['label_bin'].to(self.device)
                labels_mag = batch['label_mag'].to(self.device)
                labels_azi = batch['label_azi'].to(self.device)
                
                out_bin, out_mag, out_azi = model(imgs)
                
                # Predictions
                _, preds_mag = torch.max(out_mag, 1)
                _, preds_azi = torch.max(out_azi, 1)
                
                all_preds_mag.extend(preds_mag.cpu().numpy())
                all_labels_mag.extend(labels_mag.cpu().numpy())
                all_preds_azi.extend(preds_azi.cpu().numpy())
                all_labels_azi.extend(labels_azi.cpu().numpy())
                
        # Compute Metrics
        from sklearn.metrics import accuracy_score, recall_score
        import numpy as np
        
        mag_acc = accuracy_score(all_labels_mag, all_preds_mag)
        azi_acc = accuracy_score(all_labels_azi, all_preds_azi)
        
        # Recall Large (Class 3)
        # Note: Need to handle case where no Large samples exist in batch, but Golden Set has them.
        large_indices = [i for i, x in enumerate(all_labels_mag) if x == 3]
        if large_indices:
            large_preds = [all_preds_mag[i] for i in large_indices]
            large_recall = sum([1 for p in large_preds if p == 3]) / len(large_indices)
        else:
            large_recall = 0.0 # Or 1.0? 0.0 is safer (fail if tested on wrong set)
            
        # Composite Score
        w_mag = self.config['evaluation']['weights']['magnitude_accuracy']
        w_rec = self.config['evaluation']['weights']['magnitude_recall_large']
        w_azi = self.config['evaluation']['weights']['azimuth_accuracy']
        
        score = (w_mag * mag_acc) + (w_rec * large_recall) + (w_azi * azi_acc)
        
        metrics = {
            "magnitude_accuracy": float(mag_acc * 100),
            "azimuth_accuracy": float(azi_acc * 100),
            "magnitude_recall_large": float(large_recall * 100),
            "composite_score": float(score * 100) # Scale to 0-100
        }
        
        logger.info(f"📊 Golden Set Results: Score={metrics['composite_score']:.2f} | Recall(L)={metrics['magnitude_recall_large']:.2f}% | MagAcc={metrics['magnitude_accuracy']:.2f}%")
        
        # Update Metadata with Golden Metrics
        meta_path = self.challenger_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            meta['metrics'] = metrics
            meta['validation_type'] = "GOLDEN_SET" # Mark this
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
                
        return metrics

    def _save_checkpoint(self, model, mag_acc, azi_acc, recall_large, epoch):
        """Save model and metadata for comparison."""
        save_path = self.challenger_dir / 'best_model.pth'
        torch.save(model.state_dict(), save_path)
        
        metadata = {
            "model_id": f"effi_challenger_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "architecture": "efficientnet_b0_hierarchical",
            "trained_at": datetime.now().isoformat(),
            "epoch": epoch,
            "metrics": {
                "magnitude_accuracy": float(mag_acc * 100),
                "azimuth_accuracy": float(azi_acc * 100),
                "magnitude_recall_large": float(recall_large * 100),
                "composite_score": float(((0.6 * mag_acc) + (0.4 * azi_acc)) * 100)
            }
        }
        
        with open(self.challenger_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Also copy class mappings for deployment safety
        mappings = {
            "magnitude": ["Normal", "Moderate", "Medium", "Large"],
            "azimuth": ["Normal", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        }
        with open(self.challenger_dir / 'class_mappings.json', 'w') as f:
            json.dump(mappings, f, indent=2)

if __name__ == "__main__":
    # Test script if run directly
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="../dataset_unified/metadata/unified_metadata.csv")
    parser.add_argument("--root", type=str, default="../dataset_unified/spectrograms")
    args = parser.parse_args()
    
    if os.path.exists(args.meta):
        trainer = AutoEffiTrainer()
        trainer.train(args.meta, args.root)
    else:
        print(f"Metadata not found at {args.meta}. Please provide correct --meta path.")
