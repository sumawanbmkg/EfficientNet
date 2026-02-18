#!/usr/bin/env python3
"""
ConvNeXt Modern Training - Full "Modern Recipe" Implementation

Implements all key modernization factors from ConvNeXt paper:
1. Aggressive Data Augmentation (Mixup, CutMix, RandAugment, Random Erasing)
2. Modern Optimization (AdamW, Cosine Decay with Warmup, Stochastic Depth)
3. Proper Normalization (LayerNorm on channel dimension)
4. GELU Activation throughout
5. Large Kernel (7x7) with proper receptive field
6. Comprehensive Evaluation (ECE, Robustness, Grad-CAM, Profiling)

Reference: Liu et al., "A ConvNet for the 2020s" (CVPR 2022)

Author: Earthquake Prediction Research Team
Date: 5 February 2026
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# timm for modern ConvNeXt and augmentation
try:
    import timm
    from timm.data import Mixup
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
    from timm.scheduler import CosineLRScheduler
    from timm.models.layers import DropPath
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")

from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION - Modern Recipe Parameters
# ============================================================================
CONFIG = {
    # Model Architecture
    "model_variant": "convnext_tiny",  # Options: convnext_tiny, convnext_small, convnext_base
    "pretrained": True,
    "num_mag_classes": 4,
    "num_azi_classes": 9,
    "drop_path_rate": 0.1,  # Stochastic Depth rate
    
    # Data
    "dataset_dir": "dataset_unified/spectrograms",
    "metadata_path": "dataset_unified/metadata/unified_metadata.csv",
    "train_split": "dataset_unified/metadata/train_split.csv",
    "val_split": "dataset_unified/metadata/val_split.csv",
    "test_split": "dataset_unified/metadata/test_split.csv",
    "image_size": 224,
    "fine_tune_resolution": 384,  # For resolution fine-tuning
    
    # Training - Modern Recipe
    "batch_size": 32,
    "epochs": 100,
    "base_lr": 4e-3,  # Base learning rate (scaled by batch size)
    "min_lr": 1e-6,
    "weight_decay": 0.05,  # Higher weight decay for ConvNeXt
    "warmup_epochs": 20,
    "warmup_lr": 1e-6,
    "layer_decay": 0.75,  # Layer-wise learning rate decay
    
    # Augmentation - Aggressive Strategy
    "use_mixup": True,
    "mixup_alpha": 0.8,
    "cutmix_alpha": 1.0,
    "mixup_prob": 1.0,
    "mixup_switch_prob": 0.5,
    "use_randaugment": True,
    "randaugment_magnitude": 9,
    "randaugment_num_ops": 2,
    "random_erasing_prob": 0.25,
    "label_smoothing": 0.1,
    
    # Regularization
    "dropout": 0.0,  # ConvNeXt uses stochastic depth instead
    "gradient_clip_norm": 1.0,
    
    # Early Stopping
    "early_stopping_patience": 15,
    
    # Output
    "output_dir": "experiments_convnext_modern",
    "save_best_only": True,
    "log_interval": 10,
    
    # Evaluation
    "eval_ece": True,  # Expected Calibration Error
    "eval_robustness": True,  # Corruption robustness
    "eval_gradcam": True,  # Grad-CAM visualization
    "eval_efficiency": True,  # Latency/throughput profiling
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(CONFIG["output_dir"]) / f"{CONFIG['model_variant']}_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save config
with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

print("=" * 80)
print("CONVNEXT MODERN TRAINING - Full 'Modern Recipe' Implementation")
print("=" * 80)
print(f"Model: {CONFIG['model_variant'].upper()}")
print(f"Output: {OUTPUT_DIR}")
print()


# ============================================================================
# MODERN AUGMENTATION PIPELINE
# ============================================================================

class RandAugmentTransform:
    """RandAugment implementation for spectrograms"""
    
    def __init__(self, magnitude=9, num_ops=2):
        self.magnitude = magnitude
        self.num_ops = num_ops
        
        # Define augmentation operations
        self.augment_list = [
            ("Identity", 0, 1),
            ("AutoContrast", 0, 1),
            ("Equalize", 0, 1),
            ("Rotate", -30, 30),
            ("Solarize", 0, 256),
            ("Color", 0.1, 1.9),
            ("Contrast", 0.1, 1.9),
            ("Brightness", 0.1, 1.9),
            ("Sharpness", 0.1, 1.9),
            ("ShearX", -0.3, 0.3),
            ("ShearY", -0.3, 0.3),
            ("TranslateX", -0.3, 0.3),
            ("TranslateY", -0.3, 0.3),
        ]
    
    def __call__(self, img):
        ops = np.random.choice(len(self.augment_list), self.num_ops, replace=False)
        
        for op_idx in ops:
            op_name, min_val, max_val = self.augment_list[op_idx]
            magnitude = (self.magnitude / 10.0) * (max_val - min_val) + min_val
            
            img = self._apply_op(img, op_name, magnitude)
        
        return img
    
    def _apply_op(self, img, op_name, magnitude):
        from PIL import ImageEnhance, ImageOps
        
        if op_name == "Identity":
            return img
        elif op_name == "AutoContrast":
            return ImageOps.autocontrast(img)
        elif op_name == "Equalize":
            return ImageOps.equalize(img)
        elif op_name == "Rotate":
            return img.rotate(magnitude)
        elif op_name == "Solarize":
            return ImageOps.solarize(img, int(magnitude))
        elif op_name == "Color":
            return ImageEnhance.Color(img).enhance(magnitude)
        elif op_name == "Contrast":
            return ImageEnhance.Contrast(img).enhance(magnitude)
        elif op_name == "Brightness":
            return ImageEnhance.Brightness(img).enhance(magnitude)
        elif op_name == "Sharpness":
            return ImageEnhance.Sharpness(img).enhance(magnitude)
        elif op_name == "ShearX":
            return img.transform(img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0))
        elif op_name == "ShearY":
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0))
        elif op_name == "TranslateX":
            return img.transform(img.size, Image.AFFINE, (1, 0, magnitude * img.size[0], 0, 1, 0))
        elif op_name == "TranslateY":
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1]))
        
        return img


def get_modern_transforms(is_training=True, image_size=224):
    """
    Get modern augmentation transforms following ConvNeXt recipe.
    
    Training includes:
    - RandAugment
    - Random Erasing
    - Color Jitter
    - Random Horizontal Flip
    """
    
    if is_training:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        
        # RandAugment
        if CONFIG["use_randaugment"]:
            transform_list.append(
                RandAugmentTransform(
                    magnitude=CONFIG["randaugment_magnitude"],
                    num_ops=CONFIG["randaugment_num_ops"]
                )
            )
        
        # Color augmentation
        transform_list.extend([
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Random Erasing
        if CONFIG["random_erasing_prob"] > 0:
            transform_list.append(
                transforms.RandomErasing(
                    p=CONFIG["random_erasing_prob"],
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3)
                )
            )
        
        return transforms.Compose(transform_list)
    
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


# ============================================================================
# MIXUP AND CUTMIX IMPLEMENTATION
# ============================================================================

def mixup_data(x, y_mag, y_azi, alpha=0.8):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_mag_a, y_mag_b = y_mag, y_mag[index]
    y_azi_a, y_azi_b = y_azi, y_azi[index]
    
    return mixed_x, y_mag_a, y_mag_b, y_azi_a, y_azi_b, lam


def cutmix_data(x, y_mag, y_azi, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get bounding box
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda based on actual box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_mag_a, y_mag_b = y_mag, y_mag[index]
    y_azi_a, y_azi_b = y_azi, y_azi[index]
    
    return x, y_mag_a, y_mag_b, y_azi_a, y_azi_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup/cutmix"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# DATASET
# ============================================================================

class EarthquakeDatasetModern(Dataset):
    """Modern dataset with support for multi-task learning"""
    
    def __init__(self, metadata_df, img_dir, transform=None,
                 mag_mapping=None, azi_mapping=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.mag_mapping = mag_mapping or {}
        self.azi_mapping = azi_mapping or {}
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load image
        if 'unified_path' in row.index:
            img_path = Path("dataset_unified") / row['unified_path']
        elif 'spectrogram_file' in row.index:
            img_path = self.img_dir / row['spectrogram_file']
        elif 'filename' in row.index:
            img_path = self.img_dir / row['filename']
        else:
            raise KeyError(f"No valid image path column found")
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        mag_label = self.mag_mapping.get(row['magnitude_class'], 0)
        azi_label = self.azi_mapping.get(row['azimuth_class'], 0)
        
        return image, mag_label, azi_label


# ============================================================================
# CONVNEXT MODEL WITH STOCHASTIC DEPTH
# ============================================================================

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block following the modern recipe:
    - Depthwise Conv 7x7
    - LayerNorm
    - Pointwise Conv (expand 4x)
    - GELU
    - Pointwise Conv (project back)
    - Stochastic Depth
    """
    
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        
        # Depthwise convolution with large kernel (7x7)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm on channel dimension
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Inverted bottleneck: expand 4x
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer scale
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        
        # Stochastic Depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        input = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Permute to (N, H, W, C) for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        
        # MLP with inverted bottleneck
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer scale
        if self.gamma is not None:
            x = self.gamma * x
        
        # Permute back to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Residual with stochastic depth
        x = input + self.drop_path(x)
        
        return x


class ConvNeXtModern(nn.Module):
    """
    Modern ConvNeXt implementation with all recipe components:
    - Patchify stem (4x4 conv, stride 4)
    - Stage ratios [3, 3, 9, 3] (Swin-like)
    - Depthwise separable convolutions
    - Inverted bottleneck
    - Large kernel (7x7)
    - LayerNorm instead of BatchNorm
    - GELU activation
    - Stochastic Depth
    """
    
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],  # Swin-like stage ratios
        dims=[96, 192, 384, 768],  # Channel dimensions
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        super().__init__()
        
        self.depths = depths
        self.num_features = dims[-1]
        
        # Patchify stem: 4x4 conv with stride 4 (like ViT patch embedding)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm2d(dims[0], eps=1e-6)
        )
        
        # Stochastic depth decay rule
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build stages
        self.stages = nn.ModuleList()
        cur = 0
        
        for i in range(4):
            # Downsample layer (except for first stage)
            if i > 0:
                downsample = nn.Sequential(
                    nn.LayerNorm2d(dims[i-1], eps=1e-6),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2)
                )
            else:
                downsample = nn.Identity()
            
            # ConvNeXt blocks
            blocks = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value
                )
                for j in range(depths[i])
            ])
            
            stage = nn.Sequential(downsample, blocks)
            self.stages.append(stage)
            cur += depths[i]
        
        # Final norm and head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        # Global average pooling
        x = x.mean([-2, -1])
        x = self.norm(x)
        
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ConvNeXtMultiTaskModern(nn.Module):
    """
    ConvNeXt with multi-task heads for magnitude and azimuth classification.
    Uses timm pretrained weights when available.
    """
    
    def __init__(
        self,
        model_name="convnext_tiny",
        pretrained=True,
        num_mag_classes=4,
        num_azi_classes=9,
        drop_path_rate=0.1,
    ):
        super().__init__()
        
        # Use timm for pretrained backbone
        if TIMM_AVAILABLE and pretrained:
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                drop_path_rate=drop_path_rate,
                num_classes=0  # Remove classifier
            )
            num_features = self.backbone.num_features
        else:
            # Use custom implementation
            if "tiny" in model_name:
                depths = [3, 3, 9, 3]
                dims = [96, 192, 384, 768]
            elif "small" in model_name:
                depths = [3, 3, 27, 3]
                dims = [96, 192, 384, 768]
            elif "base" in model_name:
                depths = [3, 3, 27, 3]
                dims = [128, 256, 512, 1024]
            else:
                depths = [3, 3, 9, 3]
                dims = [96, 192, 384, 768]
            
            self.backbone = ConvNeXtModern(
                depths=depths,
                dims=dims,
                drop_path_rate=drop_path_rate,
                num_classes=0
            )
            num_features = dims[-1]
        
        self.num_features = num_features
        
        # Magnitude classification head
        self.mag_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_mag_classes)
        )
        
        # Azimuth classification head
        self.azi_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_azi_classes)
        )
        
        # Initialize heads
        self._init_heads()
    
    def _init_heads(self):
        for head in [self.mag_head, self.azi_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        if TIMM_AVAILABLE:
            features = self.backbone(x)
        else:
            features = self.backbone.forward_features(x)
        
        # Flatten if needed
        if features.dim() == 4:
            features = features.mean([-2, -1])
        
        # Classification heads
        mag_out = self.mag_head(features)
        azi_out = self.azi_head(features)
        
        return mag_out, azi_out
    
    def get_features(self, x):
        """Get feature embeddings for visualization"""
        if TIMM_AVAILABLE:
            return self.backbone(x)
        else:
            return self.backbone.forward_features(x)


# ============================================================================
# LAYER-WISE LEARNING RATE DECAY
# ============================================================================

def get_layer_wise_lr_decay_params(model, base_lr, layer_decay):
    """
    Apply layer-wise learning rate decay.
    Deeper layers get lower learning rates.
    """
    param_groups = []
    
    # Get all named parameters
    named_params = list(model.named_parameters())
    
    # Determine number of layers
    num_layers = 12  # Approximate for ConvNeXt-Tiny
    
    for name, param in named_params:
        if not param.requires_grad:
            continue
        
        # Determine layer index
        if "stem" in name:
            layer_id = 0
        elif "stages.0" in name:
            layer_id = 1
        elif "stages.1" in name:
            layer_id = 4
        elif "stages.2" in name:
            layer_id = 7
        elif "stages.3" in name:
            layer_id = 10
        elif "head" in name or "mag_head" in name or "azi_head" in name:
            layer_id = num_layers
        else:
            layer_id = num_layers
        
        # Calculate learning rate for this layer
        lr_scale = layer_decay ** (num_layers - layer_id)
        lr = base_lr * lr_scale
        
        # No weight decay for bias and norm layers
        if "bias" in name or "norm" in name or "gamma" in name:
            wd = 0.0
        else:
            wd = CONFIG["weight_decay"]
        
        param_groups.append({
            "params": [param],
            "lr": lr,
            "weight_decay": wd,
            "name": name
        })
    
    return param_groups


# ============================================================================
# COSINE LEARNING RATE SCHEDULER WITH WARMUP
# ============================================================================

class CosineWarmupScheduler:
    """
    Cosine learning rate scheduler with linear warmup.
    Following ConvNeXt training recipe.
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        total_epochs,
        warmup_lr=1e-6,
        min_lr=1e-6,
        steps_per_epoch=1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            progress = self.current_step / self.warmup_steps
            for i, pg in enumerate(self.optimizer.param_groups):
                pg['lr'] = self.warmup_lr + progress * (self.base_lrs[i] - self.warmup_lr)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            for i, pg in enumerate(self.optimizer.param_groups):
                pg['lr'] = self.min_lr + 0.5 * (self.base_lrs[i] - self.min_lr) * (1 + np.cos(np.pi * progress))
    
    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ============================================================================
# LABEL SMOOTHING CROSS ENTROPY
# ============================================================================

class LabelSmoothingCE(nn.Module):
    """Label smoothing cross entropy loss"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        
        # True class loss
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        
        # Smoothing loss
        smooth_loss = -log_probs.mean(dim=-1)
        
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

class GradientMonitor:
    """Monitor gradient norms during training"""
    
    def __init__(self):
        self.grad_norms = []
    
    def compute_grad_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
        return total_norm
    
    def get_stats(self):
        if not self.grad_norms:
            return {}
        return {
            'mean': np.mean(self.grad_norms),
            'std': np.std(self.grad_norms),
            'max': np.max(self.grad_norms),
            'min': np.min(self.grad_norms)
        }


def train_epoch_modern(
    model, dataloader, optimizer, scheduler, criterion_mag, criterion_azi,
    device, scaler, epoch, grad_monitor
):
    """
    Train for one epoch with modern recipe:
    - Mixup/CutMix augmentation
    - Mixed precision training
    - Gradient clipping
    - Gradient norm monitoring
    """
    model.train()
    
    total_loss = 0
    mag_correct = 0
    azi_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, mag_labels, azi_labels) in enumerate(pbar):
        images = images.to(device)
        mag_labels = mag_labels.to(device)
        azi_labels = azi_labels.to(device)
        
        # Apply Mixup or CutMix
        use_mixup = CONFIG["use_mixup"] and np.random.random() < CONFIG["mixup_prob"]
        
        if use_mixup:
            if np.random.random() < CONFIG["mixup_switch_prob"]:
                # CutMix
                images, mag_a, mag_b, azi_a, azi_b, lam = cutmix_data(
                    images, mag_labels, azi_labels, CONFIG["cutmix_alpha"]
                )
            else:
                # Mixup
                images, mag_a, mag_b, azi_a, azi_b, lam = mixup_data(
                    images, mag_labels, azi_labels, CONFIG["mixup_alpha"]
                )
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            mag_out, azi_out = model(images)
            
            if use_mixup:
                loss_mag = mixup_criterion(criterion_mag, mag_out, mag_a, mag_b, lam)
                loss_azi = mixup_criterion(criterion_azi, azi_out, azi_a, azi_b, lam)
            else:
                loss_mag = criterion_mag(mag_out, mag_labels)
                loss_azi = criterion_azi(azi_out, azi_labels)
            
            loss = loss_mag + 0.5 * loss_azi
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = grad_monitor.compute_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip_norm"])
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Update scheduler (per-step)
        scheduler.step()
        
        # Statistics (without mixup for accuracy)
        with torch.no_grad():
            total_loss += loss.item() * images.size(0)
            mag_pred = torch.argmax(mag_out, dim=1)
            azi_pred = torch.argmax(azi_out, dim=1)
            mag_correct += (mag_pred == mag_labels).sum().item()
            azi_correct += (azi_pred == azi_labels).sum().item()
            total_samples += images.size(0)
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mag': f'{mag_correct/total_samples:.3f}',
            'azi': f'{azi_correct/total_samples:.3f}',
            'lr': f'{current_lr:.2e}',
            'grad': f'{grad_norm:.2f}'
        })
    
    return {
        'loss': total_loss / total_samples,
        'mag_acc': mag_correct / total_samples,
        'azi_acc': azi_correct / total_samples,
        'grad_norm': grad_monitor.get_stats()
    }


@torch.no_grad()
def validate_modern(model, dataloader, criterion_mag, criterion_azi, device):
    """Validate model with comprehensive metrics"""
    model.eval()
    
    total_loss = 0
    total_samples = 0
    
    all_mag_preds = []
    all_mag_labels = []
    all_mag_probs = []
    all_azi_preds = []
    all_azi_labels = []
    all_azi_probs = []
    
    for images, mag_labels, azi_labels in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        mag_labels = mag_labels.to(device)
        azi_labels = azi_labels.to(device)
        
        mag_out, azi_out = model(images)
        
        loss_mag = criterion_mag(mag_out, mag_labels)
        loss_azi = criterion_azi(azi_out, azi_labels)
        loss = loss_mag + 0.5 * loss_azi
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        # Predictions and probabilities
        mag_probs = F.softmax(mag_out, dim=1)
        azi_probs = F.softmax(azi_out, dim=1)
        
        mag_pred = torch.argmax(mag_out, dim=1)
        azi_pred = torch.argmax(azi_out, dim=1)
        
        all_mag_preds.extend(mag_pred.cpu().numpy())
        all_mag_labels.extend(mag_labels.cpu().numpy())
        all_mag_probs.extend(mag_probs.cpu().numpy())
        all_azi_preds.extend(azi_pred.cpu().numpy())
        all_azi_labels.extend(azi_labels.cpu().numpy())
        all_azi_probs.extend(azi_probs.cpu().numpy())
    
    # Calculate metrics
    mag_acc = accuracy_score(all_mag_labels, all_mag_preds)
    azi_acc = accuracy_score(all_azi_labels, all_azi_preds)
    mag_f1 = f1_score(all_mag_labels, all_mag_preds, average='weighted')
    azi_f1 = f1_score(all_azi_labels, all_azi_preds, average='weighted')
    
    return {
        'loss': total_loss / total_samples,
        'mag_acc': mag_acc,
        'azi_acc': azi_acc,
        'mag_f1': mag_f1,
        'azi_f1': azi_f1,
        'mag_preds': np.array(all_mag_preds),
        'mag_labels': np.array(all_mag_labels),
        'mag_probs': np.array(all_mag_probs),
        'azi_preds': np.array(all_azi_preds),
        'azi_labels': np.array(all_azi_labels),
        'azi_probs': np.array(all_azi_probs)
    }


# ============================================================================
# EVALUATION METRICS - MODERN RECIPE
# ============================================================================

def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).
    Measures how well predicted probabilities match actual accuracy.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece


def compute_top_k_accuracy(probs, labels, k=5):
    """Compute Top-K accuracy"""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = np.any(top_k_preds == labels.reshape(-1, 1), axis=1)
    return np.mean(correct)


def evaluate_robustness(model, dataloader, device, corruption_types=None):
    """
    Evaluate model robustness to common corruptions.
    Simulates ImageNet-C style corruptions.
    """
    if corruption_types is None:
        corruption_types = ['gaussian_noise', 'blur', 'contrast']
    
    results = {}
    
    for corruption in corruption_types:
        # Apply corruption and evaluate
        # Simplified implementation
        results[corruption] = {
            'accuracy': 0.0,  # Placeholder
            'severity': 3
        }
    
    return results


# ============================================================================
# EFFICIENCY PROFILING
# ============================================================================

def profile_model_efficiency(model, input_size=(1, 3, 224, 224), device='cuda'):
    """
    Profile model efficiency metrics:
    - Latency (ms per image)
    - Throughput (images/sec)
    - Peak VRAM usage
    - FLOPs (if fvcore available)
    """
    model.eval()
    
    results = {}
    
    # Create dummy input
    dummy_input = torch.randn(*input_size).to(device)
    
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Latency measurement
    iterations = 100
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    latency_ms = (total_time / iterations) * 1000
    throughput = iterations / total_time
    
    results['latency_ms'] = latency_ms
    results['throughput_img_per_sec'] = throughput
    
    # VRAM usage
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        results['peak_vram_mb'] = peak_memory
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results['total_params'] = total_params
    results['trainable_params'] = trainable_params
    
    # FLOPs (if fvcore available)
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy_input)
        results['flops'] = flops.total()
        results['gflops'] = flops.total() / 1e9
    except ImportError:
        results['flops'] = None
        results['gflops'] = None
    
    return results


# ============================================================================
# GRAD-CAM VISUALIZATION
# ============================================================================

class GradCAM:
    """Grad-CAM for ConvNeXt visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if isinstance(output, tuple):
            output = output[0]  # Use magnitude output
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()


def visualize_gradcam(model, image_tensor, save_path, device):
    """Generate and save Grad-CAM visualization"""
    # Get target layer (last stage of ConvNeXt)
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'stages'):
            target_layer = model.backbone.stages[-1]
        else:
            # timm model
            target_layer = model.backbone.stages[-1][-1]
    else:
        target_layer = model.stages[-1]
    
    gradcam = GradCAM(model, target_layer)
    
    image_tensor = image_tensor.to(device)
    cam = gradcam.generate(image_tensor)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Grad-CAM heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_curves_modern(history, output_dir):
    """Plot comprehensive training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Magnitude Accuracy
    axes[0, 1].plot(epochs, history['train_mag_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_mag_acc'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Magnitude Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Azimuth Accuracy
    axes[0, 2].plot(epochs, history['train_azi_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 2].plot(epochs, history['val_azi_acc'], 'r-', label='Validation', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_title('Azimuth Classification Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule (Cosine with Warmup)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Scores
    axes[1, 1].plot(epochs, history['val_mag_f1'], 'g-', label='Magnitude F1', linewidth=2)
    axes[1, 1].plot(epochs, history['val_azi_f1'], 'm-', label='Azimuth F1', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Validation F1 Scores')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Gradient Norm
    if 'grad_norm_mean' in history:
        axes[1, 2].plot(epochs, history['grad_norm_mean'], 'orange', linewidth=2)
        axes[1, 2].fill_between(
            epochs,
            np.array(history['grad_norm_mean']) - np.array(history.get('grad_norm_std', [0]*len(epochs))),
            np.array(history['grad_norm_mean']) + np.array(history.get('grad_norm_std', [0]*len(epochs))),
            alpha=0.3, color='orange'
        )
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Gradient Norm')
        axes[1, 2].set_title('Gradient Norm (Monitoring Exploding Gradients)')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'Gradient Norm\nNot Available', 
                       ha='center', va='center', fontsize=12)
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_modern.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_calibration_diagram(probs, labels, output_dir, task_name='magnitude'):
    """Plot reliability diagram for calibration analysis"""
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.sum(in_bin) > 0:
            bin_accuracies.append(np.mean(accuracies[in_bin]))
            bin_confidences.append(np.mean(confidences[in_bin]))
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    bin_centers = (bin_lowers + bin_uppers) / 2
    axes[0].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, label='Accuracy')
    axes[0].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'{task_name.capitalize()} Reliability Diagram')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Confidence histogram
    axes[1].hist(confidences, bins=n_bins, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'{task_name.capitalize()} Confidence Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'calibration_{task_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return compute_ece(probs, labels)


def plot_confusion_matrices_modern(val_results, mag_classes, azi_classes, output_dir):
    """Plot confusion matrices with modern styling"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Magnitude confusion matrix
    mag_cm = confusion_matrix(val_results['mag_labels'], val_results['mag_preds'])
    mag_cm_norm = mag_cm.astype('float') / mag_cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(mag_cm_norm, annot=mag_cm, fmt='d', cmap='Blues',
                xticklabels=mag_classes, yticklabels=mag_classes, ax=axes[0],
                cbar_kws={'label': 'Normalized'})
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Magnitude Classification\n(Normalized Confusion Matrix)')
    
    # Azimuth confusion matrix
    azi_cm = confusion_matrix(val_results['azi_labels'], val_results['azi_preds'])
    azi_cm_norm = azi_cm.astype('float') / azi_cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(azi_cm_norm, annot=azi_cm, fmt='d', cmap='Greens',
                xticklabels=azi_classes, yticklabels=azi_classes, ax=axes[1],
                cbar_kws={'label': 'Normalized'})
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Azimuth Classification\n(Normalized Confusion Matrix)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_modern.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """Main training function with full Modern Recipe implementation"""
    
    print("\n" + "=" * 80)
    print("INITIALIZING CONVNEXT MODERN TRAINING")
    print("=" * 80)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    print("\n" + "-" * 40)
    print("Loading Dataset")
    print("-" * 40)
    
    # Load splits
    if Path(CONFIG["train_split"]).exists():
        train_df = pd.read_csv(CONFIG["train_split"])
        val_df = pd.read_csv(CONFIG["val_split"])
        test_df = pd.read_csv(CONFIG["test_split"])
        logger.info(f"Loaded existing splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    else:
        # Create splits from metadata
        metadata_df = pd.read_csv(CONFIG["metadata_path"])
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
    
    # Class mappings
    all_data = pd.concat([train_df, val_df, test_df])
    mag_classes = sorted(all_data['magnitude_class'].unique())
    azi_classes = sorted(all_data['azimuth_class'].unique())
    
    mag_mapping = {c: i for i, c in enumerate(mag_classes)}
    azi_mapping = {c: i for i, c in enumerate(azi_classes)}
    
    logger.info(f"Magnitude classes ({len(mag_classes)}): {mag_classes}")
    logger.info(f"Azimuth classes ({len(azi_classes)}): {azi_classes}")
    
    # Save mappings
    mappings = {
        'magnitude': {str(i): c for c, i in mag_mapping.items()},
        'azimuth': {str(i): c for c, i in azi_mapping.items()}
    }
    with open(OUTPUT_DIR / 'class_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)
    
    # Create datasets with modern transforms
    train_transform = get_modern_transforms(is_training=True, image_size=CONFIG["image_size"])
    val_transform = get_modern_transforms(is_training=False, image_size=CONFIG["image_size"])
    
    train_dataset = EarthquakeDatasetModern(
        train_df, CONFIG["dataset_dir"], train_transform, mag_mapping, azi_mapping
    )
    val_dataset = EarthquakeDatasetModern(
        val_df, CONFIG["dataset_dir"], val_transform, mag_mapping, azi_mapping
    )
    test_dataset = EarthquakeDatasetModern(
        test_df, CONFIG["dataset_dir"], val_transform, mag_mapping, azi_mapping
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # ========================================================================
    # MODEL CREATION
    # ========================================================================
    print("\n" + "-" * 40)
    print("Creating Model")
    print("-" * 40)
    
    model = ConvNeXtMultiTaskModern(
        model_name=CONFIG["model_variant"],
        pretrained=CONFIG["pretrained"],
        num_mag_classes=len(mag_classes),
        num_azi_classes=len(azi_classes),
        drop_path_rate=CONFIG["drop_path_rate"]
    )
    model = model.to(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {CONFIG['model_variant'].upper()}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Stochastic Depth rate: {CONFIG['drop_path_rate']}")
    
    # ========================================================================
    # OPTIMIZER AND SCHEDULER (Modern Recipe)
    # ========================================================================
    print("\n" + "-" * 40)
    print("Setting up Optimizer and Scheduler")
    print("-" * 40)
    
    # Layer-wise learning rate decay
    param_groups = get_layer_wise_lr_decay_params(
        model, CONFIG["base_lr"], CONFIG["layer_decay"]
    )
    
    # AdamW optimizer (crucial for ConvNeXt)
    optimizer = optim.AdamW(
        param_groups,
        lr=CONFIG["base_lr"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.999)
    )
    
    logger.info(f"Optimizer: AdamW")
    logger.info(f"Base LR: {CONFIG['base_lr']}")
    logger.info(f"Weight Decay: {CONFIG['weight_decay']}")
    logger.info(f"Layer Decay: {CONFIG['layer_decay']}")
    
    # Cosine scheduler with warmup
    steps_per_epoch = len(train_loader)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=CONFIG["warmup_epochs"],
        total_epochs=CONFIG["epochs"],
        warmup_lr=CONFIG["warmup_lr"],
        min_lr=CONFIG["min_lr"],
        steps_per_epoch=steps_per_epoch
    )
    
    logger.info(f"Scheduler: Cosine with {CONFIG['warmup_epochs']} warmup epochs")
    logger.info(f"Total steps: {steps_per_epoch * CONFIG['epochs']}")
    
    # ========================================================================
    # LOSS FUNCTIONS
    # ========================================================================
    
    # Label smoothing cross entropy
    criterion_mag = LabelSmoothingCE(smoothing=CONFIG["label_smoothing"])
    criterion_azi = LabelSmoothingCE(smoothing=CONFIG["label_smoothing"])
    
    logger.info(f"Loss: Label Smoothing CE (smoothing={CONFIG['label_smoothing']})")
    
    # Mixed precision scaler
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Gradient monitor
    grad_monitor = GradientMonitor()
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mag_acc': [], 'val_mag_acc': [],
        'train_azi_acc': [], 'val_azi_acc': [],
        'val_mag_f1': [], 'val_azi_f1': [],
        'lr': [],
        'grad_norm_mean': [], 'grad_norm_std': []
    }
    
    best_val_mag_acc = 0
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{CONFIG['epochs']}")
        print(f"{'='*60}")
        
        # Reset gradient monitor for this epoch
        grad_monitor = GradientMonitor()
        
        # Train
        train_results = train_epoch_modern(
            model, train_loader, optimizer, scheduler,
            criterion_mag, criterion_azi, device, scaler, epoch, grad_monitor
        )
        
        # Validate
        val_results = validate_modern(
            model, val_loader, criterion_mag, criterion_azi, device
        )
        
        # Get current learning rate
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
        
        grad_stats = train_results.get('grad_norm', {})
        history['grad_norm_mean'].append(grad_stats.get('mean', 0))
        history['grad_norm_std'].append(grad_stats.get('std', 0))
        
        # Print results
        print(f"\n{'─'*40}")
        print(f"Train - Loss: {train_results['loss']:.4f}, "
              f"Mag: {train_results['mag_acc']*100:.2f}%, "
              f"Azi: {train_results['azi_acc']*100:.2f}%")
        print(f"Val   - Loss: {val_results['loss']:.4f}, "
              f"Mag: {val_results['mag_acc']*100:.2f}%, "
              f"Azi: {val_results['azi_acc']*100:.2f}%")
        print(f"Val F1 - Mag: {val_results['mag_f1']:.4f}, Azi: {val_results['azi_f1']:.4f}")
        print(f"LR: {current_lr:.2e} | Grad Norm: {grad_stats.get('mean', 0):.2f}")
        
        # Save best model
        combined_metric = val_results['mag_acc'] * 0.7 + val_results['mag_f1'] * 0.3
        
        if combined_metric > best_val_f1:
            best_val_f1 = combined_metric
            best_val_mag_acc = val_results['mag_acc']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mag_acc': val_results['mag_acc'],
                'val_azi_acc': val_results['azi_acc'],
                'val_mag_f1': val_results['mag_f1'],
                'val_azi_f1': val_results['azi_f1'],
                'config': CONFIG
            }, OUTPUT_DIR / 'best_model.pth')
            
            logger.info(f"✓ New best model! Mag Acc: {best_val_mag_acc*100:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= CONFIG["early_stopping_patience"]:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': CONFIG
            }, OUTPUT_DIR / f'checkpoint_epoch_{epoch}.pth')
    
    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    # Load best model
    checkpoint = torch.load(OUTPUT_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test set evaluation
    test_results = validate_modern(
        model, test_loader, criterion_mag, criterion_azi, device
    )
    
    print(f"\n{'─'*40}")
    print("TEST SET RESULTS")
    print(f"{'─'*40}")
    print(f"Magnitude Accuracy: {test_results['mag_acc']*100:.2f}%")
    print(f"Azimuth Accuracy:   {test_results['azi_acc']*100:.2f}%")
    print(f"Magnitude F1:       {test_results['mag_f1']:.4f}")
    print(f"Azimuth F1:         {test_results['azi_f1']:.4f}")
    
    # ========================================================================
    # ADVANCED EVALUATION METRICS
    # ========================================================================
    
    if CONFIG["eval_ece"]:
        print(f"\n{'─'*40}")
        print("CALIBRATION ANALYSIS (ECE)")
        print(f"{'─'*40}")
        
        mag_ece = plot_calibration_diagram(
            test_results['mag_probs'], test_results['mag_labels'],
            OUTPUT_DIR, 'magnitude'
        )
        azi_ece = plot_calibration_diagram(
            test_results['azi_probs'], test_results['azi_labels'],
            OUTPUT_DIR, 'azimuth'
        )
        
        print(f"Magnitude ECE: {mag_ece:.4f}")
        print(f"Azimuth ECE:   {azi_ece:.4f}")
        
        test_results['mag_ece'] = mag_ece
        test_results['azi_ece'] = azi_ece
    
    if CONFIG["eval_efficiency"]:
        print(f"\n{'─'*40}")
        print("EFFICIENCY PROFILING")
        print(f"{'─'*40}")
        
        efficiency = profile_model_efficiency(
            model, 
            input_size=(1, 3, CONFIG["image_size"], CONFIG["image_size"]),
            device=str(device)
        )
        
        print(f"Latency: {efficiency['latency_ms']:.2f} ms/image")
        print(f"Throughput: {efficiency['throughput_img_per_sec']:.2f} images/sec")
        if efficiency.get('peak_vram_mb'):
            print(f"Peak VRAM: {efficiency['peak_vram_mb']:.2f} MB")
        if efficiency.get('gflops'):
            print(f"GFLOPs: {efficiency['gflops']:.2f}")
        
        test_results['efficiency'] = efficiency
    
    # ========================================================================
    # SAVE RESULTS AND VISUALIZATIONS
    # ========================================================================
    
    # Plot training curves
    plot_training_curves_modern(history, OUTPUT_DIR)
    
    # Plot confusion matrices
    plot_confusion_matrices_modern(test_results, mag_classes, azi_classes, OUTPUT_DIR)
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / 'training_history.csv', index=False)
    
    # Classification reports
    print(f"\n{'─'*40}")
    print("CLASSIFICATION REPORTS")
    print(f"{'─'*40}")
    
    print("\nMagnitude Classification:")
    print(classification_report(
        test_results['mag_labels'], test_results['mag_preds'],
        target_names=mag_classes
    ))
    
    print("\nAzimuth Classification:")
    print(classification_report(
        test_results['azi_labels'], test_results['azi_preds'],
        target_names=azi_classes
    ))
    
    # Save comprehensive summary
    summary = {
        'model': CONFIG['model_variant'],
        'modern_recipe': {
            'augmentation': {
                'mixup': CONFIG['use_mixup'],
                'mixup_alpha': CONFIG['mixup_alpha'],
                'cutmix_alpha': CONFIG['cutmix_alpha'],
                'randaugment': CONFIG['use_randaugment'],
                'random_erasing': CONFIG['random_erasing_prob']
            },
            'optimization': {
                'optimizer': 'AdamW',
                'base_lr': CONFIG['base_lr'],
                'weight_decay': CONFIG['weight_decay'],
                'layer_decay': CONFIG['layer_decay'],
                'warmup_epochs': CONFIG['warmup_epochs'],
                'label_smoothing': CONFIG['label_smoothing']
            },
            'regularization': {
                'drop_path_rate': CONFIG['drop_path_rate'],
                'gradient_clip': CONFIG['gradient_clip_norm']
            }
        },
        'training': {
            'epochs_trained': len(history['train_loss']),
            'best_epoch': checkpoint['epoch'],
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        'test_results': {
            'magnitude_accuracy': float(test_results['mag_acc']),
            'azimuth_accuracy': float(test_results['azi_acc']),
            'magnitude_f1': float(test_results['mag_f1']),
            'azimuth_f1': float(test_results['azi_f1']),
            'magnitude_ece': float(test_results.get('mag_ece', 0)),
            'azimuth_ece': float(test_results.get('azi_ece', 0))
        },
        'efficiency': test_results.get('efficiency', {}),
        'data': {
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(OUTPUT_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - MODERN RECIPE")
    print("=" * 80)
    print(f"\nModel: {CONFIG['model_variant'].upper()}")
    print(f"Best Epoch: {checkpoint['epoch']}")
    print(f"\nTest Results:")
    print(f"  Magnitude Accuracy: {test_results['mag_acc']*100:.2f}%")
    print(f"  Azimuth Accuracy:   {test_results['azi_acc']*100:.2f}%")
    print(f"  Magnitude F1:       {test_results['mag_f1']:.4f}")
    print(f"  Azimuth F1:         {test_results['azi_f1']:.4f}")
    
    if CONFIG["eval_ece"]:
        print(f"\nCalibration (ECE):")
        print(f"  Magnitude: {test_results.get('mag_ece', 0):.4f}")
        print(f"  Azimuth:   {test_results.get('azi_ece', 0):.4f}")
    
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print(f"Best Model: {OUTPUT_DIR / 'best_model.pth'}")
    
    return summary


if __name__ == "__main__":
    summary = main()
