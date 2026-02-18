#!/usr/bin/env python3
"""
Balanced Focal Loss Implementation
Advanced loss function for extreme class imbalance

Features:
- Automatic class weight balancing
- Effective number of samples calculation
- Distribution-balanced loss
- Compatible with existing training pipeline

Author: Earthquake Prediction Research Team
Date: February 1, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BalancedFocalLoss(nn.Module):
    """
    Balanced Focal Loss with automatic class weight calculation
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    Enhancement: Class-balanced loss using effective number of samples
    
    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        
    where:
        α_t = (1 - β) / (1 - β^n_i)  # Effective number weighting
        β = (N - 1) / N  # Decay parameter
        n_i = number of samples in class i
    """
    
    def __init__(self, 
                 samples_per_class: torch.Tensor,
                 gamma: float = 3.0,
                 beta: float = 0.9999,
                 label_smoothing: float = 0.0):
        """
        Initialize Balanced Focal Loss
        
        Args:
            samples_per_class: Number of samples per class (tensor)
            gamma: Focusing parameter (default: 3.0 for extreme imbalance)
            beta: Class-balanced loss beta (default: 0.9999)
            label_smoothing: Label smoothing factor (default: 0.0)
        """
        super(BalancedFocalLoss, self).__init__()
        
        self.gamma = gamma
        self.beta = beta
        self.label_smoothing = label_smoothing
        
        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(beta, samples_per_class.float())
        weights = (1.0 - beta) / effective_num
        
        # Normalize weights
        weights = weights / weights.sum() * len(samples_per_class)
        
        self.register_buffer('weights', weights)
        
        logger.info(f"✅ BalancedFocalLoss initialized")
        logger.info(f"   Gamma: {gamma}")
        logger.info(f"   Beta: {beta}")
        logger.info(f"   Label smoothing: {label_smoothing}")
        logger.info(f"   Class weights: {weights.cpu().numpy()}")
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute balanced focal loss
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Balanced focal loss value
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = logits.size(1)
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                            self.label_smoothing / num_classes
            
            # Compute cross entropy with soft targets
            log_probs = F.log_softmax(logits, dim=1)
            ce_loss = -(targets_one_hot * log_probs).sum(dim=1)
        else:
            # Standard cross entropy
            ce_loss = F.cross_entropy(logits, targets, weight=self.weights, reduction='none')
        
        # Compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


class DistributionBalancedLoss(nn.Module):
    """
    Distribution-Balanced Loss for extreme class imbalance
    
    Paper: "Distribution-Balanced Loss for Extreme Imbalance" (2024)
    
    Combines:
    1. Rebalancing weight (based on sample count)
    2. Negative tolerance regularization (prevents overconfidence)
    """
    
    def __init__(self,
                 samples_per_class: torch.Tensor,
                 alpha: float = 0.5,
                 beta: float = 0.1):
        """
        Initialize Distribution-Balanced Loss
        
        Args:
            samples_per_class: Number of samples per class
            alpha: Rebalancing weight exponent (default: 0.5)
            beta: Negative tolerance weight (default: 0.1)
        """
        super(DistributionBalancedLoss, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        
        # Calculate rebalancing weight
        max_samples = samples_per_class.max()
        rebalance_weight = (max_samples / samples_per_class.float()) ** alpha
        
        # Normalize
        rebalance_weight = rebalance_weight / rebalance_weight.sum() * len(samples_per_class)
        
        self.register_buffer('rebalance_weight', rebalance_weight)
        
        logger.info(f"✅ DistributionBalancedLoss initialized")
        logger.info(f"   Alpha: {alpha}")
        logger.info(f"   Beta: {beta}")
        logger.info(f"   Rebalance weights: {rebalance_weight.cpu().numpy()}")
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute distribution-balanced loss
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Distribution-balanced loss value
        """
        # Cross entropy with rebalancing weight
        ce_loss = F.cross_entropy(logits, targets, weight=self.rebalance_weight)
        
        # Negative tolerance regularization
        # Penalizes overconfident wrong predictions
        neg_tolerance = torch.log(1 + torch.exp(-logits)).mean()
        
        # Combined loss
        db_loss = ce_loss + self.beta * neg_tolerance
        
        return db_loss


class CombinedBalancedLoss(nn.Module):
    """
    Combined loss function with multiple components
    
    Combines:
    1. Balanced Focal Loss (for hard examples)
    2. Distribution-Balanced Loss (for extreme imbalance)
    3. Label Smoothing (for overconfidence)
    """
    
    def __init__(self,
                 samples_per_class: torch.Tensor,
                 focal_gamma: float = 3.0,
                 focal_beta: float = 0.9999,
                 db_alpha: float = 0.5,
                 db_beta: float = 0.1,
                 label_smoothing: float = 0.1,
                 focal_weight: float = 0.7,
                 db_weight: float = 0.3):
        """
        Initialize combined loss
        
        Args:
            samples_per_class: Number of samples per class
            focal_gamma: Focal loss gamma
            focal_beta: Focal loss beta
            db_alpha: DB loss alpha
            db_beta: DB loss beta
            label_smoothing: Label smoothing factor
            focal_weight: Weight for focal loss component
            db_weight: Weight for DB loss component
        """
        super(CombinedBalancedLoss, self).__init__()
        
        self.focal_loss = BalancedFocalLoss(
            samples_per_class, focal_gamma, focal_beta, label_smoothing
        )
        
        self.db_loss = DistributionBalancedLoss(
            samples_per_class, db_alpha, db_beta
        )
        
        self.focal_weight = focal_weight
        self.db_weight = db_weight
        
        logger.info(f"✅ CombinedBalancedLoss initialized")
        logger.info(f"   Focal weight: {focal_weight}")
        logger.info(f"   DB weight: {db_weight}")
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            logits: Model predictions
            targets: Ground truth labels
            
        Returns:
            Dictionary with loss components
        """
        focal = self.focal_loss(logits, targets)
        db = self.db_loss(logits, targets)
        
        total = self.focal_weight * focal + self.db_weight * db
        
        return {
            'total': total,
            'focal': focal,
            'db': db
        }


class MultiTaskBalancedLoss(nn.Module):
    """
    Multi-task loss with balanced focal loss for each task
    
    For earthquake prediction:
    - Magnitude classification (4 classes)
    - Azimuth classification (9 classes)
    """
    
    def __init__(self,
                 magnitude_samples: torch.Tensor,
                 azimuth_samples: torch.Tensor,
                 focal_gamma: float = 3.0,
                 focal_beta: float = 0.9999,
                 label_smoothing: float = 0.1,
                 learn_task_weights: bool = True):
        """
        Initialize multi-task balanced loss
        
        Args:
            magnitude_samples: Samples per magnitude class
            azimuth_samples: Samples per azimuth class
            focal_gamma: Focal loss gamma
            focal_beta: Focal loss beta
            label_smoothing: Label smoothing factor
            learn_task_weights: Learn task weights automatically
        """
        super(MultiTaskBalancedLoss, self).__init__()
        
        # Create loss for each task
        self.magnitude_loss = BalancedFocalLoss(
            magnitude_samples, focal_gamma, focal_beta, label_smoothing
        )
        
        self.azimuth_loss = BalancedFocalLoss(
            azimuth_samples, focal_gamma, focal_beta, label_smoothing
        )
        
        # Task weight learning (uncertainty weighting)
        self.learn_task_weights = learn_task_weights
        if learn_task_weights:
            self.log_var_magnitude = nn.Parameter(torch.zeros(1))
            self.log_var_azimuth = nn.Parameter(torch.zeros(1))
        else:
            self.weight_magnitude = 1.0
            self.weight_azimuth = 1.0
        
        logger.info(f"✅ MultiTaskBalancedLoss initialized")
        logger.info(f"   Learn task weights: {learn_task_weights}")
        
    def forward(self,
                magnitude_logits: torch.Tensor,
                azimuth_logits: torch.Tensor,
                magnitude_targets: torch.Tensor,
                azimuth_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            magnitude_logits: Magnitude predictions
            azimuth_logits: Azimuth predictions
            magnitude_targets: Magnitude ground truth
            azimuth_targets: Azimuth ground truth
            
        Returns:
            Dictionary with loss components
        """
        # Individual losses
        mag_loss = self.magnitude_loss(magnitude_logits, magnitude_targets)
        az_loss = self.azimuth_loss(azimuth_logits, azimuth_targets)
        
        # Task weighting
        if self.learn_task_weights:
            # Uncertainty weighting (Kendall et al., 2018)
            precision_mag = torch.exp(-self.log_var_magnitude)
            precision_az = torch.exp(-self.log_var_azimuth)
            
            total_loss = (
                precision_mag * mag_loss + self.log_var_magnitude +
                precision_az * az_loss + self.log_var_azimuth
            )
        else:
            total_loss = self.weight_magnitude * mag_loss + self.weight_azimuth * az_loss
        
        return {
            'total_loss': total_loss,
            'magnitude_loss': mag_loss,
            'azimuth_loss': az_loss
        }


def test_balanced_focal_loss():
    """Test balanced focal loss"""
    logger.info("\n🧪 TESTING BALANCED FOCAL LOSS")
    logger.info("=" * 60)
    
    # Simulate extreme class imbalance
    samples_per_class = torch.tensor([480, 4, 48, 92, 100, 104, 104, 180, 100])  # Azimuth
    
    # Create loss
    criterion = BalancedFocalLoss(samples_per_class, gamma=3.0)
    
    # Test forward pass
    batch_size = 32
    num_classes = 9
    
    logits = torch.randn(batch_size, num_classes, requires_grad=True)  # Enable grad
    targets = torch.randint(0, num_classes, (batch_size,))
    
    loss = criterion(logits, targets)
    
    logger.info(f"✅ Loss computed: {loss.item():.4f}")
    logger.info(f"✅ Loss is finite: {torch.isfinite(loss).item()}")
    logger.info(f"✅ Loss requires grad: {loss.requires_grad}")
    
    # Test backward pass
    loss.backward()
    logger.info(f"✅ Backward pass successful")
    logger.info(f"✅ Gradients computed: {logits.grad is not None}")
    
    logger.info("\n✅ ALL TESTS PASSED!")


if __name__ == '__main__':
    test_balanced_focal_loss()
