"""
Focal Loss Implementation for Multi-Task Earthquake Precursor Detection
Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in earthquake precursor detection.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma (float): Focusing parameter. Default: 2.0
        alpha (Tensor): Class weights. Default: None (uniform)
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'
    """
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions (N, C) where C = number of classes
            targets: Ground truth labels (N,)
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight
            
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskFocalLoss(nn.Module):
    """
    Combined Focal Loss for magnitude and azimuth prediction.
    
    L_total = L_mag + lambda * L_azi
    
    Args:
        gamma (float): Focusing parameter for both tasks
        mag_weights (Tensor): Class weights for magnitude (4 classes)
        azi_weights (Tensor): Class weights for azimuth (9 classes)
        lambda_azi (float): Weight for azimuth loss. Default: 0.5
    """
    
    def __init__(self, gamma=2.0, mag_weights=None, azi_weights=None, lambda_azi=0.5):
        super(MultiTaskFocalLoss, self).__init__()
        self.mag_loss = FocalLoss(gamma=gamma, alpha=mag_weights)
        self.azi_loss = FocalLoss(gamma=gamma, alpha=azi_weights)
        self.lambda_azi = lambda_azi
        
    def forward(self, mag_pred, azi_pred, mag_target, azi_target):
        """
        Args:
            mag_pred: Magnitude predictions (N, 4)
            azi_pred: Azimuth predictions (N, 9)
            mag_target: Magnitude labels (N,)
            azi_target: Azimuth labels (N,)
        Returns:
            Total loss, magnitude loss, azimuth loss
        """
        l_mag = self.mag_loss(mag_pred, mag_target)
        l_azi = self.azi_loss(azi_pred, azi_target)
        l_total = l_mag + self.lambda_azi * l_azi
        
        return l_total, l_mag, l_azi


# Example usage
if __name__ == "__main__":
    # Class weights for imbalanced dataset
    # Magnitude: [Normal, Moderate, Medium, Large] = [888, 20, 1036, 28]
    mag_weights = torch.tensor([0.25, 2.5, 0.22, 2.0])
    
    # Azimuth: 9 directions (approximately balanced)
    azi_weights = torch.ones(9)
    
    # Initialize loss
    criterion = MultiTaskFocalLoss(
        gamma=2.0,
        mag_weights=mag_weights,
        azi_weights=azi_weights,
        lambda_azi=0.5
    )
    
    # Example forward pass
    batch_size = 32
    mag_pred = torch.randn(batch_size, 4)
    azi_pred = torch.randn(batch_size, 9)
    mag_target = torch.randint(0, 4, (batch_size,))
    azi_target = torch.randint(0, 9, (batch_size,))
    
    total_loss, mag_loss, azi_loss = criterion(mag_pred, azi_pred, mag_target, azi_target)
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Magnitude Loss: {mag_loss.item():.4f}")
    print(f"Azimuth Loss: {azi_loss.item():.4f}")
