"""
Multi-Task CNN Model for Earthquake Prediction
Predicts both Magnitude and Azimuth from geomagnetic spectrograms

Architecture:
- Shared CNN backbone (feature extraction)
- Two separate heads (magnitude & azimuth classification)
- Multi-task learning with weighted loss

Author: Earthquake Prediction Research Team
Date: 29 January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiTaskEarthquakeCNN(nn.Module):
    """
    Multi-Task CNN for earthquake prediction
    
    Outputs:
    - Magnitude: 5 classes (Small, Moderate, Medium, Large, Major)
    - Azimuth: 8 classes (N, NE, E, SE, S, SW, W, NW)
    """
    
    def __init__(self, backbone='resnet50', pretrained=True, 
                 num_magnitude_classes=5, num_azimuth_classes=8,
                 dropout_rate=0.5):
        """
        Initialize multi-task model
        
        Args:
            backbone: CNN backbone ('resnet50', 'resnet18', 'efficientnet_b0')
            pretrained: Use ImageNet pretrained weights
            num_magnitude_classes: Number of magnitude classes (default: 5)
            num_azimuth_classes: Number of azimuth classes (default: 8)
            dropout_rate: Dropout rate for regularization
        """
        super(MultiTaskEarthquakeCNN, self).__init__()
        
        self.backbone_name = backbone
        self.num_magnitude_classes = num_magnitude_classes
        self.num_azimuth_classes = num_azimuth_classes
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
            
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Shared feature processing
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(512)
        )
        
        # Magnitude classification head
        self.magnitude_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_magnitude_classes)
        )
        
        # Azimuth classification head
        self.azimuth_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_azimuth_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 3, 224, 224)
            
        Returns:
            magnitude_logits: (batch_size, num_magnitude_classes)
            azimuth_logits: (batch_size, num_azimuth_classes)
        """
        # Extract features with backbone
        features = self.backbone(x)
        
        # Shared feature processing
        shared_features = self.shared_fc(features)
        
        # Task-specific predictions
        magnitude_logits = self.magnitude_head(shared_features)
        azimuth_logits = self.azimuth_head(shared_features)
        
        return magnitude_logits, azimuth_logits
    
    def freeze_backbone(self):
        """Freeze backbone for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learnable weights
    
    Loss = w1 * magnitude_loss + w2 * azimuth_loss
    
    Weights are learned during training using uncertainty weighting
    (Kendall et al., 2018)
    """
    
    def __init__(self, learn_weights=True):
        """
        Initialize multi-task loss
        
        Args:
            learn_weights: Learn task weights automatically
        """
        super(MultiTaskLoss, self).__init__()
        
        self.learn_weights = learn_weights
        
        if learn_weights:
            # Learnable log variance for each task
            self.log_var_magnitude = nn.Parameter(torch.zeros(1))
            self.log_var_azimuth = nn.Parameter(torch.zeros(1))
        else:
            # Fixed weights
            self.weight_magnitude = 1.0
            self.weight_azimuth = 1.0
    
    def forward(self, magnitude_logits, azimuth_logits, 
                magnitude_targets, azimuth_targets):
        """
        Compute multi-task loss
        
        Args:
            magnitude_logits: Predicted magnitude logits
            azimuth_logits: Predicted azimuth logits
            magnitude_targets: True magnitude labels
            azimuth_targets: True azimuth labels
            
        Returns:
            total_loss: Combined loss
            magnitude_loss: Magnitude classification loss
            azimuth_loss: Azimuth classification loss
        """
        # Individual losses
        magnitude_loss = F.cross_entropy(magnitude_logits, magnitude_targets)
        azimuth_loss = F.cross_entropy(azimuth_logits, azimuth_targets)
        
        if self.learn_weights:
            # Uncertainty weighting (Kendall et al., 2018)
            # Loss = (1 / (2 * sigma^2)) * loss + log(sigma)
            precision_magnitude = torch.exp(-self.log_var_magnitude)
            precision_azimuth = torch.exp(-self.log_var_azimuth)
            
            total_loss = (
                precision_magnitude * magnitude_loss + self.log_var_magnitude +
                precision_azimuth * azimuth_loss + self.log_var_azimuth
            )
        else:
            # Fixed weights
            total_loss = (
                self.weight_magnitude * magnitude_loss +
                self.weight_azimuth * azimuth_loss
            )
        
        return total_loss, magnitude_loss, azimuth_loss


def create_model(config):
    """
    Create model from configuration
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        model: MultiTaskEarthquakeCNN instance
        criterion: MultiTaskLoss instance
    """
    model = MultiTaskEarthquakeCNN(
        backbone=config.get('backbone', 'resnet50'),
        pretrained=config.get('pretrained', True),
        num_magnitude_classes=config.get('num_magnitude_classes', 5),
        num_azimuth_classes=config.get('num_azimuth_classes', 8),
        dropout_rate=config.get('dropout_rate', 0.5)
    )
    
    criterion = MultiTaskLoss(
        learn_weights=config.get('learn_weights', True)
    )
    
    return model, criterion


if __name__ == '__main__':
    # Test model
    print("Testing Multi-Task Earthquake CNN Model...")
    
    # Create model
    config = {
        'backbone': 'resnet50',
        'pretrained': True,
        'num_magnitude_classes': 5,
        'num_azimuth_classes': 8,
        'dropout_rate': 0.5,
        'learn_weights': True
    }
    
    model, criterion = create_model(config)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    magnitude_logits, azimuth_logits = model(x)
    
    print(f"\nModel: {config['backbone']}")
    print(f"Input shape: {x.shape}")
    print(f"Magnitude output shape: {magnitude_logits.shape}")
    print(f"Azimuth output shape: {azimuth_logits.shape}")
    
    # Test loss
    magnitude_targets = torch.randint(0, 5, (batch_size,))
    azimuth_targets = torch.randint(0, 8, (batch_size,))
    
    total_loss, mag_loss, az_loss = criterion(
        magnitude_logits, azimuth_logits,
        magnitude_targets, azimuth_targets
    )
    
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"Magnitude loss: {mag_loss.item():.4f}")
    print(f"Azimuth loss: {az_loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n[OK] Model test passed!")
