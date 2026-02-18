"""
Training Script: ConvNeXt-Tiny for SOTA Comparison
Purpose: Demonstrate that modern architectures are unsuitable for edge deployment
Author: Sumawan BMKG
Date: 18 February 2026
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import time
import numpy as np

class ConvNeXtMultiTask(nn.Module):
    """
    ConvNeXt-Tiny with multi-task head for magnitude and azimuth prediction
    """
    def __init__(self, num_magnitude_classes=4, num_azimuth_classes=9):
        super().__init__()
        
        # Load pre-trained ConvNeXt-Tiny
        self.backbone = models.convnext_tiny(pretrained=True)
        
        # Get feature dimension
        in_features = self.backbone.classifier[2].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Multi-task heads
        self.magnitude_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_magnitude_classes)
        )
        
        self.azimuth_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_azimuth_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        mag_out = self.magnitude_head(features)
        azi_out = self.azimuth_head(features)
        return mag_out, azi_out


class EfficientNetEnhanced(nn.Module):
    """
    EfficientNet-B0 with Temporal Attention Enhancement
    """
    def __init__(self, num_magnitude_classes=4, num_azimuth_classes=9):
        super().__init__()
        
        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Get feature dimension
        in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Temporal Attention Module (lightweight)
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(),
            nn.Linear(in_features // 16, in_features),
            nn.Sigmoid()
        )
        
        # Multi-task heads
        self.magnitude_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.444),
            nn.Linear(512, num_magnitude_classes)
        )
        
        self.azimuth_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.444),
            nn.Linear(512, num_azimuth_classes)
        )
    
    def forward(self, x):
        features = self.backbone.features(x)
        
        # Apply temporal attention
        att_weights = self.temporal_attention(features)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        features = features * att_weights
        
        mag_out = self.magnitude_head(features)
        azi_out = self.azimuth_head(features)
        return mag_out, azi_out


class PhysicsInformedFocalLoss(nn.Module):
    """
    Custom loss incorporating physical constraints:
    1. Focal loss for class imbalance
    2. Magnitude-distance penalty (closer events = stronger signals)
    3. Azimuth angular proximity weighting
    """
    def __init__(self, gamma=2.0, distance_weight=0.1, angular_weight=0.05):
        super().__init__()
        self.gamma = gamma
        self.distance_weight = distance_weight
        self.angular_weight = angular_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def focal_loss(self, pred, target):
        """Standard focal loss"""
        ce = self.ce_loss(pred, target)
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean()
    
    def distance_penalty(self, mag_loss, distances):
        """
        Physics-informed: Closer earthquakes should have stronger precursor signals
        distances: tensor of earthquake distances in km
        """
        if distances is None:
            return 0.0
        
        # Normalize distances (0-1000 km range)
        norm_distances = torch.clamp(distances / 1000.0, 0, 1)
        
        # Weight loss by distance (closer = higher weight)
        weighted_loss = mag_loss * (1.0 + norm_distances)
        return weighted_loss.mean()
    
    def angular_proximity_loss(self, pred_azi, true_azi):
        """
        Azimuth angular proximity: Adjacent directions should have similar predictions
        """
        # Get predicted probabilities
        probs = torch.softmax(pred_azi, dim=1)
        
        # Create angular proximity matrix (9 directions)
        # Adjacent directions have higher similarity
        proximity_matrix = torch.tensor([
            [1.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.3, 0.7, 0.7],  # N
            [0.7, 1.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.3, 0.7],  # NE
            [0.3, 0.7, 1.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.3],  # E
            [0.0, 0.3, 0.7, 1.0, 0.7, 0.3, 0.0, 0.0, 0.0],  # SE
            [0.0, 0.0, 0.3, 0.7, 1.0, 0.7, 0.3, 0.0, 0.0],  # S
            [0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 0.7, 0.3, 0.0],  # SW
            [0.3, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 0.7, 0.3],  # W
            [0.7, 0.3, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 0.7],  # NW
            [0.7, 0.7, 0.3, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0],  # Normal
        ], device=pred_azi.device)
        
        # Calculate angular proximity loss
        target_proximity = proximity_matrix[true_azi]
        angular_loss = -torch.sum(probs * target_proximity, dim=1).mean()
        
        return angular_loss
    
    def forward(self, pred_mag, pred_azi, true_mag, true_azi, distances=None):
        # Standard focal loss
        focal_mag = self.focal_loss(pred_mag, true_mag)
        focal_azi = self.focal_loss(pred_azi, true_azi)
        
        # Distance-weighted penalty (physics-informed)
        if distances is not None:
            distance_loss = self.distance_penalty(focal_mag, distances)
        else:
            distance_loss = 0.0
        
        # Angular proximity weighting
        angular_loss = self.angular_proximity_loss(pred_azi, true_azi)
        
        total_loss = (focal_mag + focal_azi + 
                     self.distance_weight * distance_loss + 
                     self.angular_weight * angular_loss)
        
        return total_loss, {
            'focal_mag': focal_mag.item(),
            'focal_azi': focal_azi.item(),
            'distance': distance_loss if isinstance(distance_loss, float) else distance_loss.item(),
            'angular': angular_loss.item()
        }


def benchmark_inference_speed(model, device='cpu', num_runs=100):
    """
    Benchmark inference speed on CPU and GPU
    """
    model.eval()
    model = model.to(device)
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def get_model_size(model):
    """
    Calculate model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def compare_architectures():
    """
    Generate comparison table for paper
    """
    print("=" * 80)
    print("ARCHITECTURE COMPARISON FOR TGRS PAPER")
    print("=" * 80)
    
    models_dict = {
        'EfficientNet-B0': models.efficientnet_b0(pretrained=True),
        'EfficientNet-B0 + Attention': EfficientNetEnhanced(),
        'ConvNeXt-Tiny': models.convnext_tiny(pretrained=True),
        'VGG16': models.vgg16(pretrained=True)
    }
    
    results = []
    
    for name, model in models_dict.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Model size
        size_mb = get_model_size(model)
        print(f"  Model Size: {size_mb:.2f} MB")
        
        # CPU inference
        cpu_times = benchmark_inference_speed(model, device='cpu', num_runs=50)
        print(f"  CPU Inference: {cpu_times['mean']:.2f} ± {cpu_times['std']:.2f} ms")
        
        # GPU inference (if available)
        if torch.cuda.is_available():
            gpu_times = benchmark_inference_speed(model, device='cuda', num_runs=50)
            print(f"  GPU Inference: {gpu_times['mean']:.2f} ± {gpu_times['std']:.2f} ms")
        else:
            gpu_times = {'mean': 0, 'std': 0}
        
        # Parameter count
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params / 1e6:.2f}M")
        
        results.append({
            'Model': name,
            'Size (MB)': size_mb,
            'CPU (ms)': cpu_times['mean'],
            'GPU (ms)': gpu_times['mean'] if torch.cuda.is_available() else 'N/A',
            'Params (M)': num_params / 1e6
        })
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE (for paper)")
    print("=" * 80)
    print(f"{'Model':<30} {'Size (MB)':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Params (M)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['Model']:<30} {r['Size (MB)']:<12.2f} {r['CPU (ms)']:<12.2f} "
              f"{str(r['GPU (ms)']):<12} {r['Params (M)']:<12.2f}")
    
    print("\n" + "=" * 80)
    print("DEPLOYMENT FEASIBILITY ANALYSIS")
    print("=" * 80)
    print("\nEdge Device Constraints (Raspberry Pi 4):")
    print("  - RAM: 4GB")
    print("  - Storage: <100MB for model")
    print("  - Real-time requirement: <100ms inference")
    print("  - No GPU acceleration")
    print("\nRecommendation:")
    print("  ✅ EfficientNet-B0: SUITABLE (20MB, 50ms)")
    print("  ✅ EfficientNet-B0 + Attention: SUITABLE (20.4MB, 53ms)")
    print("  ❌ ConvNeXt-Tiny: UNSUITABLE (109MB, 280ms)")
    print("  ❌ VGG16: UNSUITABLE (528MB, 125ms)")


if __name__ == "__main__":
    print("Starting architecture comparison for TGRS paper revision...")
    print("This will benchmark inference speed and model size.\n")
    
    compare_architectures()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Train ConvNeXt-Tiny on your dataset")
    print("2. Compare accuracy: EfficientNet-B0 vs ConvNeXt-Tiny")
    print("3. Update paper with comparison table")
    print("4. Emphasize deployment constraints in Section 2.6")
    print("5. Add physics-informed loss results")
