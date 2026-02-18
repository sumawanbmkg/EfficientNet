#!/usr/bin/env python3
"""
Grad-CAM Visualization for ConvNeXt Earthquake Prediction Model
Generates interpretability visualizations for publication

Target Journals:
- IEEE Transactions on Geoscience and Remote Sensing (TGRS)
- Journal of Geophysical Research (JGR): Solid Earth
- Scientific Reports (Nature Portfolio)

Author: Earthquake Prediction Research Team
Date: 13 February 2026
Version: 1.0
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import cv2
from tqdm import tqdm

from earthquake_cnn_v3 import EarthquakeCNNV3
from earthquake_dataset_v3 import create_dataloaders_v3


class GradCAM:
    """Grad-CAM implementation for ConvNeXt"""
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Initialize Grad-CAM
        
        Args:
            model: The model to analyze
            target_layer: Name of target layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        layer = dict(model.named_modules())[target_layer]
        layer.register_forward_hook(self.save_activation)
        layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activation from forward pass"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradient from backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image: torch.Tensor, 
                    target_class: int, task: str = 'magnitude') -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Input image tensor (1, C, H, W)
            target_class: Target class index
            task: 'magnitude' or 'azimuth'
            
        Returns:
            Heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        mag_logits, az_logits = self.model(input_image)
        
        # Select task
        if task == 'magnitude':
            logits = mag_logits
        else:
            logits = az_logits
        
        # Backward pass
        self.model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, 
                   alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlay heatmap on image
    
    Args:
        image: Original image (H, W, 3) in range [0, 1]
        heatmap: Heatmap (H, W) in range [0, 1]
        alpha: Overlay transparency
        colormap: OpenCV colormap
        
    Returns:
        Overlayed image
    """
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert to colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_colored = heatmap_colored.astype(np.float32) / 255
    
    # Convert image to RGB if needed
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    
    # Overlay
    overlayed = (1 - alpha) * image + alpha * heatmap_colored
    overlayed = np.clip(overlayed, 0, 1)
    
    return overlayed


def generate_gradcam_visualizations(model_path: str, output_dir: str, 
                                   num_samples: int = 50):
    """
    Generate Grad-CAM visualizations
    
    Args:
        model_path: Path to trained model checkpoint
        output_dir: Output directory for visualizations
        num_samples: Number of samples to visualize
    """
    print("🎨 Generating Grad-CAM Visualizations for ConvNeXt")
    print("=" * 80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model config from checkpoint
    config = checkpoint.get('config', {})
    
    # Create model
    from earthquake_cnn_v3 import create_model_v3
    model, _ = create_model_v3(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    print(f"Device: {device}")
    
    #  Load data
    print("Loading dataset...")
    _, _, test_loader = create_dataloaders_v3(
        dataset_dir='dataset_experiment_3',
        batch_size=1,
        image_size=224,
        num_workers=0
    )
    
    # Target layers for visualization
    target_layers = [
        'backbone.features.6',  # Mid-level features
        'backbone.features.7'   # High-level features
    ]
    
    # Class names
    magnitude_classes = ['Medium', 'Large', 'Moderate', 'Normal']
    azimuth_classes = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'Normal']
    
    # Generate visualizations
    visualizations = []
    
    print(f"\nGenerating Grad-CAM for {num_samples} samples...")
    
    with torch.no_grad():
        for idx, (images, mag_labels, az_labels) in enumerate(tqdm(test_loader)):
            if idx >= num_samples:
                break
            
            images = images.to(device)
            mag_labels = mag_labels.to(device)
            az_labels = az_labels.to(device)
            
            # Get predictions
            mag_logits, az_logits = model(images)
            mag_pred = mag_logits.argmax(1).item()
            az_pred = az_logits.argmax(1).item()
            mag_true = mag_labels.item()
            az_true = az_labels.item()
            
            # Convert image for visualization
            img_np = images[0].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            # Generate Grad-CAM for each layer and task
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Sample {idx+1}\n'
                        f'Magnitude: True={magnitude_classes[mag_true]}, '
                        f'Pred={magnitude_classes[mag_pred]}\n'
                        f'Azimuth: True={azimuth_classes[az_true]}, '
                        f'Pred={azimuth_classes[az_pred]}',
                        fontsize=12, fontweight='bold')
            
            # Original image
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title('Original Spectrogram')
            axes[0, 0].axis('off')
            
            # Magnitude Grad-CAM
            for layer_idx, layer_name in enumerate(target_layers[:2]):
                gradcam = GradCAM(model, layer_name)
                
                # Magnitude task
                images.requires_grad = True
                cam = gradcam.generate_cam(images, mag_pred, task='magnitude')
                overlayed = overlay_heatmap(img_np, cam, alpha=0.4)
                
                axes[0, layer_idx + 1].imshow(overlayed)
                axes[0, layer_idx + 1].set_title(f'Magnitude CAM\n{layer_name.split(".")[-1]}')
                axes[0, layer_idx + 1].axis('off')
            
            # Azimuth Grad-CAM
            axes[1, 0].imshow(img_np)
            axes[1, 0].set_title('Original Spectrogram')
            axes[1, 0].axis('off')
            
            for layer_idx, layer_name in enumerate(target_layers[:2]):
                gradcam = GradCAM(model, layer_name)
                
                # Azimuth task
                images.requires_grad = True
                cam = gradcam.generate_cam(images, az_pred, task='azimuth')
                overlayed = overlay_heatmap(img_np, cam, alpha=0.4)
                
                axes[1, layer_idx + 1].imshow(overlayed)
                axes[1, layer_idx + 1].set_title(f'Azimuth CAM\n{layer_name.split(".")[-1]}')
                axes[1, layer_idx + 1].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            output_file = output_path / f'gradcam_sample_{idx+1:03d}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualizations.append({
                'sample_id': idx + 1,
                'magnitude_true': magnitude_classes[mag_true],
                'magnitude_pred': magnitude_classes[mag_pred],
                'azimuth_true': azimuth_classes[az_true],
                'azimuth_pred': azimuth_classes[az_pred],
                'file': str(output_file.name)
            })
    
    # Save metadata
    metadata = {
        'model_path': str(model_path),
        'num_samples': num_samples,
        'target_layers': target_layers,
        'visualizations': visualizations
    }
    
    metadata_file = output_path / 'gradcam_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Generate summary statistics
    correct_mag = sum(1 for v in visualizations 
                     if v['magnitude_true'] == v['magnitude_pred'])
    correct_az = sum(1 for v in visualizations 
                    if v['azimuth_true'] == v['azimuth_pred'])
    
    print("\n" + "=" * 80)
    print("✅ Grad-CAM Generation Completed!")
    print("=" * 80)
    print(f"📁 Output directory: {output_path}")
    print(f"📊 Samples visualized: {len(visualizations)}")
    print(f"📊 Magnitude accuracy: {correct_mag}/{len(visualizations)} "
          f"({100*correct_mag/len(visualizations):.1f}%)")
    print(f"📊 Azimuth accuracy: {correct_az}/{len(visualizations)} "
          f"({100*correct_az/len(visualizations):.1f}%)")
    print(f"📄 Metadata saved to: {metadata_file}")
    print("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM visualizations for ConvNeXt'
    )
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, 
                       default='gradcam_analysis',
                       help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Generate visualizations
    generate_gradcam_visualizations(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
