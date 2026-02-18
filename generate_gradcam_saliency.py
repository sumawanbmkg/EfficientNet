#!/usr/bin/env python3
"""
Generate Grad-CAM and Saliency Maps for VGG16 Model
Visualize which parts of spectrogram are important for prediction

Features:
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Saliency Maps (Gradient visualization)
- Multiple sample visualization
- Save high-quality figures for paper

Author: Earthquake Prediction Research Team
Date: 3 February 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cv2

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


# Define the model architecture
class MultiTaskVGG16(nn.Module):
    """Multi-task VGG16 model"""
    
    def __init__(self, num_magnitude_classes, num_azimuth_classes):
        super(MultiTaskVGG16, self).__init__()
        vgg16 = models.vgg16(pretrained=False)
        self.features = vgg16.features
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.shared = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),  # Reduced dimension
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        
        # Magnitude head
        self.magnitude_head = nn.Linear(512, num_magnitude_classes)
        
        # Azimuth head
        self.azimuth_head = nn.Linear(512, num_azimuth_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared(x)
        
        mag_out = self.magnitude_head(x)
        azi_out = self.azimuth_head(x)
        
        return mag_out, azi_out


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization
    
    Reference:
    Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization. ICCV 2017.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained CNN model
            target_layer: Layer to visualize (e.g., last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Input tensor [1, 3, H, W]
            target_class: Target class index
            
        Returns:
            cam: Grad-CAM heatmap [H, W]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # For multi-task model, select magnitude or azimuth output
        if isinstance(output, tuple):
            output = output[0]  # Magnitude output
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


class SaliencyMap:
    """
    Saliency Map implementation
    Shows which pixels have highest gradient magnitude
    """
    
    def __init__(self, model):
        """
        Args:
            model: Trained CNN model
        """
        self.model = model
    
    def generate_saliency(self, input_image, target_class):
        """
        Generate saliency map
        
        Args:
            input_image: Input tensor [1, 3, H, W]
            target_class: Target class index
            
        Returns:
            saliency: Saliency map [H, W]
        """
        # Require gradient
        input_image.requires_grad = True
        
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # For multi-task model
        if isinstance(output, tuple):
            output = output[0]  # Magnitude output
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Get gradients
        gradients = input_image.grad.data.abs()  # [1, 3, H, W]
        
        # Take maximum across color channels
        saliency = torch.max(gradients[0], dim=0)[0]  # [H, W]
        
        # Normalize
        saliency = saliency - saliency.min()
        saliency = saliency / (saliency.max() + 1e-8)
        
        return saliency.cpu().numpy()


def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image
    
    Args:
        image: Original image [H, W, 3]
        heatmap: Heatmap [H, W]
        alpha: Transparency
        colormap: OpenCV colormap
        
    Returns:
        overlayed: Overlayed image [H, W, 3]
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), 
        colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = (1 - alpha) * image + alpha * heatmap_colored
    overlayed = np.uint8(overlayed)
    
    return overlayed


def visualize_sample(model, image_path, target_layer, class_names, 
                     output_dir, sample_name):
    """
    Generate and save Grad-CAM and Saliency Map for one sample
    
    Args:
        model: Trained model
        image_path: Path to spectrogram image
        target_layer: Target layer for Grad-CAM
        class_names: List of class names
        output_dir: Output directory
        sample_name: Sample identifier
    """
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        if isinstance(output, tuple):
            output = output[0]  # Magnitude
        pred_class = torch.argmax(output, dim=1).item()
        pred_prob = F.softmax(output, dim=1)[0, pred_class].item()
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate_cam(image_tensor, pred_class)
    
    # Generate Saliency Map
    saliency = SaliencyMap(model)
    saliency_map = saliency.generate_saliency(image_tensor.clone(), pred_class)
    
    # Prepare original image for overlay
    image_np = np.array(image.resize((224, 224)))
    
    # Create overlays
    gradcam_overlay = overlay_heatmap(image_np, cam, alpha=0.4)
    saliency_overlay = overlay_heatmap(image_np, saliency_map, alpha=0.4)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Grad-CAM
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Spectrogram', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cam, cmap='jet')
    axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gradcam_overlay)
    axes[0, 2].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Saliency Map
    axes[1, 0].imshow(image_np)
    axes[1, 0].set_title('Original Spectrogram', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(saliency_map, cmap='hot')
    axes[1, 1].set_title('Saliency Map', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(saliency_overlay)
    axes[1, 2].set_title('Saliency Overlay', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add prediction info
    pred_text = f"Prediction: {class_names[pred_class]}\nConfidence: {pred_prob:.2%}"
    fig.text(0.5, 0.02, pred_text, ha='center', fontsize=14, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save
    output_path = output_dir / f'{sample_name}_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    
    return {
        'sample': sample_name,
        'prediction': class_names[pred_class],
        'confidence': pred_prob,
        'visualization': str(output_path)
    }


def generate_comparison_figure(results, output_dir):
    """
    Generate comparison figure with multiple samples
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory
    """
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Load visualization
        vis_path = result['visualization']
        vis_img = Image.open(vis_path)
        
        # This is simplified - in practice, you'd load individual components
        axes[i, 0].text(0.5, 0.5, f"Sample {i+1}\n{result['prediction']}\n{result['confidence']:.2%}",
                       ha='center', va='center', fontsize=12)
        axes[i, 0].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'comparison_all_samples.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparison: {output_path}")


def main():
    """Main function"""
    print("="*70)
    print("GRAD-CAM AND SALIENCY MAP GENERATION")
    print("="*70)
    
    # Configuration
    model_path = 'experiments_fixed/exp_fixed_20260202_163643/best_model.pth'
    metadata_path = 'dataset_unified/metadata/unified_metadata.csv'
    output_dir = Path('visualization_gradcam')
    output_dir.mkdir(exist_ok=True)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ Model not found at: {model_path}")
        print("   Please check the path and try again.")
        return
    
    # Load model
    print("\n📊 Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load class mappings
    class_mappings_path = Path(model_path).parent / 'class_mappings.json'
    with open(class_mappings_path) as f:
        class_mappings = json.load(f)
    
    magnitude_classes = class_mappings['magnitude_classes']
    azimuth_classes = class_mappings['azimuth_classes']
    
    print(f"Magnitude classes: {magnitude_classes}")
    print(f"Azimuth classes: {azimuth_classes}")
    
    # Create model
    model = MultiTaskVGG16(len(magnitude_classes), len(azimuth_classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Get target layer (last conv layer of VGG16)
    target_layer = model.features[-1]  # Last conv layer
    print(f"Target layer: {target_layer}")
    
    # Select sample images
    import pandas as pd
    df = pd.read_csv(metadata_path)
    
    # Select one sample from each magnitude class (excluding Normal for now)
    samples = []
    for mag_class in magnitude_classes:
        if mag_class == 'Normal':
            continue  # Skip normal for precursor visualization
        
        class_samples = df[df['magnitude_class'] == mag_class]
        if len(class_samples) == 0:
            print(f"⚠️  No samples found for class: {mag_class}")
            continue
        
        sample = class_samples.iloc[0]
        sample_path = Path('dataset_unified') / sample['unified_path']
        
        if not sample_path.exists():
            print(f"⚠️  Sample not found: {sample_path}")
            continue
        
        samples.append({
            'path': sample_path,
            'class': mag_class,
            'name': f"{mag_class}_{sample['station']}_{sample['date']}"
        })
    
    print(f"\n📊 Selected {len(samples)} samples (one per class)")
    
    # Generate visualizations
    results = []
    for sample in samples:
        print(f"\nProcessing: {sample['name']}")
        result = visualize_sample(
            model, sample['path'], target_layer,
            magnitude_classes, output_dir, sample['name']
        )
        results.append(result)
    
    # Save results
    with open(output_dir / 'visualization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - Individual visualizations: {len(samples)} files")
    print(f"  - Results JSON: visualization_results.json")
    print(f"\nTo view the visualizations:")
    print(f"  cd {output_dir}")
    print(f"  # Open PNG files with image viewer")


if __name__ == '__main__':
    main()
