#!/usr/bin/env python3
"""
Simple Grad-CAM for ConvNeXt - FIXED VERSION
Uses correct dataset: dataset_experiment_3/final_metadata/test_exp3.csv

Author: Earthquake Prediction Research Team
Date: 13 February 2026
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from earthquake_cnn_v3 import EarthquakeCNNV3


class SimpleGradCAM:
    """Simple Grad-CAM implementation"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer = dict(model.named_modules())[target_layer_name]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class, task='magnitude'):
        """Generate CAM"""
        self.model.eval()
        self.model.zero_grad()
        
        # Forward
        mag_out, az_out = self.model(input_tensor)
        
        # Select task
        output = mag_out if task == 'magnitude' else az_out
        
        # Backward
        output[0, target_class].backward()
        
        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:])
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


class SimpleDataset(Dataset):
    """Simple dataset for Grad-CAM"""
    
    def __init__(self, df, dataset_root='dataset_experiment_3'):
        self.df = df.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mag_classes = sorted(df['magnitude_class'].unique())
        self.mag_to_idx = {c: i for i, c in enumerate(self.mag_classes)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.dataset_root / row['consolidation_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Labels
        mag_label = self.mag_to_idx[row['magnitude_class']]
        
        return image, mag_label, 0


def overlay_cam(image, cam, alpha=0.4):
    """Overlay CAM on image"""
    # Resize CAM
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR_RGB) / 255.0
    
    # Overlay
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    result = (1 - alpha) * image + alpha * heatmap
    return np.clip(result, 0, 1)


def generate_gradcam_simple(model_path, test_csv='dataset_experiment_3/final_metadata/test_exp3.csv', num_samples=30):
    """Generate Grad-CAM visualizations"""
    
    print("="*80)
    print("GRAD-CAM GENERATION - Simple Version")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path('gradcam_results')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Model: {model_path}")
    print(f"Test CSV: {test_csv}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Samples: {num_samples}")
    
    # Load model
    print("\n[1/3] Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = EarthquakeCNNV3(
        num_magnitude_classes=4,
        num_azimuth_classes=9
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("  Model loaded OK")
    
    # Load data
    print("\n[2/3] Loading test data...")
    df_test = pd.read_csv(test_csv)
    print(f"  Total test samples: {len(df_test)}")
    
    test_dataset = SimpleDataset(df_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Class names
    mag_names = sorted(df_test['magnitude_class'].unique())
    
    # Generate Grad-CAM
    print("\n[3/3] Generating Grad-CAM visualizations...")
    
    # Target layer
    target_layer = 'backbone.features.7'
    
    count = 0
    for batch_idx, (images, mag_labels, _) in enumerate(tqdm(test_loader)):
        if count >= num_samples:
            break
        
        images = images.to(device)
        images.requires_grad = True
        
        # Get predictions
        with torch.no_grad():
            mag_out, _ = model(images)
            mag_pred = mag_out.argmax(1).item()
        
        mag_true = mag_labels.item()
        
        # Prepare image
        img_np = images[0].cpu().detach().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Create visualizations
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(img_np)
        axes[0].set_title('Original Spectrogram', fontsize=10)
        axes[0].axis('off')
        
        # Magnitude Grad-CAM
        gradcam = SimpleGradCAM(model, target_layer)
        cam_mag = gradcam.generate(images, mag_pred, task='magnitude')
        overlay_mag = overlay_cam(img_np, cam_mag)
        
        axes[1].imshow(overlay_mag)
        axes[1].set_title(f'Magnitude CAM\nTrue: {mag_names[mag_true]}, Pred: {mag_names[mag_pred]}', 
                         fontsize=10)
        axes[1].axis('off')
        
        # Heatmap only
        axes[2].imshow(cam_mag, cmap='jet')
        axes[2].set_title('Magnitude Heatmap', fontsize=10)
        axes[2].axis('off')
        
        # Main title
        correct_mag = 'CORRECT' if mag_pred == mag_true else 'WRONG'
        fig.suptitle(f'Sample {count+1} - Prediction: {correct_mag}', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_file = output_dir / f'gradcam_{count+1:03d}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        count += 1
    
    # Summary
    print("\n" + "="*80)
    print("GRAD-CAM GENERATION COMPLETED")
    print("="*80)
    print(f"Samples generated: {count}")
    print(f"Output directory: {output_dir}")
    print(f"Files: gradcam_001.png to gradcam_{count:03d}.png")
    print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Grad-CAM for ConvNeXt')
    parser.add_argument('--model-path', type=str,
                       default='experiments_convnext/convnext_tiny_20260205_100924/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--test-csv', type=str,
                       default='dataset_experiment_3/final_metadata/test_exp3.csv',
                       help='Path to test CSV')
    parser.add_argument('--num-samples', type=int, default=30,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Check model
    if not Path(args.model_path).exists():
        print(f"ERROR: Model not found: {args.model_path}")
        print("\nAvailable models:")
        for f in Path('experiments_convnext').rglob('*.pth'):
            print(f"  {f}")
        return 1
    
    # Check test CSV
    if not Path(args.test_csv).exists():
        print(f"ERROR: Test CSV not found: {args.test_csv}")
        return 1
    
    # Generate
    generate_gradcam_simple(args.model_path, args.test_csv, args.num_samples)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
