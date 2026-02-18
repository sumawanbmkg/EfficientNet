#!/usr/bin/env python3
"""
Failure Analysis Script for Earthquake Precursor Model
Generates misclassification examples with Grad-CAM visualizations

This script:
1. Loads the trained model and test data
2. Identifies misclassified samples
3. Generates Grad-CAM visualizations for failure cases
4. Creates a comprehensive failure analysis report

Author: Earthquake Prediction Research Team
Date: 5 February 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import cv2

# Configuration
OUTPUT_DIR = Path("failure_analysis")
MODEL_PATH = Path("final_production_model/best_final_model.pth")
DATASET_DIR = Path("dataset_unified/spectrograms")
METADATA_PATH = Path("dataset_unified/metadata/unified_metadata.csv")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("FAILURE ANALYSIS - Earthquake Precursor Model")
print("=" * 60)


class EarthquakeCNN(nn.Module):
    """EfficientNet-B0 based multi-task model"""
    
    def __init__(self, num_mag_classes=4, num_azi_classes=9):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.mag_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_mag_classes)
        )
        
        self.azi_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_azi_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        mag_out = self.mag_head(features)
        azi_out = self.azi_head(features)
        return mag_out, azi_out


class GradCAM:
    """Grad-CAM implementation for model interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class, task='magnitude'):
        """Generate Grad-CAM heatmap"""
        self.model.zero_grad()
        
        # Forward pass
        mag_out, azi_out = self.model(input_tensor)
        
        # Select output based on task
        if task == 'magnitude':
            output = mag_out
        else:
            output = azi_out
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate heatmap
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()


def load_model():
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = EarthquakeCNN(num_mag_classes=4, num_azi_classes=9)
    
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}")
        return None, device
    
    model.to(device)
    model.eval()
    return model, device


def load_class_mappings():
    """Load or create class mappings"""
    mappings_path = Path("final_production_model/class_mappings.json")
    
    if mappings_path.exists():
        with open(mappings_path, 'r') as f:
            return json.load(f)
    
    # Default mappings
    return {
        "magnitude": {"0": "Medium", "1": "Normal", "2": "Large", "3": "Moderate"},
        "azimuth": {"0": "E", "1": "N", "2": "NE", "3": "NW", "4": "Normal",
                   "5": "S", "6": "SE", "7": "SW", "8": "W"}
    }


def get_transform():
    """Get image transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def find_misclassifications(model, device, metadata_df, transform, class_mappings):
    """Find all misclassified samples"""
    misclassifications = []
    
    # Reverse mappings
    mag_to_idx = {v: int(k) for k, v in class_mappings['magnitude'].items()}
    azi_to_idx = {v: int(k) for k, v in class_mappings['azimuth'].items()}
    
    print("\nScanning for misclassifications...")
    
    for idx, row in metadata_df.iterrows():
        img_path = DATASET_DIR / row['filename']
        
        if not img_path.exists():
            continue
        
        try:
            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Get predictions
            with torch.no_grad():
                mag_out, azi_out = model(img_tensor)
                mag_pred = torch.argmax(mag_out, dim=1).item()
                azi_pred = torch.argmax(azi_out, dim=1).item()
                mag_conf = torch.softmax(mag_out, dim=1)[0].cpu().numpy()
                azi_conf = torch.softmax(azi_out, dim=1)[0].cpu().numpy()
            
            # Get true labels
            true_mag = row.get('magnitude_class', row.get('mag_class', 'Unknown'))
            true_azi = row.get('azimuth_class', row.get('azi_class', 'Unknown'))
            
            true_mag_idx = mag_to_idx.get(true_mag, -1)
            true_azi_idx = azi_to_idx.get(true_azi, -1)
            
            # Check for misclassification
            mag_wrong = (true_mag_idx != -1) and (mag_pred != true_mag_idx)
            azi_wrong = (true_azi_idx != -1) and (azi_pred != true_azi_idx)
            
            if mag_wrong or azi_wrong:
                misclassifications.append({
                    'filename': row['filename'],
                    'path': str(img_path),
                    'true_mag': true_mag,
                    'pred_mag': class_mappings['magnitude'][str(mag_pred)],
                    'mag_confidence': mag_conf,
                    'true_azi': true_azi,
                    'pred_azi': class_mappings['azimuth'][str(azi_pred)],
                    'azi_confidence': azi_conf,
                    'mag_wrong': mag_wrong,
                    'azi_wrong': azi_wrong,
                    'station': row.get('station', 'Unknown'),
                    'date': row.get('date', 'Unknown'),
                    'event_id': row.get('event_id', idx)
                })
        
        except Exception as e:
            continue
    
    print(f"Found {len(misclassifications)} misclassified samples")
    return misclassifications


def generate_gradcam_visualization(model, device, img_path, true_label, pred_label, 
                                   confidence, task, class_mappings, output_path):
    """Generate Grad-CAM visualization for a misclassified sample"""
    
    # Load image
    image = Image.open(img_path).convert('RGB')
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get target layer (last conv layer of EfficientNet)
    target_layer = model.backbone.features[-1]
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Get class index
    if task == 'magnitude':
        class_to_idx = {v: int(k) for k, v in class_mappings['magnitude'].items()}
    else:
        class_to_idx = {v: int(k) for k, v in class_mappings['azimuth'].items()}
    
    pred_idx = class_to_idx.get(pred_label, 0)
    
    # Generate heatmap
    heatmap = gradcam.generate(img_tensor, pred_idx, task)
    
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img_array = np.array(image.resize((224, 224)))
    axes[0].imshow(img_array)
    axes[0].set_title(f'Original Spectrogram\nTrue: {true_label}')
    axes[0].axis('off')
    
    # Grad-CAM heatmap
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title(f'Grad-CAM Heatmap\nPred: {pred_label}')
    axes[1].axis('off')
    
    # Overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\nConf: {confidence:.1%}')
    axes[2].axis('off')
    
    plt.suptitle(f'Failure Analysis: {task.capitalize()} Misclassification', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def analyze_error_patterns(misclassifications, class_mappings):
    """Analyze patterns in misclassifications"""
    
    mag_errors = defaultdict(lambda: defaultdict(int))
    azi_errors = defaultdict(lambda: defaultdict(int))
    
    for m in misclassifications:
        if m['mag_wrong']:
            mag_errors[m['true_mag']][m['pred_mag']] += 1
        if m['azi_wrong']:
            azi_errors[m['true_azi']][m['pred_azi']] += 1
    
    return dict(mag_errors), dict(azi_errors)


def create_confusion_heatmap(errors, title, output_path, class_names):
    """Create confusion heatmap for errors"""
    
    # Create matrix
    n_classes = len(class_names)
    matrix = np.zeros((n_classes, n_classes))
    
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    
    for true_class, preds in errors.items():
        if true_class in name_to_idx:
            for pred_class, count in preds.items():
                if pred_class in name_to_idx:
                    matrix[name_to_idx[true_class], name_to_idx[pred_class]] = count
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path


def generate_failure_report(misclassifications, mag_errors, azi_errors, output_dir):
    """Generate comprehensive failure analysis report"""
    
    report_path = output_dir / "FAILURE_ANALYSIS_REPORT.md"
    
    with open(report_path, 'w') as f:
        f.write("# Failure Analysis Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Summary
        f.write("## 1. Summary\n\n")
        mag_wrong = sum(1 for m in misclassifications if m['mag_wrong'])
        azi_wrong = sum(1 for m in misclassifications if m['azi_wrong'])
        f.write(f"- **Total Misclassifications**: {len(misclassifications)}\n")
        f.write(f"- **Magnitude Errors**: {mag_wrong}\n")
        f.write(f"- **Azimuth Errors**: {azi_wrong}\n\n")
        
        # Magnitude Error Patterns
        f.write("## 2. Magnitude Error Patterns\n\n")
        f.write("| True Class | Predicted Class | Count |\n")
        f.write("|------------|-----------------|-------|\n")
        for true_class, preds in mag_errors.items():
            for pred_class, count in preds.items():
                f.write(f"| {true_class} | {pred_class} | {count} |\n")
        f.write("\n")
        
        # Azimuth Error Patterns
        f.write("## 3. Azimuth Error Patterns\n\n")
        f.write("| True Class | Predicted Class | Count |\n")
        f.write("|------------|-----------------|-------|\n")
        for true_class, preds in azi_errors.items():
            for pred_class, count in preds.items():
                f.write(f"| {true_class} | {pred_class} | {count} |\n")
        f.write("\n")
        
        # Detailed Cases
        f.write("## 4. Detailed Failure Cases\n\n")
        for i, m in enumerate(misclassifications[:10]):  # Top 10
            f.write(f"### Case {i+1}: {m['filename']}\n\n")
            f.write(f"- **Station**: {m['station']}\n")
            f.write(f"- **Date**: {m['date']}\n")
            if m['mag_wrong']:
                f.write(f"- **Magnitude**: True={m['true_mag']}, Pred={m['pred_mag']}\n")
                f.write(f"- **Confidence**: {m['mag_confidence'].max():.1%}\n")
            if m['azi_wrong']:
                f.write(f"- **Azimuth**: True={m['true_azi']}, Pred={m['pred_azi']}\n")
                f.write(f"- **Confidence**: {m['azi_confidence'].max():.1%}\n")
            f.write("\n")
        
        # Analysis
        f.write("## 5. Root Cause Analysis\n\n")
        f.write("### 5.1 Magnitude Misclassifications\n\n")
        f.write("Common causes:\n")
        f.write("1. **Adjacent class confusion**: Borderline magnitudes near class boundaries\n")
        f.write("2. **Signal amplitude ambiguity**: Similar ULF patterns across magnitude ranges\n")
        f.write("3. **Noise interference**: High noise in critical frequency bands\n\n")
        
        f.write("### 5.2 Azimuth Misclassifications\n\n")
        f.write("Common causes:\n")
        f.write("1. **Angular proximity**: Adjacent directions (45° apart) often confused\n")
        f.write("2. **Wave propagation complexity**: Multi-path effects in heterogeneous crust\n")
        f.write("3. **Station geometry**: Single-station limitation for directional estimation\n\n")
        
        # Recommendations
        f.write("## 6. Recommendations\n\n")
        f.write("1. **For rare class errors**: Collect more samples for Large/Moderate classes\n")
        f.write("2. **For adjacent class confusion**: Consider soft labels or regression approach\n")
        f.write("3. **For azimuth errors**: Implement multi-station fusion for better direction estimation\n")
        f.write("4. **For confidence calibration**: Apply temperature scaling post-training\n")
    
    print(f"Report saved to {report_path}")
    return report_path


def main():
    """Main execution function"""
    
    # Load model
    model, device = load_model()
    if model is None:
        print("Cannot proceed without model. Exiting.")
        return
    
    # Load class mappings
    class_mappings = load_class_mappings()
    print(f"Class mappings loaded")
    
    # Load metadata
    if METADATA_PATH.exists():
        metadata_df = pd.read_csv(METADATA_PATH)
        print(f"Loaded {len(metadata_df)} samples from metadata")
    else:
        print(f"Metadata not found at {METADATA_PATH}")
        # Try alternative paths
        alt_paths = [
            Path("dataset_smote/metadata/smote_metadata.csv"),
            Path("dataset_augmented/metadata/augmented_metadata.csv"),
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                metadata_df = pd.read_csv(alt_path)
                print(f"Using alternative metadata: {alt_path}")
                break
        else:
            print("No metadata found. Creating sample analysis...")
            metadata_df = pd.DataFrame()
    
    # Get transforms
    transform = get_transform()
    
    # Find misclassifications
    if len(metadata_df) > 0:
        misclassifications = find_misclassifications(
            model, device, metadata_df, transform, class_mappings
        )
    else:
        misclassifications = []
        print("No metadata available for misclassification analysis")
    
    # Analyze error patterns
    mag_errors, azi_errors = analyze_error_patterns(misclassifications, class_mappings)
    
    # Generate visualizations for top failures
    print("\nGenerating Grad-CAM visualizations...")
    for i, m in enumerate(misclassifications[:5]):
        if m['mag_wrong']:
            output_path = OUTPUT_DIR / f"failure_mag_{i+1}.png"
            generate_gradcam_visualization(
                model, device, m['path'], m['true_mag'], m['pred_mag'],
                m['mag_confidence'].max(), 'magnitude', class_mappings, output_path
            )
            print(f"  Generated: {output_path}")
        
        if m['azi_wrong']:
            output_path = OUTPUT_DIR / f"failure_azi_{i+1}.png"
            generate_gradcam_visualization(
                model, device, m['path'], m['true_azi'], m['pred_azi'],
                m['azi_confidence'].max(), 'azimuth', class_mappings, output_path
            )
            print(f"  Generated: {output_path}")
    
    # Create confusion heatmaps
    if mag_errors:
        mag_classes = list(class_mappings['magnitude'].values())
        create_confusion_heatmap(
            mag_errors, "Magnitude Misclassification Patterns",
            OUTPUT_DIR / "mag_error_heatmap.png", mag_classes
        )
    
    if azi_errors:
        azi_classes = list(class_mappings['azimuth'].values())
        create_confusion_heatmap(
            azi_errors, "Azimuth Misclassification Patterns",
            OUTPUT_DIR / "azi_error_heatmap.png", azi_classes
        )
    
    # Generate report
    generate_failure_report(misclassifications, mag_errors, azi_errors, OUTPUT_DIR)
    
    # Save misclassifications to JSON
    json_path = OUTPUT_DIR / "misclassifications.json"
    with open(json_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for m in misclassifications:
            m['mag_confidence'] = m['mag_confidence'].tolist()
            m['azi_confidence'] = m['azi_confidence'].tolist()
        json.dump(misclassifications, f, indent=2, default=str)
    print(f"Misclassifications saved to {json_path}")
    
    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total failures analyzed: {len(misclassifications)}")


if __name__ == "__main__":
    main()
