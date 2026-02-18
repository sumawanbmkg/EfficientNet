#!/usr/bin/env python3
"""
Comprehensive ConvNeXt Evaluation Script

Implements all evaluation metrics from the Modern Recipe:
1. Standard Metrics: Accuracy, Precision, Recall, F1, Top-1/Top-5 Error
2. Calibration: Expected Calibration Error (ECE)
3. Robustness: Corruption robustness (mCE-style)
4. Representation Analysis: Effective Receptive Field (ERF), Grad-CAM, CKA
5. Efficiency Profiling: MACs/FLOPs, Throughput, Latency, VRAM

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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

EVAL_CONFIG = {
    "model_path": "experiments_convnext_modern/best_model.pth",
    "dataset_dir": "dataset_unified/spectrograms",
    "test_split": "dataset_unified/metadata/test_split.csv",
    "output_dir": "evaluation_convnext_comprehensive",
    "image_size": 224,
    "batch_size": 32,
    
    # Evaluation options
    "eval_standard_metrics": True,
    "eval_calibration": True,
    "eval_robustness": True,
    "eval_receptive_field": True,
    "eval_gradcam": True,
    "eval_efficiency": True,
    "eval_per_layer": True,
    
    # Robustness corruption types
    "corruption_types": [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "gaussian_blur", "motion_blur", "defocus_blur",
        "brightness", "contrast", "saturate",
        "jpeg_compression", "pixelate"
    ],
    "corruption_severities": [1, 2, 3, 4, 5],
}

OUTPUT_DIR = Path(EVAL_CONFIG["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATASET
# ============================================================================

class EarthquakeDataset(Dataset):
    """Dataset for evaluation"""
    
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
        
        if 'unified_path' in row.index:
            img_path = Path("dataset_unified") / row['unified_path']
        elif 'spectrogram_file' in row.index:
            img_path = self.img_dir / row['spectrogram_file']
        else:
            img_path = self.img_dir / row.get('filename', '')
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        mag_label = self.mag_mapping.get(row['magnitude_class'], 0)
        azi_label = self.azi_mapping.get(row['azimuth_class'], 0)
        
        return image, mag_label, azi_label, str(img_path)


def get_eval_transform(image_size=224):
    """Standard evaluation transform"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
# MODEL LOADING
# ============================================================================

class ConvNeXtMultiTask(nn.Module):
    """ConvNeXt model for evaluation"""
    
    def __init__(self, model_name="convnext_tiny", num_mag_classes=4, num_azi_classes=9):
        super().__init__()
        
        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
            num_features = self.backbone.num_features
        else:
            raise ImportError("timm required for model loading")
        
        self.mag_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_mag_classes)
        )
        
        self.azi_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_azi_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.mean([-2, -1])
        return self.mag_head(features), self.azi_head(features)
    
    def get_features(self, x):
        return self.backbone(x)


def load_model(model_path, device):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model_name = config.get('model_variant', 'convnext_tiny')
    num_mag = config.get('num_mag_classes', 4)
    num_azi = config.get('num_azi_classes', 9)
    
    model = ConvNeXtMultiTask(model_name, num_mag, num_azi)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


# ============================================================================
# 1. STANDARD METRICS
# ============================================================================

@torch.no_grad()
def evaluate_standard_metrics(model, dataloader, device):
    """
    Compute standard classification metrics:
    - Accuracy, Precision, Recall, F1
    - Top-1 and Top-5 Error
    - Per-class metrics
    """
    model.eval()
    
    all_mag_preds = []
    all_mag_labels = []
    all_mag_probs = []
    all_azi_preds = []
    all_azi_labels = []
    all_azi_probs = []
    
    for images, mag_labels, azi_labels, _ in tqdm(dataloader, desc="Standard Metrics"):
        images = images.to(device)
        
        mag_out, azi_out = model(images)
        
        mag_probs = F.softmax(mag_out, dim=1)
        azi_probs = F.softmax(azi_out, dim=1)
        
        all_mag_preds.extend(torch.argmax(mag_out, dim=1).cpu().numpy())
        all_mag_labels.extend(mag_labels.numpy())
        all_mag_probs.extend(mag_probs.cpu().numpy())
        all_azi_preds.extend(torch.argmax(azi_out, dim=1).cpu().numpy())
        all_azi_labels.extend(azi_labels.numpy())
        all_azi_probs.extend(azi_probs.cpu().numpy())
    
    # Convert to arrays
    mag_preds = np.array(all_mag_preds)
    mag_labels = np.array(all_mag_labels)
    mag_probs = np.array(all_mag_probs)
    azi_preds = np.array(all_azi_preds)
    azi_labels = np.array(all_azi_labels)
    azi_probs = np.array(all_azi_probs)
    
    # Top-1 Accuracy
    mag_top1_acc = accuracy_score(mag_labels, mag_preds)
    azi_top1_acc = accuracy_score(azi_labels, azi_preds)
    
    # Top-5 Accuracy (if applicable)
    def top_k_accuracy(probs, labels, k=5):
        k = min(k, probs.shape[1])
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        return np.mean([labels[i] in top_k_preds[i] for i in range(len(labels))])
    
    mag_top5_acc = top_k_accuracy(mag_probs, mag_labels, k=min(5, mag_probs.shape[1]))
    azi_top5_acc = top_k_accuracy(azi_probs, azi_labels, k=min(5, azi_probs.shape[1]))
    
    # F1, Precision, Recall
    mag_precision, mag_recall, mag_f1, _ = precision_recall_fscore_support(
        mag_labels, mag_preds, average='weighted'
    )
    azi_precision, azi_recall, azi_f1, _ = precision_recall_fscore_support(
        azi_labels, azi_preds, average='weighted'
    )
    
    results = {
        'magnitude': {
            'top1_accuracy': float(mag_top1_acc),
            'top1_error': float(1 - mag_top1_acc),
            'top5_accuracy': float(mag_top5_acc),
            'top5_error': float(1 - mag_top5_acc),
            'precision': float(mag_precision),
            'recall': float(mag_recall),
            'f1_score': float(mag_f1),
            'predictions': mag_preds,
            'labels': mag_labels,
            'probabilities': mag_probs
        },
        'azimuth': {
            'top1_accuracy': float(azi_top1_acc),
            'top1_error': float(1 - azi_top1_acc),
            'top5_accuracy': float(azi_top5_acc),
            'top5_error': float(1 - azi_top5_acc),
            'precision': float(azi_precision),
            'recall': float(azi_recall),
            'f1_score': float(azi_f1),
            'predictions': azi_preds,
            'labels': azi_labels,
            'probabilities': azi_probs
        }
    }
    
    return results


# ============================================================================
# 2. CALIBRATION METRICS (ECE)
# ============================================================================

def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy.
    A well-calibrated model has ECE close to 0.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0.0
    bin_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
            
            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'confidence': avg_confidence,
                'accuracy': avg_accuracy,
                'count': np.sum(in_bin),
                'gap': abs(avg_accuracy - avg_confidence)
            })
    
    return ece, bin_data


def compute_mce(probs, labels, n_bins=15):
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE is the maximum gap between confidence and accuracy across all bins.
    """
    _, bin_data = compute_ece(probs, labels, n_bins)
    
    if not bin_data:
        return 0.0
    
    return max(b['gap'] for b in bin_data)


def evaluate_calibration(standard_results, output_dir):
    """Comprehensive calibration evaluation"""
    
    results = {}
    
    for task in ['magnitude', 'azimuth']:
        probs = standard_results[task]['probabilities']
        labels = standard_results[task]['labels']
        
        ece, bin_data = compute_ece(probs, labels)
        mce = compute_mce(probs, labels)
        
        # Overconfidence analysis
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        correct = predictions == labels
        
        avg_confidence_correct = np.mean(confidences[correct]) if np.any(correct) else 0
        avg_confidence_incorrect = np.mean(confidences[~correct]) if np.any(~correct) else 0
        
        results[task] = {
            'ece': float(ece),
            'mce': float(mce),
            'avg_confidence': float(np.mean(confidences)),
            'avg_confidence_correct': float(avg_confidence_correct),
            'avg_confidence_incorrect': float(avg_confidence_incorrect),
            'overconfidence_gap': float(avg_confidence_incorrect - (1 - standard_results[task]['top1_accuracy'])),
            'bin_data': bin_data
        }
        
        # Plot reliability diagram
        plot_reliability_diagram(probs, labels, output_dir, task)
    
    return results


def plot_reliability_diagram(probs, labels, output_dir, task_name):
    """Plot reliability diagram for calibration analysis"""
    n_bins = 10
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.sum(in_bin) > 0:
            bin_accuracies.append(np.mean(accuracies[in_bin]))
            bin_confidences.append(np.mean(confidences[in_bin]))
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reliability diagram
    width = 0.08
    axes[0].bar(bin_centers, bin_accuracies, width=width, alpha=0.7, 
                label='Accuracy', color='steelblue', edgecolor='black')
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    
    # Gap visualization
    for i, (bc, ba, bconf) in enumerate(zip(bin_centers, bin_accuracies, bin_confidences)):
        if bin_counts[i] > 0:
            gap_color = 'red' if bconf > ba else 'green'
            axes[0].plot([bc, bc], [ba, bconf], color=gap_color, linewidth=2, alpha=0.5)
    
    axes[0].set_xlabel('Confidence', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title(f'{task_name.capitalize()} Reliability Diagram', fontsize=14)
    axes[0].legend(loc='upper left')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    
    # Confidence histogram
    axes[1].hist(confidences, bins=n_bins, alpha=0.7, color='steelblue', 
                 edgecolor='black', density=True)
    axes[1].axvline(np.mean(confidences), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
    axes[1].set_xlabel('Confidence', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title(f'{task_name.capitalize()} Confidence Distribution', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'calibration_{task_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 3. ROBUSTNESS EVALUATION (mCE-style)
# ============================================================================

class ImageCorruption:
    """Apply various corruptions to images (ImageNet-C style)"""
    
    @staticmethod
    def gaussian_noise(image, severity=3):
        """Add Gaussian noise"""
        c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        img_array = np.array(image) / 255.0
        noise = np.random.normal(0, c, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))
    
    @staticmethod
    def shot_noise(image, severity=3):
        """Add shot (Poisson) noise"""
        c = [60, 25, 12, 5, 3][severity - 1]
        img_array = np.array(image) / 255.0
        noisy = np.clip(np.random.poisson(img_array * c) / c, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))
    
    @staticmethod
    def impulse_noise(image, severity=3):
        """Add salt and pepper noise"""
        c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
        img_array = np.array(image) / 255.0
        
        # Salt
        salt = np.random.random(img_array.shape[:2]) < c / 2
        img_array[salt] = 1
        
        # Pepper
        pepper = np.random.random(img_array.shape[:2]) < c / 2
        img_array[pepper] = 0
        
        return Image.fromarray((img_array * 255).astype(np.uint8))
    
    @staticmethod
    def gaussian_blur(image, severity=3):
        """Apply Gaussian blur"""
        c = [1, 2, 3, 4, 6][severity - 1]
        return image.filter(ImageFilter.GaussianBlur(radius=c))
    
    @staticmethod
    def motion_blur(image, severity=3):
        """Simulate motion blur"""
        c = [10, 15, 15, 15, 20][severity - 1]
        # Simplified motion blur using box blur
        return image.filter(ImageFilter.BoxBlur(radius=c // 3))
    
    @staticmethod
    def defocus_blur(image, severity=3):
        """Apply defocus blur"""
        c = [3, 4, 6, 8, 10][severity - 1]
        return image.filter(ImageFilter.GaussianBlur(radius=c))
    
    @staticmethod
    def brightness(image, severity=3):
        """Adjust brightness"""
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1 + c)
    
    @staticmethod
    def contrast(image, severity=3):
        """Adjust contrast"""
        c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(c)
    
    @staticmethod
    def saturate(image, severity=3):
        """Adjust saturation"""
        c = [0.3, 0.1, 2.0, 5.0, 20.0][severity - 1]
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(c)
    
    @staticmethod
    def jpeg_compression(image, severity=3):
        """Apply JPEG compression artifacts"""
        c = [25, 18, 15, 10, 7][severity - 1]
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=c)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
    
    @staticmethod
    def pixelate(image, severity=3):
        """Pixelate image"""
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
        w, h = image.size
        small = image.resize((int(w * c), int(h * c)), Image.NEAREST)
        return small.resize((w, h), Image.NEAREST)


class CorruptedDataset(Dataset):
    """Dataset with corruption applied"""
    
    def __init__(self, base_dataset, corruption_type, severity, transform):
        self.base_dataset = base_dataset
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform
        self.corruption_fn = getattr(ImageCorruption, corruption_type)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        row = self.base_dataset.metadata.iloc[idx]
        
        if 'unified_path' in row.index:
            img_path = Path("dataset_unified") / row['unified_path']
        elif 'spectrogram_file' in row.index:
            img_path = self.base_dataset.img_dir / row['spectrogram_file']
        else:
            img_path = self.base_dataset.img_dir / row.get('filename', '')
        
        image = Image.open(img_path).convert('RGB')
        
        # Apply corruption
        image = self.corruption_fn(image, self.severity)
        
        if self.transform:
            image = self.transform(image)
        
        mag_label = self.base_dataset.mag_mapping.get(row['magnitude_class'], 0)
        azi_label = self.base_dataset.azi_mapping.get(row['azimuth_class'], 0)
        
        return image, mag_label, azi_label


@torch.no_grad()
def evaluate_robustness(model, base_dataset, transform, device, 
                        corruption_types, severities, batch_size=32):
    """
    Evaluate model robustness to corruptions.
    
    Returns mean Corruption Error (mCE) style metrics.
    """
    model.eval()
    
    results = {
        'corruption_errors': {},
        'severity_breakdown': {}
    }
    
    for corruption in tqdm(corruption_types, desc="Evaluating Robustness"):
        corruption_errors = []
        severity_results = []
        
        for severity in severities:
            # Create corrupted dataset
            corrupted_dataset = CorruptedDataset(
                base_dataset, corruption, severity, transform
            )
            corrupted_loader = DataLoader(
                corrupted_dataset, batch_size=batch_size, 
                shuffle=False, num_workers=2
            )
            
            # Evaluate
            mag_correct = 0
            total = 0
            
            for images, mag_labels, azi_labels in corrupted_loader:
                images = images.to(device)
                mag_out, _ = model(images)
                mag_pred = torch.argmax(mag_out, dim=1)
                mag_correct += (mag_pred.cpu() == mag_labels).sum().item()
                total += images.size(0)
            
            error = 1 - (mag_correct / total)
            corruption_errors.append(error)
            severity_results.append({
                'severity': severity,
                'error': error,
                'accuracy': 1 - error
            })
        
        # Mean error across severities
        mean_error = np.mean(corruption_errors)
        results['corruption_errors'][corruption] = mean_error
        results['severity_breakdown'][corruption] = severity_results
    
    # Compute mCE (mean Corruption Error)
    results['mCE'] = np.mean(list(results['corruption_errors'].values()))
    
    return results


def plot_robustness_results(robustness_results, output_dir):
    """Visualize robustness evaluation results"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart of corruption errors
    corruptions = list(robustness_results['corruption_errors'].keys())
    errors = list(robustness_results['corruption_errors'].values())
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(corruptions)))
    bars = axes[0].bar(range(len(corruptions)), errors, color=colors, edgecolor='black')
    axes[0].set_xticks(range(len(corruptions)))
    axes[0].set_xticklabels(corruptions, rotation=45, ha='right')
    axes[0].set_ylabel('Error Rate')
    axes[0].set_title(f'Corruption Robustness (mCE: {robustness_results["mCE"]:.4f})')
    axes[0].axhline(robustness_results['mCE'], color='red', linestyle='--', 
                    label=f'mCE: {robustness_results["mCE"]:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Heatmap of severity breakdown
    severity_data = []
    for corruption in corruptions:
        severity_errors = [s['error'] for s in robustness_results['severity_breakdown'][corruption]]
        severity_data.append(severity_errors)
    
    severity_matrix = np.array(severity_data)
    im = axes[1].imshow(severity_matrix, cmap='RdYlGn_r', aspect='auto')
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(['Sev 1', 'Sev 2', 'Sev 3', 'Sev 4', 'Sev 5'])
    axes[1].set_yticks(range(len(corruptions)))
    axes[1].set_yticklabels(corruptions)
    axes[1].set_title('Error Rate by Corruption Type and Severity')
    plt.colorbar(im, ax=axes[1], label='Error Rate')
    
    # Add text annotations
    for i in range(len(corruptions)):
        for j in range(5):
            text = axes[1].text(j, i, f'{severity_matrix[i, j]:.2f}',
                               ha='center', va='center', color='black', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 4. EFFECTIVE RECEPTIVE FIELD (ERF) ANALYSIS
# ============================================================================

def compute_effective_receptive_field(model, input_size=(1, 3, 224, 224), device='cuda'):
    """
    Compute Effective Receptive Field (ERF).
    
    ERF shows which input pixels actually influence the output.
    ConvNeXt with 7x7 kernels should have a wider ERF than ResNet with 3x3.
    """
    model.eval()
    
    # Create input with gradient tracking
    x = torch.zeros(*input_size, requires_grad=True, device=device)
    
    # Forward pass
    output = model(x)
    if isinstance(output, tuple):
        output = output[0]
    
    # Backward from center of output
    grad_output = torch.zeros_like(output)
    grad_output[0, 0] = 1  # Gradient from first class
    
    output.backward(gradient=grad_output)
    
    # Get gradient magnitude
    erf = x.grad.abs().squeeze().cpu().numpy()
    
    # Average across channels
    if erf.ndim == 3:
        erf = erf.mean(axis=0)
    
    # Normalize
    erf = erf / erf.max()
    
    return erf


def plot_effective_receptive_field(erf, output_dir):
    """Visualize Effective Receptive Field"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ERF heatmap
    im = axes[0].imshow(erf, cmap='hot')
    axes[0].set_title('Effective Receptive Field (ERF)')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    
    # ERF with contours
    axes[1].imshow(erf, cmap='hot')
    contours = axes[1].contour(erf, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='white', linewidths=1)
    axes[1].clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    axes[1].set_title('ERF with Contours')
    axes[1].axis('off')
    
    # Cross-section through center
    center = erf.shape[0] // 2
    axes[2].plot(erf[center, :], 'b-', linewidth=2, label='Horizontal')
    axes[2].plot(erf[:, center], 'r-', linewidth=2, label='Vertical')
    axes[2].set_xlabel('Pixel Position')
    axes[2].set_ylabel('ERF Magnitude')
    axes[2].set_title('ERF Cross-Section (Center)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'effective_receptive_field.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compute ERF statistics
    # Effective radius (where ERF > 0.1)
    threshold = 0.1
    effective_area = np.sum(erf > threshold)
    effective_radius = np.sqrt(effective_area / np.pi)
    
    # Concentration (how focused is the ERF)
    center_region = erf[erf.shape[0]//4:3*erf.shape[0]//4, 
                        erf.shape[1]//4:3*erf.shape[1]//4]
    concentration = center_region.sum() / erf.sum()
    
    return {
        'effective_area': int(effective_area),
        'effective_radius': float(effective_radius),
        'concentration': float(concentration),
        'max_value': float(erf.max()),
        'mean_value': float(erf.mean())
    }


# ============================================================================
# 5. GRAD-CAM VISUALIZATION
# ============================================================================

class GradCAM:
    """Grad-CAM for ConvNeXt visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
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
        
        # Resize
        cam = F.interpolate(cam, size=input_tensor.shape[2:], 
                           mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy(), target_class


def generate_gradcam_samples(model, dataset, device, output_dir, num_samples=10):
    """Generate Grad-CAM visualizations for sample images"""
    
    # Get target layer
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'stages'):
            target_layer = model.backbone.stages[-1]
        else:
            # For timm models
            target_layer = model.backbone.stages[-1][-1] if hasattr(model.backbone.stages[-1], '__getitem__') else model.backbone.stages[-1]
    
    gradcam = GradCAM(model, target_layer)
    
    # Sample indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    for i, idx in enumerate(indices):
        image, mag_label, azi_label, img_path = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        
        # Generate Grad-CAM
        cam, pred_class = gradcam.generate(image_tensor)
        
        # Original image
        img_display = image.permute(1, 2, 0).numpy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f'Original\nTrue: {mag_label}')
        axes[i, 0].axis('off')
        
        # Grad-CAM heatmap
        axes[i, 1].imshow(cam, cmap='jet')
        axes[i, 1].set_title(f'Grad-CAM\nPred: {pred_class}')
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(img_display)
        axes[i, 2].imshow(cam, cmap='jet', alpha=0.5)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')
        
        # Attention focus
        threshold = 0.5
        attention_mask = cam > threshold
        axes[i, 3].imshow(img_display)
        axes[i, 3].contour(attention_mask, colors='red', linewidths=2)
        axes[i, 3].set_title(f'Focus Region (>{threshold})')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradcam_samples.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 6. EFFICIENCY PROFILING
# ============================================================================

def profile_efficiency(model, input_size=(1, 3, 224, 224), device='cuda', 
                       warmup_iterations=50, test_iterations=200):
    """
    Comprehensive efficiency profiling:
    - MACs/FLOPs
    - Throughput (images/sec)
    - Latency (ms/image)
    - Peak VRAM usage
    - Per-layer latency breakdown
    """
    model.eval()
    results = {}
    
    dummy_input = torch.randn(*input_size).to(device)
    
    # ========== Parameter Count ==========
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results['total_params'] = total_params
    results['trainable_params'] = trainable_params
    results['params_mb'] = total_params * 4 / (1024 ** 2)  # Assuming float32
    
    # ========== FLOPs/MACs ==========
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        flops = FlopCountAnalysis(model, dummy_input)
        results['flops'] = flops.total()
        results['gflops'] = flops.total() / 1e9
        results['macs'] = flops.total() / 2  # MACs ≈ FLOPs / 2
        results['gmacs'] = results['macs'] / 1e9
    except ImportError:
        logger.warning("fvcore not available for FLOPs calculation")
        results['flops'] = None
        results['gflops'] = None
    
    # ========== Warm-up ==========
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # ========== Latency Measurement ==========
    latencies = []
    
    with torch.no_grad():
        for _ in range(test_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    results['latency_mean_ms'] = np.mean(latencies)
    results['latency_std_ms'] = np.std(latencies)
    results['latency_min_ms'] = np.min(latencies)
    results['latency_max_ms'] = np.max(latencies)
    results['latency_p50_ms'] = np.percentile(latencies, 50)
    results['latency_p95_ms'] = np.percentile(latencies, 95)
    results['latency_p99_ms'] = np.percentile(latencies, 99)
    
    # ========== Throughput ==========
    results['throughput_img_per_sec'] = 1000 / results['latency_mean_ms']
    
    # ========== Batch Throughput ==========
    batch_sizes = [1, 4, 8, 16, 32]
    batch_throughputs = {}
    
    for bs in batch_sizes:
        try:
            batch_input = torch.randn(bs, *input_size[1:]).to(device)
            
            # Warm-up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(batch_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # Measure
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(50):
                    _ = model(batch_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            batch_throughputs[bs] = (bs * 50) / elapsed
            
        except RuntimeError:  # OOM
            batch_throughputs[bs] = None
            break
    
    results['batch_throughputs'] = batch_throughputs
    
    # ========== VRAM Usage ==========
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        results['peak_vram_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        results['peak_vram_gb'] = results['peak_vram_mb'] / 1024
        
        # Memory for different batch sizes
        vram_by_batch = {}
        for bs in [1, 4, 8, 16, 32]:
            try:
                torch.cuda.reset_peak_memory_stats()
                batch_input = torch.randn(bs, *input_size[1:]).to(device)
                with torch.no_grad():
                    _ = model(batch_input)
                vram_by_batch[bs] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            except RuntimeError:
                vram_by_batch[bs] = None
                break
        
        results['vram_by_batch_mb'] = vram_by_batch
    
    return results


def profile_per_layer_latency(model, input_size=(1, 3, 224, 224), device='cuda'):
    """Profile latency for each layer/module"""
    
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    
    layer_times = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            # The actual computation alre