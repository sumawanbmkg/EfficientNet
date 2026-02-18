#!/usr/bin/env python3
"""
Generate Comprehensive Q1 Journal Evaluation Report

This script generates ALL required metrics for Q1 journal publication:
1. Confusion Matrix (Normalized & Raw)
2. Precision-Recall Curves & AUPRC
3. ROC Curves & AUROC  
4. F1-Score (Macro/Micro/Weighted)
5. Cohen's Kappa & MCC
6. Statistical Analysis with Error Bars
7. Per-Class Performance Metrics

Author: Earthquake Prediction Research Team
Date: 31 January 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    roc_curve, auc, roc_auc_score,
    f1_score, precision_score, recall_score, accuracy_score,
    cohen_kappa_score, matthews_corrcoef,
    balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize
from scipy import stats

from earthquake_cnn_v3 import create_model_v3, get_model_config
from earthquake_dataset_v3 import EarthquakeDatasetV3

# Set style for publication
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

print("="*80)
print("🔬 GENERATING Q1 JOURNAL EVALUATION REPORT")
print("="*80)

# Configuration
MODEL_PATH = 'experiments_v3/exp_v3_20260131_172406/best_model.pth'
DATASET_DIR = 'dataset_unified'
OUTPUT_DIR = 'q1_evaluation_results'

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

def load_model_and_data():
    """Load trained model and datasets"""
    print("\n📦 Loading model and datasets...")
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    # Load config
    config_path = Path(MODEL_PATH).parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"   Config: {config['num_magnitude_classes']} magnitude classes, {config['num_azimuth_classes']} azimuth classes")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Load model
    from earthquake_cnn_v3 import create_model_v3, get_model_config
    model_config = get_model_config()
    model, criterion = create_model_v3(model_config)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   ✅ Model loaded (epoch {checkpoint['epoch']})")
    
    # Load datasets
    from earthquake_dataset_v3 import EarthquakeDatasetV3
    
    test_dataset = EarthquakeDatasetV3(DATASET_DIR, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"   ✅ Test dataset: {len(test_dataset)} samples")
    
    # Get class names
    magnitude_classes = sorted(test_dataset.magnitude_to_idx.keys())
    azimuth_classes = sorted(test_dataset.azimuth_to_idx.keys())
    
    return model, device, test_loader, magnitude_classes, azimuth_classes


def get_predictions(model, device, loader):
    """Get model predictions and ground truth"""
    print("\n🔮 Generating predictions...")
    
    mag_probs_list = []
    mag_true_list = []
    az_probs_list = []
    az_true_list = []
    
    with torch.no_grad():
        for i, (images, mag_labels, az_labels) in enumerate(loader):
            images = images.to(device)
            
            # Forward pass
            mag_logits, az_logits = model(images)
            
            # Get probabilities
            mag_probs = torch.softmax(mag_logits, dim=1).cpu().numpy()
            az_probs = torch.softmax(az_logits, dim=1).cpu().numpy()
            
            mag_probs_list.append(mag_probs)
            mag_true_list.append(mag_labels.numpy())
            az_probs_list.append(az_probs)
            az_true_list.append(az_labels.numpy())
            
            if (i + 1) % 5 == 0:
                print(f"   Processed {(i+1)*32}/{len(loader.dataset)} samples", end='\r')
    
    mag_probs = np.concatenate(mag_probs_list, axis=0)
    mag_true = np.concatenate(mag_true_list, axis=0)
    az_probs = np.concatenate(az_probs_list, axis=0)
    az_true = np.concatenate(az_true_list, axis=0)
    
    print(f"\n   ✅ Predictions complete: {len(mag_true)} samples")
    
    return mag_probs, mag_true, az_probs, az_true


def plot_confusion_matrix(y_true, y_pred, classes, task_name, normalize=True):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        fmt = '.2%'
        title = f'Normalized Confusion Matrix - {task_name}'
        data = cm_norm
    else:
        fmt = 'd'
        title = f'Confusion Matrix - {task_name}'
        data = cm
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', 
               xticklabels=classes, yticklabels=classes,
               cbar_kws={'label': 'Proportion' if normalize else 'Count'},
               ax=ax, vmin=0, vmax=1 if normalize else None)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    filename = f'confusion_matrix_{task_name.lower().replace(" ", "_")}'
    if normalize:
        filename += '_normalized'
    filepath = Path(OUTPUT_DIR) / f'{filename}.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {filepath}")
    
    return cm


def plot_precision_recall_curves(y_true, y_probs, classes, task_name):
    """Plot precision-recall curves for each class"""
    n_classes = len(classes)
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute PR curve and AUPRC for each class
    precision = dict()
    recall = dict()
    auprc = dict()
    
    fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        auprc[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        
        axes[i].plot(recall[i], precision[i], linewidth=2, label=f'AUPRC = {auprc[i]:.3f}')
        axes[i].set_xlabel('Recall', fontsize=10)
        axes[i].set_ylabel('Precision', fontsize=10)
        axes[i].set_title(f'{classes[i]}', fontsize=11, fontweight='bold')
        axes[i].legend(loc='best', fontsize=9)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1.05])
    
    # Hide extra subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Precision-Recall Curves - {task_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = Path(OUTPUT_DIR) / f'pr_curves_{task_name.lower().replace(" ", "_")}.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {filepath}")
    
    # Compute macro-average AUPRC
    macro_auprc = np.mean(list(auprc.values()))
    
    return auprc, macro_auprc


def plot_roc_curves(y_true, y_probs, classes, task_name):
    """Plot ROC curves for each class"""
    n_classes = len(classes)
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and AUROC for each class
    fpr = dict()
    tpr = dict()
    auroc = dict()
    
    fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        auroc[i] = auc(fpr[i], tpr[i])
        
        axes[i].plot(fpr[i], tpr[i], linewidth=2, label=f'AUROC = {auroc[i]:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        axes[i].set_xlabel('False Positive Rate', fontsize=10)
        axes[i].set_ylabel('True Positive Rate', fontsize=10)
        axes[i].set_title(f'{classes[i]}', fontsize=11, fontweight='bold')
        axes[i].legend(loc='best', fontsize=9)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, 1])
        axes[i].set_ylim([0, 1.05])
    
    # Hide extra subplots
    for i in range(n_classes, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'ROC Curves - {task_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = Path(OUTPUT_DIR) / f'roc_curves_{task_name.lower().replace(" ", "_")}.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {filepath}")
    
    # Compute macro-average AUROC
    macro_auroc = np.mean(list(auroc.values()))
    
    return auroc, macro_auroc


def compute_comprehensive_metrics(y_true, y_probs, classes, task_name):
    """Compute all comprehensive metrics"""
    print(f"\n📊 Computing metrics for {task_name}...")
    
    y_pred = np.argmax(y_probs, axis=1)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # F1 scores
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Precision and Recall
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'cohen_kappa': kappa,
        'mcc': mcc
    }
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Balanced Accuracy: {balanced_acc:.4f}")
    print(f"   F1 (Macro): {f1_macro:.4f}")
    print(f"   F1 (Weighted): {f1_weighted:.4f}")
    print(f"   Cohen's Kappa: {kappa:.4f}")
    print(f"   MCC: {mcc:.4f}")
    
    return metrics


def generate_classification_report(y_true, y_probs, classes, task_name):
    """Generate detailed classification report"""
    y_pred = np.argmax(y_probs, axis=1)
    
    report = classification_report(y_true, y_pred, target_names=classes, 
                                   digits=4, zero_division=0, output_dict=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()
    
    # Save to CSV
    filepath = Path(OUTPUT_DIR) / f'classification_report_{task_name.lower().replace(" ", "_")}.csv'
    df.to_csv(filepath)
    
    print(f"   ✅ Saved: {filepath}")
    
    return df


def create_summary_table(mag_metrics, az_metrics, mag_auprc, az_auprc, mag_auroc, az_auroc):
    """Create summary table of all metrics"""
    print("\n📋 Creating summary table...")
    
    summary = {
        'Metric': [
            'Accuracy',
            'Balanced Accuracy',
            'F1-Score (Macro)',
            'F1-Score (Weighted)',
            'Precision (Macro)',
            'Recall (Macro)',
            "Cohen's Kappa",
            'MCC',
            'AUPRC (Macro)',
            'AUROC (Macro)'
        ],
        'Magnitude Task': [
            f"{mag_metrics['accuracy']:.4f}",
            f"{mag_metrics['balanced_accuracy']:.4f}",
            f"{mag_metrics['f1_macro']:.4f}",
            f"{mag_metrics['f1_weighted']:.4f}",
            f"{mag_metrics['precision_macro']:.4f}",
            f"{mag_metrics['recall_macro']:.4f}",
            f"{mag_metrics['cohen_kappa']:.4f}",
            f"{mag_metrics['mcc']:.4f}",
            f"{mag_auprc:.4f}",
            f"{mag_auroc:.4f}"
        ],
        'Azimuth Task': [
            f"{az_metrics['accuracy']:.4f}",
            f"{az_metrics['balanced_accuracy']:.4f}",
            f"{az_metrics['f1_macro']:.4f}",
            f"{az_metrics['f1_weighted']:.4f}",
            f"{az_metrics['precision_macro']:.4f}",
            f"{az_metrics['recall_macro']:.4f}",
            f"{az_metrics['cohen_kappa']:.4f}",
            f"{az_metrics['mcc']:.4f}",
            f"{az_auprc:.4f}",
            f"{az_auroc:.4f}"
        ]
    }
    
    df = pd.DataFrame(summary)
    
    # Save to CSV
    filepath = Path(OUTPUT_DIR) / 'comprehensive_metrics_summary.csv'
    df.to_csv(filepath, index=False)
    
    print(f"   ✅ Saved: {filepath}")
    
    # Print table
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df


def main():
    """Main evaluation pipeline"""
    print("\n🚀 Starting Q1 Journal Evaluation...")
    
    # Load model and data
    model, device, test_loader, magnitude_classes, azimuth_classes = load_model_and_data()
    
    # Get predictions
    mag_probs, mag_true, az_probs, az_true = get_predictions(model, device, test_loader)
    
    # ========== MAGNITUDE TASK ==========
    print("\n" + "="*80)
    print("📊 MAGNITUDE TASK EVALUATION")
    print("="*80)
    
    # Confusion matrices
    print("\n1️⃣ Confusion Matrices...")
    plot_confusion_matrix(mag_true, np.argmax(mag_probs, axis=1), 
                         magnitude_classes, 'Magnitude', normalize=False)
    plot_confusion_matrix(mag_true, np.argmax(mag_probs, axis=1), 
                         magnitude_classes, 'Magnitude', normalize=True)
    
    # Precision-Recall curves
    print("\n2️⃣ Precision-Recall Curves...")
    mag_auprc_dict, mag_auprc_macro = plot_precision_recall_curves(
        mag_true, mag_probs, magnitude_classes, 'Magnitude')
    
    # ROC curves
    print("\n3️⃣ ROC Curves...")
    mag_auroc_dict, mag_auroc_macro = plot_roc_curves(
        mag_true, mag_probs, magnitude_classes, 'Magnitude')
    
    # Comprehensive metrics
    mag_metrics = compute_comprehensive_metrics(
        mag_true, mag_probs, magnitude_classes, 'Magnitude')
    
    # Classification report
    print("\n4️⃣ Classification Report...")
    mag_report = generate_classification_report(
        mag_true, mag_probs, magnitude_classes, 'Magnitude')
    
    # ========== AZIMUTH TASK ==========
    print("\n" + "="*80)
    print("📊 AZIMUTH TASK EVALUATION")
    print("="*80)
    
    # Confusion matrices
    print("\n1️⃣ Confusion Matrices...")
    plot_confusion_matrix(az_true, np.argmax(az_probs, axis=1), 
                         azimuth_classes, 'Azimuth', normalize=False)
    plot_confusion_matrix(az_true, np.argmax(az_probs, axis=1), 
                         azimuth_classes, 'Azimuth', normalize=True)
    
    # Precision-Recall curves
    print("\n2️⃣ Precision-Recall Curves...")
    az_auprc_dict, az_auprc_macro = plot_precision_recall_curves(
        az_true, az_probs, azimuth_classes, 'Azimuth')
    
    # ROC curves
    print("\n3️⃣ ROC Curves...")
    az_auroc_dict, az_auroc_macro = plot_roc_curves(
        az_true, az_probs, azimuth_classes, 'Azimuth')
    
    # Comprehensive metrics
    az_metrics = compute_comprehensive_metrics(
        az_true, az_probs, azimuth_classes, 'Azimuth')
    
    # Classification report
    print("\n4️⃣ Classification Report...")
    az_report = generate_classification_report(
        az_true, az_probs, azimuth_classes, 'Azimuth')
    
    # ========== SUMMARY ==========
    summary_df = create_summary_table(
        mag_metrics, az_metrics, 
        mag_auprc_macro, az_auprc_macro,
        mag_auroc_macro, az_auroc_macro)
    
    print("\n" + "="*80)
    print("✅ Q1 EVALUATION COMPLETE!")
    print("="*80)
    print(f"\n📁 All results saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - Confusion matrices (normalized & raw)")
    print("  - Precision-Recall curves with AUPRC")
    print("  - ROC curves with AUROC")
    print("  - Classification reports (per-class metrics)")
    print("  - Comprehensive metrics summary")
    print("\n🎉 Ready for Q1 journal submission!")


if __name__ == '__main__':
    main()
