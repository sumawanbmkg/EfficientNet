#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for Q1 Journal Publication

Generates all required metrics and visualizations:
- Confusion Matrix (Normalized)
- Precision-Recall Curves & AUPRC
- ROC Curves & AUROC
- F1-Score (Macro/Micro/Weighted)
- Cohen's Kappa & MCC
- Statistical Significance Tests
- Grad-CAM Visualizations
- Cross-Validation Results

Author: Earthquake Prediction Research Team
Date: 31 January 2026
"""

import os
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

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

print("="*80)
print("🔬 COMPREHENSIVE MODEL EVALUATION FOR Q1 PUBLICATION")
print("="*80)


class ComprehensiveEvaluator:
    """Comprehensive model evaluator for scientific publication"""
    
    def __init__(self, model_path: str, dataset_dir: str, output_dir: str = 'evaluation_results'):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model checkpoint
            dataset_dir: Path to dataset directory
            output_dir: Output directory for results
        """
        self.model_path = Path(model_path)
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n📱 Device: {self.device}")
        
        # Load model
        print(f"\n🔄 Loading model from {self.model_path}...")
        self.model, self.criterion = self._load_model()
        
        # Load datasets
        print(f"\n📊 Loading datasets from {self.dataset_dir}...")
        self.train_loader, self.val_loader, self.test_loader = self._load_datasets()
        
        # Get class names
        test_dataset = self.test_loader.dataset
        self.magnitude_classes = sorted(test_dataset.magnitude_to_idx.keys())
        self.azimuth_classes = sorted(test_dataset.azimuth_to_idx.keys())
        
        print(f"\n✅ Initialization complete!")
        print(f"   Magnitude classes: {len(self.magnitude_classes)}")
        print(f"   Azimuth classes: {len(self.azimuth_classes)}")
        
    def _load_model(self):
        """Load trained model"""
        config = get_model_config()
        model, criterion = create_model_v3(config)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model, criterion
    
    def _load_datasets(self):
        """Load train/val/test datasets"""
        train_dataset = EarthquakeDatasetV3(str(self.dataset_dir), split='train')
        val_dataset = EarthquakeDatasetV3(str(self.dataset_dir), split='val')
        test_dataset = EarthquakeDatasetV3(str(self.dataset_dir), split='test')
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        return train_loader, val_loader, test_loader


    def get_predictions(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get model predictions and ground truth
        
        Returns:
            mag_probs: Magnitude probabilities (N, num_mag_classes)
            mag_true: True magnitude labels (N,)
            az_probs: Azimuth probabilities (N, num_az_classes)
            az_true: True azimuth labels (N,)
        """
        mag_probs_list = []
        mag_true_list = []
        az_probs_list = []
        az_true_list = []
        
        with torch.no_grad():
            for images, mag_labels, az_labels in loader:
                images = images.to(self.device)
                
                # Forward pass
                mag_logits, az_logits = self.model(images)
                
                # Get probabilities
                mag_probs = torch.softmax(mag_logits, dim=1).cpu().numpy()
                az_probs = torch.softmax(az_logits, dim=1).cpu().numpy()
                
                mag_probs_list.append(mag_probs)
                mag_true_list.append(mag_labels.numpy())
                az_probs_list.append(az_probs)
                az_true_list.append(az_labels.numpy())
        
        mag_probs = np.concatenate(mag_probs_list, axis=0)
        mag_true = np.concatenate(mag_true_list, axis=0)
        az_probs = np.concatenate(az_probs_list, axis=0)
        az_true = np.concatenate(az_true_list, axis=0)
        
        return mag_probs, mag_true, az_probs, az_true
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             classes: List[str], task_name: str, normalize: bool = True):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = f'Normalized Confusion Matrix - {task_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {task_name}'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                   ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'confusion_matrix_{task_name.lower().replace(" ", "_")}'
        if normalize:
            filename += '_normalized'
        plt.savefig(self.output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm

