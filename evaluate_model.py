"""
Comprehensive evaluation script for Multi-Task Earthquake CNN
Generates scientific metrics, confusion matrices, and visualizations
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import torch

from multi_task_cnn_model import create_model
from earthquake_dataset import create_dataloaders

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, exp_dir, checkpoint='best_model.pth'):
        """
        Initialize evaluator
        
        Args:
            exp_dir: Experiment directory
            checkpoint: Checkpoint filename to load
        """
        self.exp_dir = exp_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(os.path.join(exp_dir, 'config.json'), 'r') as f:
            self.config = json.load(f)
        
        # Create dataloaders
        _, _, self.test_loader, self.dataset_info = create_dataloaders(
            dataset_dir=self.config['dataset_dir'],
            batch_size=self.config['batch_size'],
            val_split=self.config.get('val_split', 0.2),
            test_split=self.config.get('test_split', 0.1),
            num_workers=0,
            seed=self.config.get('seed', 42)
        )
        
        # Load model
        self.model, _ = create_model(self.config)
        checkpoint_path = os.path.join(exp_dir, checkpoint)
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from: {checkpoint_path}")
        
        # Class names
        self.magnitude_classes = self.dataset_info['magnitude_classes']
        self.azimuth_classes = self.dataset_info['azimuth_classes']
    
    def predict(self):
        """Get predictions on test set"""
        mag_preds = []
        az_preds = []
        mag_probs = []
        az_probs = []
        mag_true = []
        az_true = []
        
        with torch.no_grad():
            for images, mag_labels, az_labels in self.test_loader:
                images = images.to(self.device)
                
                mag_logits, az_logits = self.model(images)
                
                mag_prob = torch.softmax(mag_logits, dim=1)
                az_prob = torch.softmax(az_logits, dim=1)
                
                mag_preds.extend(mag_logits.argmax(dim=1).cpu().numpy())
                az_preds.extend(az_logits.argmax(dim=1).cpu().numpy())
                
                mag_probs.extend(mag_prob.cpu().numpy())
                az_probs.extend(az_prob.cpu().numpy())
                
                mag_true.extend(mag_labels.numpy())
                az_true.extend(az_labels.numpy())
        
        return (np.array(mag_preds), np.array(az_preds),
                np.array(mag_probs), np.array(az_probs),
                np.array(mag_true), np.array(az_true))
    
    def evaluate(self):
        """Comprehensive evaluation"""
        logger.info("Evaluating model...")
        
        # Get predictions
        mag_preds, az_preds, mag_probs, az_probs, mag_true, az_true = self.predict()
        
        # Compute metrics
        results = {}
        
        # Magnitude metrics
        results['magnitude'] = self._compute_metrics(
            mag_true, mag_preds, mag_probs, self.magnitude_classes, 'Magnitude'
        )
        
        # Azimuth metrics
        results['azimuth'] = self._compute_metrics(
            az_true, az_preds, az_probs, self.azimuth_classes, 'Azimuth'
        )
        
        # Save results
        self._save_results(results)
        
        # Generate visualizations
        self._plot_confusion_matrices(mag_true, mag_preds, az_true, az_preds)
        self._plot_training_history()
        self._plot_roc_curves(mag_true, mag_probs, az_true, az_probs)
        
        logger.info("Evaluation completed!")
        
        return results

    
    def _compute_metrics(self, y_true, y_pred, y_prob, class_names, task_name):
        """Compute comprehensive metrics for a task"""
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        # Classification report
        report = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'per_class': {
                class_names[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support_per_class[i])
                }
                for i in range(len(class_names))
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        logger.info(f"\n{task_name} Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        return metrics
    
    def _save_results(self, results):
        """Save evaluation results"""
        results_path = os.path.join(self.exp_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to: {results_path}")
        
        # Save classification reports
        with open(os.path.join(self.exp_dir, 'classification_report.txt'), 'w') as f:
            f.write("="*80 + "\n")
            f.write("MAGNITUDE CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(results['magnitude']['classification_report'])
            f.write("\n\n")
            f.write("="*80 + "\n")
            f.write("AZIMUTH CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(results['azimuth']['classification_report'])
    
    def _plot_confusion_matrices(self, mag_true, mag_preds, az_true, az_preds):
        """Plot confusion matrices"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Magnitude confusion matrix
        cm_mag = confusion_matrix(mag_true, mag_preds)
        sns.heatmap(cm_mag, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.magnitude_classes,
                   yticklabels=self.magnitude_classes,
                   ax=axes[0])
        axes[0].set_title('Magnitude Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Azimuth confusion matrix
        cm_az = confusion_matrix(az_true, az_preds)
        sns.heatmap(cm_az, annot=True, fmt='d', cmap='Greens',
                   xticklabels=self.azimuth_classes,
                   yticklabels=self.azimuth_classes,
                   ax=axes[1])
        axes[1].set_title('Azimuth Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, 'confusion_matrices.png'), dpi=300)
        plt.close()
        logger.info("Confusion matrices saved")
    
    def _plot_training_history(self):
        """Plot training history"""
        history_path = os.path.join(self.exp_dir, 'training_history.csv')
        if not os.path.exists(history_path):
            logger.warning("Training history not found")
            return
        
        history = pd.read_csv(history_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Magnitude accuracy
        axes[0, 1].plot(history['train_mag_acc'], label='Train', linewidth=2)
        axes[0, 1].plot(history['val_mag_acc'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Magnitude Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Azimuth accuracy
        axes[1, 0].plot(history['train_az_acc'], label='Train', linewidth=2)
        axes[1, 0].plot(history['val_az_acc'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Azimuth Accuracy', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(history['lr'], linewidth=2, color='red')
        axes[1, 1].set_title('Learning Rate', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, 'training_history.png'), dpi=300)
        plt.close()
        logger.info("Training history plots saved")
    
    def _plot_roc_curves(self, mag_true, mag_probs, az_true, az_probs):
        """Plot ROC curves for multi-class"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Magnitude ROC
        mag_true_bin = label_binarize(mag_true, classes=range(len(self.magnitude_classes)))
        for i, class_name in enumerate(self.magnitude_classes):
            fpr, tpr, _ = roc_curve(mag_true_bin[:, i], mag_probs[:, i])
            roc_auc = auc(fpr, tpr)
            axes[0].plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})', linewidth=2)
        
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0].set_title('Magnitude ROC Curves', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        # Azimuth ROC
        az_true_bin = label_binarize(az_true, classes=range(len(self.azimuth_classes)))
        for i, class_name in enumerate(self.azimuth_classes):
            fpr, tpr, _ = roc_curve(az_true_bin[:, i], az_probs[:, i])
            roc_auc = auc(fpr, tpr)
            axes[1].plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})', linewidth=2)
        
        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[1].set_title('Azimuth ROC Curves', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend(loc='lower right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, 'roc_curves.png'), dpi=300)
        plt.close()
        logger.info("ROC curves saved")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Multi-Task Earthquake CNN')
    parser.add_argument('--exp-dir', required=True, help='Experiment directory')
    parser.add_argument('--checkpoint', default='best_model.pth', help='Checkpoint to load')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.exp_dir, args.checkpoint)
    
    # Evaluate
    results = evaluator.evaluate()
    
    print(f"\n[OK] Evaluation completed!")
    print(f"Results saved to: {args.exp_dir}")
    print(f"\nMagnitude Accuracy: {results['magnitude']['accuracy']:.4f}")
    print(f"Azimuth Accuracy: {results['azimuth']['accuracy']:.4f}")


if __name__ == '__main__':
    main()
