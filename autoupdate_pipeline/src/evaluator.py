"""
Model Evaluator Module

Evaluates model performance on benchmark test set.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, matthews_corrcoef,
    roc_auc_score
)

from .utils import load_config, format_metrics_table

logger = logging.getLogger("autoupdate_pipeline.evaluator")


class MultiTaskConvNeXt(nn.Module):
    """ConvNeXt model for evaluation."""
    
    def __init__(self, num_mag_classes: int, num_azi_classes: int):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=None)
        num_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()
        
        self.mag_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Flatten(start_dim=1),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_mag_classes)
        )
        
        self.azi_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Flatten(start_dim=1),
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


class ModelEvaluator:
    """
    Evaluates model performance on benchmark test set.
    
    Metrics computed:
    - Accuracy
    - F1 Score (macro, weighted)
    - Precision
    - Recall
    - MCC (Matthews Correlation Coefficient)
    - AUC-ROC
    - Confusion Matrix
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize evaluator."""
        self.config = config or load_config()
        self.eval_config = self.config.get('evaluation', {})
        
        self.base_path = Path(__file__).parent.parent
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Transform for evaluation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"ModelEvaluator initialized. Device: {self.device}")
    
    def load_model(self, model_path: str, class_mappings_path: str) -> Tuple[nn.Module, Dict]:
        """Load model and class mappings."""
        # Load class mappings
        with open(class_mappings_path, 'r') as f:
            mappings = json.load(f)
        
        # Handle different mapping formats
        if 'magnitude_classes' in mappings:
            mag_classes = mappings['magnitude_classes']
            azi_classes = mappings['azimuth_classes']
        else:
            mag_classes = [mappings['magnitude'][str(i)] for i in range(len(mappings['magnitude']))]
            azi_classes = [mappings['azimuth'][str(i)] for i in range(len(mappings['azimuth']))]
        
        # Create model
        model = MultiTaskConvNeXt(len(mag_classes), len(azi_classes))
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        class_info = {
            'magnitude_classes': mag_classes,
            'azimuth_classes': azi_classes
        }
        
        return model, class_info
    
    def load_benchmark_data(self) -> pd.DataFrame:
        """Load benchmark test set."""
        benchmark_path = self.base_path / self.eval_config.get('benchmark_test_set', 
                                                               'data/benchmark/benchmark_test.csv')
        
        if not benchmark_path.exists():
            logger.warning(f"Benchmark file not found: {benchmark_path}")
            # Fall back to production test set
            fallback_path = self.base_path.parent / 'dataset_unified/metadata/test_split.csv'
            if fallback_path.exists():
                logger.info(f"Using fallback test set: {fallback_path}")
                return pd.read_csv(fallback_path)
            raise FileNotFoundError("No benchmark test set found")
        
        return pd.read_csv(benchmark_path)
    
    def evaluate_model(self, model_path: str, class_mappings_path: str, 
                      benchmark_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Evaluate a model on the benchmark test set.
        
        Args:
            model_path: Path to model checkpoint
            class_mappings_path: Path to class mappings JSON
            benchmark_df: Optional benchmark DataFrame (loads default if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating model: {model_path}")
        
        # Load model
        model, class_info = self.load_model(model_path, class_mappings_path)
        
        # Load benchmark data
        if benchmark_df is None:
            benchmark_df = self.load_benchmark_data()
        
        # Create class mappings
        mag_to_idx = {c: i for i, c in enumerate(class_info['magnitude_classes'])}
        azi_to_idx = {c: i for i, c in enumerate(class_info['azimuth_classes'])}
        
        # Evaluate
        mag_preds = []
        mag_labels = []
        mag_probs = []
        azi_preds = []
        azi_labels = []
        azi_probs = []
        
        dataset_dir = self.base_path.parent / 'dataset_unified/spectrograms'
        
        # Try multiple path patterns for finding spectrograms
        def find_spectrogram(filename, mag_class):
            possible_paths = [
                dataset_dir / filename,
                dataset_dir / 'by_magnitude' / mag_class / filename,
                dataset_dir / mag_class / filename,
                dataset_dir / 'augmented' / filename,
                dataset_dir / 'original' / filename,
            ]
            for p in possible_paths:
                if p.exists():
                    return p
            return None
        
        with torch.no_grad():
            for _, row in benchmark_df.iterrows():
                # Load image
                filename = row.get('filename') or row.get('spectrogram_file')
                if pd.isna(filename):
                    continue
                    
                mag_class = row.get('magnitude_class', 'Medium')
                img_path = find_spectrogram(filename, mag_class)
                
                if img_path is None:
                    continue
                
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image).unsqueeze(0).to(self.device)
                
                # Predict
                mag_out, azi_out = model(image)
                mag_prob = torch.softmax(mag_out, dim=1)
                azi_prob = torch.softmax(azi_out, dim=1)
                
                mag_preds.append(mag_out.argmax(1).item())
                mag_labels.append(mag_to_idx.get(row['magnitude_class'], 0))
                mag_probs.append(mag_prob.cpu().numpy()[0])
                
                azi_preds.append(azi_out.argmax(1).item())
                azi_labels.append(azi_to_idx.get(row['azimuth_class'], 0))
                azi_probs.append(azi_prob.cpu().numpy()[0])
        
        # Calculate metrics
        results = {
            "magnitude": self._calculate_metrics(
                mag_labels, mag_preds, mag_probs, 
                class_info['magnitude_classes'], "magnitude"
            ),
            "azimuth": self._calculate_metrics(
                azi_labels, azi_preds, azi_probs,
                class_info['azimuth_classes'], "azimuth"
            ),
            "total_samples": len(mag_preds)
        }
        
        # Summary metrics
        results["magnitude_accuracy"] = results["magnitude"]["accuracy"]
        results["azimuth_accuracy"] = results["azimuth"]["accuracy"]
        results["magnitude_f1"] = results["magnitude"]["f1_weighted"]
        results["azimuth_f1"] = results["azimuth"]["f1_weighted"]
        results["magnitude_mcc"] = results["magnitude"]["mcc"]
        results["azimuth_mcc"] = results["azimuth"]["mcc"]
        
        logger.info(f"Evaluation complete: Mag={results['magnitude_accuracy']:.2f}%, "
                   f"Azi={results['azimuth_accuracy']:.2f}%")
        
        return results
    
    def _calculate_metrics(self, labels: List[int], preds: List[int], 
                          probs: List[np.ndarray], class_names: List[str],
                          task_name: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a task."""
        labels = np.array(labels)
        preds = np.array(preds)
        probs = np.array(probs)
        
        metrics = {
            "accuracy": accuracy_score(labels, preds) * 100,
            "f1_macro": f1_score(labels, preds, average='macro') * 100,
            "f1_weighted": f1_score(labels, preds, average='weighted') * 100,
            "precision_macro": precision_score(labels, preds, average='macro', zero_division=0) * 100,
            "recall_macro": recall_score(labels, preds, average='macro', zero_division=0) * 100,
            "mcc": matthews_corrcoef(labels, preds) * 100,
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
            "class_names": class_names
        }
        
        # Per-class metrics
        report = classification_report(labels, preds, target_names=class_names, 
                                       output_dict=True, zero_division=0)
        metrics["per_class"] = report
        
        # AUC-ROC (if multi-class)
        try:
            if len(class_names) > 2:
                metrics["auc_roc"] = roc_auc_score(labels, probs, multi_class='ovr') * 100
            else:
                metrics["auc_roc"] = roc_auc_score(labels, probs[:, 1]) * 100
        except Exception as e:
            logger.warning(f"Could not calculate AUC-ROC: {e}")
            metrics["auc_roc"] = None
        
        return metrics
    
    def compare_models(self, champion_results: Dict, challenger_results: Dict) -> Dict[str, Any]:
        """
        Compare champion and challenger model results.
        
        Args:
            champion_results: Evaluation results for champion
            challenger_results: Evaluation results for challenger
            
        Returns:
            Comparison results
        """
        comparison = {
            "metrics_comparison": {},
            "improvements": {},
            "summary": {}
        }
        
        metrics_to_compare = [
            ("magnitude_accuracy", "higher_better"),
            ("azimuth_accuracy", "higher_better"),
            ("magnitude_f1", "higher_better"),
            ("azimuth_f1", "higher_better"),
            ("magnitude_mcc", "higher_better"),
            ("azimuth_mcc", "higher_better")
        ]
        
        for metric, direction in metrics_to_compare:
            champ_val = champion_results.get(metric, 0)
            chall_val = challenger_results.get(metric, 0)
            diff = chall_val - champ_val
            
            if direction == "higher_better":
                improved = diff > 0
            else:
                improved = diff < 0
            
            comparison["metrics_comparison"][metric] = {
                "champion": champ_val,
                "challenger": chall_val,
                "difference": diff,
                "improved": improved
            }
            
            comparison["improvements"][metric] = improved
        
        # Summary
        total_metrics = len(metrics_to_compare)
        improved_count = sum(comparison["improvements"].values())
        
        comparison["summary"] = {
            "total_metrics": total_metrics,
            "improved_count": improved_count,
            "degraded_count": total_metrics - improved_count,
            "improvement_rate": improved_count / total_metrics * 100
        }
        
        return comparison
    
    def generate_evaluation_report(self, results: Dict[str, Any], model_name: str = "Model") -> str:
        """Generate human-readable evaluation report."""
        lines = [
            "=" * 60,
            f"EVALUATION REPORT: {model_name}",
            "=" * 60,
            "",
            "SUMMARY METRICS:",
            "-" * 40,
            f"  Magnitude Accuracy: {results['magnitude_accuracy']:.2f}%",
            f"  Azimuth Accuracy:   {results['azimuth_accuracy']:.2f}%",
            f"  Magnitude F1:       {results['magnitude_f1']:.2f}%",
            f"  Azimuth F1:         {results['azimuth_f1']:.2f}%",
            f"  Magnitude MCC:      {results['magnitude_mcc']:.2f}%",
            f"  Azimuth MCC:        {results['azimuth_mcc']:.2f}%",
            f"  Total Samples:      {results['total_samples']}",
            ""
        ]
        
        # Magnitude details
        mag = results['magnitude']
        lines.extend([
            "MAGNITUDE CLASSIFICATION:",
            "-" * 40,
            f"  Classes: {', '.join(mag['class_names'])}",
            f"  Precision (macro): {mag['precision_macro']:.2f}%",
            f"  Recall (macro):    {mag['recall_macro']:.2f}%",
            ""
        ])
        
        # Azimuth details
        azi = results['azimuth']
        lines.extend([
            "AZIMUTH CLASSIFICATION:",
            "-" * 40,
            f"  Classes: {', '.join(azi['class_names'])}",
            f"  Precision (macro): {azi['precision_macro']:.2f}%",
            f"  Recall (macro):    {azi['recall_macro']:.2f}%",
            ""
        ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def evaluate_challenger(self) -> Dict[str, Any]:
        """
        Evaluate the challenger model.
        
        Returns:
            Dictionary with evaluation results
        """
        challenger_path = self.base_path / self.config['paths']['challenger_model']
        model_path = challenger_path / 'best_model.pth'
        mappings_path = challenger_path / 'class_mappings.json'
        
        if not model_path.exists():
            return {
                "success": False,
                "error": f"Challenger model not found: {model_path}"
            }
        
        if not mappings_path.exists():
            return {
                "success": False,
                "error": f"Class mappings not found: {mappings_path}"
            }
        
        try:
            results = self.evaluate_model(str(model_path), str(mappings_path))
            return {
                "success": True,
                "results": results,
                "model_path": str(model_path)
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def evaluate_champion(self) -> Dict[str, Any]:
        """
        Evaluate the champion model.
        
        Returns:
            Dictionary with evaluation results
        """
        champion_path = self.base_path / self.config['paths']['champion_model']
        model_path = champion_path / 'best_model.pth'
        mappings_path = champion_path / 'class_mappings.json'
        
        # If no champion exists, try production model
        if not model_path.exists():
            prod_model = self.base_path / self.config['paths']['production_model']
            prod_config = self.base_path / self.config['paths']['production_config']
            
            if prod_model.exists() and prod_config.exists():
                model_path = prod_model
                mappings_path = prod_config
            else:
                return {
                    "success": False,
                    "error": "No champion or production model found"
                }
        
        try:
            results = self.evaluate_model(str(model_path), str(mappings_path))
            return {
                "success": True,
                "results": results,
                "model_path": str(model_path)
            }
        except Exception as e:
            logger.error(f"Champion evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
