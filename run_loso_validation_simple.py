#!/usr/bin/env python3
"""
LOSO (Leave-One-Station-Out) Validation - FIXED VERSION
Uses correct dataset: dataset_experiment_3/final_metadata/test_exp3.csv

Author: Earthquake Prediction Research Team
Date: 13 February 2026
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Import model
from earthquake_cnn_v3 import EarthquakeCNNV3


class SimpleEarthquakeDataset(Dataset):
    """Simple dataset for LOSO validation"""
    
    def __init__(self, df, dataset_root='dataset_experiment_3', image_size=224):
        self.df = df.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class mappings
        self.mag_classes = sorted(df['magnitude_class'].unique())
        self.mag_to_idx = {c: i for i, c in enumerate(self.mag_classes)}
        
        print(f"Dataset: {len(df)} samples")
        print(f"Magnitude classes: {self.mag_classes}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.dataset_root / row['consolidation_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Labels (azimuth hardcoded to 0 since we focus on magnitude)
        mag_label = self.mag_to_idx[row['magnitude_class']]
        az_label = 0  # Dummy
        
        return image, mag_label, az_label


class LOSOValidator:
    """Leave-One-Station-Out Cross-Validation"""
    
    def __init__(self, model_path: str, test_csv: str = 'dataset_experiment_3/final_metadata/test_exp3.csv'):
        """
        Initialize LOSO validator
        
        Args:
            model_path: Path to trained model
            test_csv: Path to test metadata CSV
        """
        self.model_path = model_path
        self.test_csv = test_csv
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Output directory
        self.output_dir = Path('loso_validation_results')
        self.output_dir.mkdir(exist_ok=True)
        
        print("="*80)
        print("LOSO VALIDATION - Leave-One-Station-Out Cross-Validation")
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Test Data: {test_csv}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print("="*80)
        
    def load_model(self) -> nn.Module:
        """Load trained model"""
        print("\n[1/4] Loading model...")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Default to 4 magnitude classes (Large, Medium, Moderate, Normal)
        # and 9 azimuth (not used but required by model)
        model = EarthquakeCNNV3(
            num_magnitude_classes=4,
            num_azimuth_classes=9
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"  Model loaded successfully")
        print(f"  Magnitude classes: 4")
        print(f"  Azimuth classes: 9")
        
        return model
    
    def load_test_data(self):
        """Load test dataset"""
        print("\n[2/4] Loading test data...")
        
        df = pd.read_csv(self.test_csv)
        print(f"  Total test samples: {len(df)}")
        
        # Group by station
        station_groups = defaultdict(list)
        for idx, row in df.iterrows():
            station = row['station']
            station_groups[station].append(idx)
        
        print(f"  Stations found: {len(station_groups)}")
        for station, indices in sorted(station_groups.items()):
            print(f"    {station}: {len(indices)} samples")
        
        return df, dict(station_groups)
    
    def validate_station(self, model: nn.Module, test_loader: DataLoader,
                        station_name: str) -> Dict:
        """Validate on one station"""
        model.eval()
        
        # Metrics
        mag_correct = 0
        total = 0
        
        mag_preds_all = []
        mag_labels_all = []
        
        with torch.no_grad():
            for images, mag_labels, _ in tqdm(test_loader, 
                                             desc=f"  Testing {station_name}",
                                             leave=False):
                images = images.to(self.device)
                mag_labels = mag_labels.to(self.device)
                
                # Forward pass
                mag_logits, _ = model(images)
                
                # Predictions
                mag_preds = mag_logits.argmax(1)
                
                # Accumulate
                mag_correct += (mag_preds == mag_labels).sum().item()
                total += images.size(0)
                
                # Store for confusion matrix
                mag_preds_all.extend(mag_preds.cpu().numpy())
                mag_labels_all.extend(mag_labels.cpu().numpy())
        
        # Calculate metrics
        mag_acc = mag_correct / total if total > 0 else 0
        
        return {
            'station': station_name,
            'samples': total,
            'magnitude_acc': mag_acc,
            'magnitude_correct': mag_correct,
            'mag_preds': mag_preds_all,
            'mag_labels': mag_labels_all
        }
    
    def run(self):
        """Run LOSO validation"""
        start_time = time.time()
        
        # Load model
        model = self.load_model()
        
        # Load test data
        df_test, station_groups = self.load_test_data()
        
        # Create full dataset
        full_dataset = SimpleEarthquakeDataset(df_test)
        
        # Run validation for each station
        print("\n[3/4] Running LOSO validation...")
        
        all_results = []
        
        for station_name, indices in sorted(station_groups.items()):
            print(f"\n--- Station: {station_name} ({len(indices)} samples) ---")
            
            # Create subset
            test_subset = Subset(full_dataset, indices)
            
            # Create loader
            test_loader = DataLoader(
                test_subset,
                batch_size=32,
                shuffle=False,
                num_workers=0
            )
            
            # Validate
            result = self.validate_station(model, test_loader, station_name)
            all_results.append(result)
            
            # Print result
            print(f"  Magnitude Acc: {result['magnitude_acc']:.4f} "
                  f"({result['magnitude_correct']}/{result['samples']})")
        
        # Aggregate results
        print("\n[4/4] Aggregating results...")
        
        total_samples = sum(r['samples'] for r in all_results)
        total_mag_correct = sum(r['magnitude_correct'] for r in all_results)
        
        overall_mag_acc = total_mag_correct / total_samples
        
        # Per-station accuracy stats
        mag_accs = [r['magnitude_acc'] for r in all_results]
        
        summary = {
            'overall': {
                'magnitude_accuracy': overall_mag_acc,
                'total_samples': total_samples,
                'num_stations': len(all_results)
            },
            'per_station_stats': {
                'magnitude': {
                    'mean': np.mean(mag_accs),
                    'std': np.std(mag_accs),
                    'min': np.min(mag_accs),
                    'max': np.max(mag_accs)
                }
            },
            'per_station_results': [
                {
                    'station': r['station'],
                    'samples': r['samples'],
                    'magnitude_acc': r['magnitude_acc']
                }
                for r in all_results
            ]
        }
        
        # Save results
        results_file = self.output_dir / 'loso_results.json'
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("LOSO VALIDATION COMPLETED")
        print("="*80)
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        print(f"\nOverall Results:")
        print(f"  Magnitude Accuracy: {overall_mag_acc:.4f} ({total_mag_correct}/{total_samples})")
        print(f"\nPer-Station Statistics:")
        print(f"  Magnitude: {np.mean(mag_accs):.4f} +/- {np.std(mag_accs):.4f}")
        print(f"    Range: [{np.min(mag_accs):.4f}, {np.max(mag_accs):.4f}]")
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - {results_file}")
        print("="*80)
        
        return summary


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LOSO Validation for ConvNeXt')
    parser.add_argument('--model-path', type=str,
                       default='experiments_convnext/convnext_tiny_20260205_100924/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--test-csv', type=str,
                       default='dataset_experiment_3/final_metadata/test_exp3.csv',
                       help='Path to test CSV')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"ERROR: Model not found at {args.model_path}")
        print("\nAvailable models:")
        for model_file in Path('experiments_convnext').rglob('*.pth'):
            print(f"  - {model_file}")
        return 1
    
    # Check if test CSV exists
    if not Path(args.test_csv).exists():
        print(f"ERROR: Test CSV not found at {args.test_csv}")
        return 1
    
    # Run validation
    validator = LOSOValidator(args.model_path, args.test_csv)
    validator.run()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
