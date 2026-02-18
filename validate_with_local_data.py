#!/usr/bin/env python3
"""
Validate with Local Data
Validasi model menggunakan data yang sudah ada di dataset lokal
Tidak perlu fetch dari server (lebih cepat dan reliable)

Author: Earthquake Prediction Research Team
Date: 3 February 2026
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        self.magnitude_head = nn.Linear(512, num_magnitude_classes)
        self.azimuth_head = nn.Linear(512, num_azimuth_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared(x)
        mag_out = self.magnitude_head(x)
        azi_out = self.azimuth_head(x)
        return mag_out, azi_out


class LocalDataValidator:
    """Validator menggunakan data lokal"""
    
    def __init__(self):
        logger.info("="*70)
        logger.info("LOCAL DATA VALIDATION")
        logger.info("="*70)
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.class_mappings = self._load_model()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Results
        self.results = []
    
    def _load_model(self):
        """Load model"""
        model_path = Path('experiments_fixed/exp_fixed_20260202_163643/best_model.pth')
        mapping_file = Path('experiments_fixed/exp_fixed_20260202_163643/class_mappings.json')
        
        # Load class mappings
        with open(mapping_file, 'r') as f:
            class_mappings = json.load(f)
        
        # Create model
        model = MultiTaskVGG16(
            len(class_mappings['magnitude_classes']),
            len(class_mappings['azimuth_classes'])
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"✅ Model loaded")
        return model, class_mappings
    
    def predict(self, image_path):
        """Predict from image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mag_output, azi_output = self.model(image_tensor)
            
            mag_probs = torch.softmax(mag_output, dim=1)[0]
            azi_probs = torch.softmax(azi_output, dim=1)[0]
            
            mag_pred_idx = torch.argmax(mag_probs).item()
            azi_pred_idx = torch.argmax(azi_probs).item()
            
            mag_pred = self.class_mappings['magnitude_classes'][mag_pred_idx]
            azi_pred = self.class_mappings['azimuth_classes'][azi_pred_idx]
            
            mag_conf = mag_probs[mag_pred_idx].item() * 100
            azi_conf = azi_probs[azi_pred_idx].item() * 100
        
        return {
            'magnitude': mag_pred,
            'azimuth': azi_pred,
            'mag_confidence': mag_conf,
            'azi_confidence': azi_conf,
            'is_precursor': mag_pred != 'Normal'
        }
    
    def validate_all(self, max_samples=None):
        """Validate all samples from test set"""
        # Load test set
        test_file = 'dataset_unified/metadata/test_split.csv'
        df = pd.read_csv(test_file)
        
        # Filter earthquake events
        earthquakes = df[df['magnitude_class'] != 'Normal'].copy()
        
        if max_samples:
            earthquakes = earthquakes.head(max_samples)
        
        logger.info(f"Validating {len(earthquakes)} earthquake samples...")
        
        # Validate each sample
        for idx, row in tqdm(earthquakes.iterrows(), total=len(earthquakes), desc="Validating"):
            spec_path = Path('dataset_unified') / row['unified_path']
            
            if not spec_path.exists():
                continue
            
            # Predict
            pred = self.predict(spec_path)
            
            # Compare with true labels
            true_mag = row['magnitude_class']
            true_azi = row['azimuth_class']
            
            detected = pred['is_precursor']
            mag_correct = (pred['magnitude'] == true_mag)
            azi_correct = (pred['azimuth'] == true_azi)
            
            result = {
                'station': row['station'],
                'date': row['date'],
                'true_magnitude': true_mag,
                'true_azimuth': true_azi,
                'pred_magnitude': pred['magnitude'],
                'pred_azimuth': pred['azimuth'],
                'mag_confidence': pred['mag_confidence'],
                'azi_confidence': pred['azi_confidence'],
                'detected': detected,
                'mag_correct': mag_correct,
                'azi_correct': azi_correct,
                'correct': detected and mag_correct
            }
            
            self.results.append(result)
        
        logger.info(f"✅ Validation complete: {len(self.results)} samples")
    
    def generate_report(self):
        """Generate validation report"""
        df = pd.DataFrame(self.results)
        
        if len(df) == 0:
            logger.error("❌ No results to report")
            return
        
        # Calculate metrics
        total = len(df)
        detected = df['detected'].sum()
        mag_correct = df['mag_correct'].sum()
        azi_correct = df['azi_correct'].sum()
        overall_correct = df['correct'].sum()
        
        detection_rate = (detected / total) * 100
        mag_accuracy = (mag_correct / total) * 100
        azi_accuracy = (azi_correct / total) * 100
        overall_accuracy = (overall_correct / total) * 100
        
        avg_mag_conf = df['mag_confidence'].mean()
        avg_azi_conf = df['azi_confidence'].mean()
        
        # Print report
        print("\n" + "="*70)
        print("VALIDATION REPORT (LOCAL DATA)")
        print("="*70)
        
        print(f"\n📊 SUMMARY")
        print(f"   Total samples: {total}")
        
        print(f"\n🎯 DETECTION RATE")
        print(f"   Detected: {detected}/{total} ({detection_rate:.1f}%)")
        print(f"   Not detected: {total-detected}/{total} ({(total-detected)/total*100:.1f}%)")
        
        if detection_rate >= 90:
            print(f"   ✅ EXCELLENT detection rate!")
        elif detection_rate >= 70:
            print(f"   ✅ GOOD detection rate")
        else:
            print(f"   ⚠️  MODERATE detection rate")
        
        print(f"\n📏 MAGNITUDE ACCURACY")
        print(f"   Correct: {mag_correct}/{total} ({mag_accuracy:.1f}%)")
        print(f"   Average confidence: {avg_mag_conf:.1f}%")
        
        print(f"\n🧭 AZIMUTH ACCURACY")
        print(f"   Correct: {azi_correct}/{total} ({azi_accuracy:.1f}%)")
        print(f"   Average confidence: {avg_azi_conf:.1f}%")
        
        print(f"\n⚡ OVERALL ACCURACY")
        print(f"   Correct: {overall_correct}/{total} ({overall_accuracy:.1f}%)")
        
        # Per-station analysis
        print(f"\n📍 PER-STATION ANALYSIS")
        station_stats = df.groupby('station').agg({
            'detected': ['count', 'sum'],
            'mag_correct': 'sum'
        })
        
        for station in station_stats.index:
            count = station_stats.loc[station, ('detected', 'count')]
            detected_count = station_stats.loc[station, ('detected', 'sum')]
            det_rate = (detected_count / count) * 100
            print(f"   {station}: {detected_count}/{count} detected ({det_rate:.1f}%)")
        
        # Save results
        output_dir = Path('validation_results')
        output_dir.mkdir(exist_ok=True)
        
        csv_file = output_dir / 'local_validation_results.csv'
        df.to_csv(csv_file, index=False)
        logger.info(f"\n💾 Results saved to: {csv_file}")
        
        print(f"\n{'='*70}")
        print("✅ VALIDATION COMPLETE!")
        print(f"{'='*70}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate with Local Data')
    parser.add_argument('--max-samples', '-n', type=int, default=None,
                       help='Maximum samples to validate')
    
    args = parser.parse_args()
    
    # Create validator
    validator = LocalDataValidator()
    
    # Validate
    validator.validate_all(max_samples=args.max_samples)
    
    # Generate report
    validator.generate_report()


if __name__ == '__main__':
    main()
