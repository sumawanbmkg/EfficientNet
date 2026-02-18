#!/usr/bin/env python3
"""
Solar Storm Negative Test
Test if model incorrectly classifies geomagnetic storms as earthquake precursors

This is a critical validation to ensure the model doesn't have high false positive rate
by confusing solar storm activity with earthquake precursor signals.

Author: Earthquake Prediction Research Team
Date: 4 February 2026
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Known major solar storm dates (Kp >= 5)
# Source: NOAA Space Weather Prediction Center
SOLAR_STORM_DATES = [
    # 2023 storms
    ('2023-02-27', 'G2 Storm', 6),
    ('2023-03-24', 'G3 Storm', 7),
    ('2023-04-24', 'G4 Storm', 8),
    ('2023-11-05', 'G3 Storm', 7),
    ('2023-12-01', 'G2 Storm', 6),
    # 2022 storms
    ('2022-03-13', 'G2 Storm', 6),
    ('2022-04-10', 'G3 Storm', 7),
    ('2022-08-07', 'G2 Storm', 6),
    # 2021 storms
    ('2021-10-12', 'G2 Storm', 6),
    ('2021-11-04', 'G3 Storm', 7),
    # 2019 storms
    ('2019-05-14', 'G2 Storm', 6),
    ('2019-08-31', 'G2 Storm', 6),
    # 2018 storms
    ('2018-03-18', 'G1 Storm', 5),
    ('2018-08-26', 'G2 Storm', 6),
    ('2018-09-11', 'G2 Storm', 6),
]


class EfficientNetMultiTask(nn.Module):
    """EfficientNet-B0 based multi-task model - matches training architecture"""
    
    def __init__(self, num_magnitude_classes=4, num_azimuth_classes=9, dropout_rate=0.5):
        super(EfficientNetMultiTask, self).__init__()
        
        # Load pretrained EfficientNet-B0
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Extract features (remove classifier)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        
        # Shared layers - matches training architecture
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(1280, 512),  # EfficientNet-B0 outputs 1280 features
            nn.ReLU(True),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific heads
        self.magnitude_head = nn.Linear(512, num_magnitude_classes)
        self.azimuth_head = nn.Linear(512, num_azimuth_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.shared(x)
        return self.magnitude_head(x), self.azimuth_head(x)


def load_model(model_path):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetMultiTask(num_magnitude_classes=4, num_azimuth_classes=9)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model, device


def find_existing_spectrograms_for_storm_dates():
    """Find if we have any spectrograms from solar storm dates in our dataset"""
    metadata_path = Path('dataset_unified/metadata/unified_metadata.csv')
    if not metadata_path.exists():
        return []
    
    df = pd.read_csv(metadata_path)
    storm_dates = [d[0] for d in SOLAR_STORM_DATES]
    
    # Check if any of our data overlaps with storm dates
    matches = []
    for _, row in df.iterrows():
        date_str = str(row['date'])
        if date_str in storm_dates:
            matches.append({
                'date': date_str,
                'station': row['station'],
                'path': row['unified_path'],
                'magnitude_class': row['magnitude_class'],
                'is_precursor': row['magnitude_class'] != 'Normal'
            })
    
    return matches


def generate_synthetic_storm_spectrogram(output_path, storm_intensity='moderate'):
    """
    Generate synthetic spectrogram that mimics solar storm characteristics
    Solar storms have:
    - High amplitude across all frequencies
    - Sudden onset (SSC - Storm Sudden Commencement)
    - Long duration (hours to days)
    - Different frequency signature than earthquake precursors
    """
    np.random.seed(42)
    
    # Create spectrogram-like image
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    
    # Time and frequency axes
    time = np.linspace(0, 24, 500)  # 24 hours
    freq = np.linspace(0, 0.1, 200)  # 0-0.1 Hz (ULF range)
    
    T, F = np.meshgrid(time, freq)
    
    # Solar storm characteristics
    if storm_intensity == 'severe':
        # G4-G5 storm: Very high amplitude, broad frequency
        amplitude = 3.0
        noise_level = 0.5
    elif storm_intensity == 'strong':
        # G3 storm
        amplitude = 2.0
        noise_level = 0.4
    else:
        # G1-G2 storm
        amplitude = 1.5
        noise_level = 0.3
    
    # Storm sudden commencement (SSC) - sharp onset
    ssc_time = 6  # Storm starts at hour 6
    storm_envelope = np.where(T > ssc_time, 1 - np.exp(-(T - ssc_time) / 2), 0)
    
    # Main phase - gradual increase then decrease
    main_phase = storm_envelope * np.exp(-((T - 12) ** 2) / 50)
    
    # Frequency content - storms affect all frequencies but especially Pc5 (0.002-0.007 Hz)
    freq_response = np.exp(-((F - 0.004) ** 2) / 0.0001) + 0.3 * np.exp(-((F - 0.02) ** 2) / 0.001)
    
    # Combine
    spectrogram = amplitude * main_phase * freq_response + noise_level * np.random.randn(*T.shape)
    spectrogram = np.clip(spectrogram, 0, 5)
    
    # Plot
    ax.pcolormesh(T, F, spectrogram, shading='auto', cmap='jet')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 0.1)
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Resize to 224x224
    img = Image.open(output_path)
    img = img.resize((224, 224), Image.LANCZOS)
    img.save(output_path)
    
    return output_path


def predict_spectrogram(model, device, image_path, transform):
    """Predict magnitude and azimuth for a spectrogram"""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mag_out, azi_out = model(image)
        mag_probs = torch.softmax(mag_out, dim=1)
        azi_probs = torch.softmax(azi_out, dim=1)
        
        mag_pred = torch.argmax(mag_probs, dim=1).item()
        azi_pred = torch.argmax(azi_probs, dim=1).item()
        mag_conf = mag_probs[0, mag_pred].item()
        azi_conf = azi_probs[0, azi_pred].item()
    
    magnitude_classes = ['Large', 'Major', 'Medium', 'Small']
    azimuth_classes = ['E', 'N', 'NE', 'NW', 'Normal', 'S', 'SE', 'SW', 'W']
    
    return {
        'magnitude': magnitude_classes[mag_pred],
        'azimuth': azimuth_classes[azi_pred],
        'mag_confidence': mag_conf,
        'azi_confidence': azi_conf,
        'is_precursor': azimuth_classes[azi_pred] != 'Normal',
        'mag_probs': mag_probs[0].cpu().numpy().tolist(),
        'azi_probs': azi_probs[0].cpu().numpy().tolist()
    }


def run_solar_storm_test():
    """Run the complete solar storm negative test"""
    print("\n" + "="*70)
    print("SOLAR STORM NEGATIVE TEST")
    print("Testing if model incorrectly classifies geomagnetic storms as precursors")
    print("="*70)
    
    # Create output directory
    output_dir = Path('solar_storm_test_results')
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    model_path = 'experiments_efficientnet_smote/exp_20260203_155623/best_efficientnet_smote_model.pth'
    if not Path(model_path).exists():
        # Try alternative paths
        alt_paths = [
            'tuning_results/best_efficientnet_smote_model.pth',
            'experiments_efficientnet_smote/exp_20260203_155239/best_efficientnet_smote_model.pth'
        ]
        for alt in alt_paths:
            if Path(alt).exists():
                model_path = alt
                break
    
    print(f"\nLoading model from: {model_path}")
    model, device = load_model(model_path)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    
    # Test 1: Check existing data for storm date overlaps
    print("\n--- Test 1: Checking existing dataset for solar storm date overlaps ---")
    overlaps = find_existing_spectrograms_for_storm_dates()
    if overlaps:
        print(f"Found {len(overlaps)} spectrograms from solar storm dates!")
        for overlap in overlaps[:5]:
            print(f"  - {overlap['date']} at {overlap['station']}: labeled as {overlap['magnitude_class']}")
    else:
        print("No overlaps found between dataset and known solar storm dates.")
    
    # Test 2: Generate synthetic solar storm spectrograms and test
    print("\n--- Test 2: Testing synthetic solar storm spectrograms ---")
    
    storm_types = [
        ('mild', 'G1-G2 Storm (Kp=5-6)'),
        ('moderate', 'G2-G3 Storm (Kp=6-7)'),
        ('strong', 'G3-G4 Storm (Kp=7-8)'),
        ('severe', 'G4-G5 Storm (Kp=8-9)')
    ]
    
    for intensity, description in storm_types:
        print(f"\nTesting {description}...")
        
        # Generate multiple samples
        for i in range(5):
            img_path = output_dir / f'synthetic_storm_{intensity}_{i}.png'
            generate_synthetic_storm_spectrogram(img_path, intensity)
            
            prediction = predict_spectrogram(model, device, img_path, transform)
            
            results.append({
                'test_type': 'synthetic',
                'storm_intensity': intensity,
                'description': description,
                'sample': i,
                'predicted_magnitude': prediction['magnitude'],
                'predicted_azimuth': prediction['azimuth'],
                'mag_confidence': prediction['mag_confidence'],
                'azi_confidence': prediction['azi_confidence'],
                'is_false_positive': prediction['is_precursor'],
                'expected': 'Normal (no earthquake)'
            })
            
            status = "❌ FALSE POSITIVE" if prediction['is_precursor'] else "✅ Correct (Normal)"
            print(f"  Sample {i+1}: {prediction['azimuth']} ({prediction['azi_confidence']:.1%}) - {status}")
    
    # Test 3: Use actual quiet day spectrograms as baseline
    print("\n--- Test 3: Testing actual quiet day spectrograms (baseline) ---")
    
    # Find Normal class spectrograms
    metadata_path = Path('dataset_unified/metadata/unified_metadata.csv')
    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
        normal_samples = df[df['magnitude_class'] == 'Normal'].head(10)
        
        for _, row in normal_samples.iterrows():
            img_path = Path('dataset_unified') / row['unified_path']
            if img_path.exists():
                prediction = predict_spectrogram(model, device, img_path, transform)
                
                results.append({
                    'test_type': 'quiet_day',
                    'storm_intensity': 'none',
                    'description': 'Quiet Day (Kp<3)',
                    'sample': row['unified_path'],
                    'predicted_magnitude': prediction['magnitude'],
                    'predicted_azimuth': prediction['azimuth'],
                    'mag_confidence': prediction['mag_confidence'],
                    'azi_confidence': prediction['azi_confidence'],
                    'is_false_positive': prediction['is_precursor'],
                    'expected': 'Normal'
                })
    
    # Calculate statistics
    print("\n" + "="*70)
    print("SOLAR STORM TEST RESULTS SUMMARY")
    print("="*70)
    
    synthetic_results = [r for r in results if r['test_type'] == 'synthetic']
    quiet_results = [r for r in results if r['test_type'] == 'quiet_day']
    
    # False positive rate for synthetic storms
    fp_synthetic = sum(1 for r in synthetic_results if r['is_false_positive'])
    fp_rate_synthetic = fp_synthetic / len(synthetic_results) * 100 if synthetic_results else 0
    
    # False positive rate for quiet days
    fp_quiet = sum(1 for r in quiet_results if r['is_false_positive'])
    fp_rate_quiet = fp_quiet / len(quiet_results) * 100 if quiet_results else 0
    
    print(f"\nSynthetic Solar Storm Spectrograms:")
    print(f"  Total tested: {len(synthetic_results)}")
    print(f"  False positives: {fp_synthetic}")
    print(f"  False positive rate: {fp_rate_synthetic:.1f}%")
    
    print(f"\nQuiet Day Spectrograms (baseline):")
    print(f"  Total tested: {len(quiet_results)}")
    print(f"  False positives: {fp_quiet}")
    print(f"  False positive rate: {fp_rate_quiet:.1f}%")
    
    # By storm intensity
    print("\nFalse Positive Rate by Storm Intensity:")
    for intensity, _ in storm_types:
        intensity_results = [r for r in synthetic_results if r['storm_intensity'] == intensity]
        fp = sum(1 for r in intensity_results if r['is_false_positive'])
        rate = fp / len(intensity_results) * 100 if intensity_results else 0
        print(f"  {intensity.capitalize()}: {fp}/{len(intensity_results)} ({rate:.1f}%)")
    
    # Assessment
    print("\n" + "="*70)
    print("ASSESSMENT")
    print("="*70)
    
    if fp_rate_synthetic < 20:
        assessment = "✅ EXCELLENT: Model correctly identifies most solar storms as non-precursor"
    elif fp_rate_synthetic < 40:
        assessment = "⚠️ ACCEPTABLE: Some false positives, but within reasonable range"
    else:
        assessment = "❌ CONCERNING: High false positive rate - model may confuse storms with precursors"
    
    print(f"\n{assessment}")
    print(f"\nOverall False Positive Rate: {fp_rate_synthetic:.1f}%")
    
    # Save results
    final_results = {
        'test_date': datetime.now().isoformat(),
        'model_path': str(model_path),
        'synthetic_storm_results': {
            'total_tested': len(synthetic_results),
            'false_positives': fp_synthetic,
            'false_positive_rate': fp_rate_synthetic
        },
        'quiet_day_results': {
            'total_tested': len(quiet_results),
            'false_positives': fp_quiet,
            'false_positive_rate': fp_rate_quiet
        },
        'by_intensity': {},
        'detailed_results': results,
        'assessment': assessment
    }
    
    for intensity, _ in storm_types:
        intensity_results = [r for r in synthetic_results if r['storm_intensity'] == intensity]
        fp = sum(1 for r in intensity_results if r['is_false_positive'])
        final_results['by_intensity'][intensity] = {
            'total': len(intensity_results),
            'false_positives': fp,
            'rate': fp / len(intensity_results) * 100 if intensity_results else 0
        }
    
    with open(output_dir / 'solar_storm_test_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}/solar_storm_test_results.json")
    
    return final_results


if __name__ == '__main__':
    results = run_solar_storm_test()
