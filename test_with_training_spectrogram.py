#!/usr/bin/env python3
"""
Test Model with Training Spectrogram
Verify that model CAN predict Normal when given correct preprocessing
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import sys
import os

print("="*70)
print("TEST MODEL WITH TRAINING SPECTROGRAM")
print("="*70)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {DEVICE}")

# ============================================================================
# LOAD MODEL V6 (Binary Classifier)
# ============================================================================

print(f"\n🔮 Loading Binary Classifier (Model V6)...")

# Model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(SimpleCNN, self).__init__()
        
        from torchvision.models import resnet18
        backbone = resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        backbone_features = 512
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# Find latest experiment
exp_dir = Path('experiments_v6/exp_v6_separate_20260202_150329')
if not exp_dir.exists():
    print(f"❌ Experiment directory not found: {exp_dir}")
    sys.exit(1)

# Load binary model
binary_model = SimpleCNN(num_classes=2)
binary_checkpoint = torch.load(exp_dir / 'binary_classifier_best.pth', map_location=DEVICE)
binary_model.load_state_dict(binary_checkpoint['model_state_dict'])
binary_model = binary_model.to(DEVICE)
binary_model.eval()

print(f"✅ Binary classifier loaded")
print(f"   Val Acc: {binary_checkpoint['val_acc']:.2f}%")

# ============================================================================
# LOAD TRAINING SPECTROGRAMS (Normal samples)
# ============================================================================

print(f"\n📊 Loading training spectrograms (Normal samples)...")

# Load metadata
metadata_file = 'dataset_unified/metadata/unified_metadata.csv'
df = pd.read_csv(metadata_file)

# Get Normal samples
normal_samples = df[df['magnitude_class'] == 'Normal'].head(10)

print(f"   Found {len(normal_samples)} Normal samples to test")

# ============================================================================
# TEST WITH TRAINING SPECTROGRAMS
# ============================================================================

print(f"\n{'='*70}")
print("TESTING WITH TRAINING SPECTROGRAMS")
print(f"{'='*70}")

results = []

for idx, sample in normal_samples.iterrows():
    station = sample['station']
    date = sample['date']
    
    print(f"\n{'='*70}")
    print(f"TEST {idx+1}/{len(normal_samples)}: {station} - {date}")
    print(f"Expected: Normal (class 1)")
    print(f"{'='*70}")
    
    # Load spectrogram
    spec_path = Path('dataset_unified') / sample['unified_path']
    
    if not spec_path.exists():
        print(f"❌ Spectrogram not found: {spec_path}")
        continue
    
    # Load image
    img = Image.open(spec_path).convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS)
    
    # Transform
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        output = binary_model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item() * 100
    
    # Check result
    is_correct = (pred == 1)  # 1 = Normal
    
    results.append({
        'station': station,
        'date': date,
        'expected': 'Normal',
        'predicted': 'Normal' if pred == 1 else 'Earthquake',
        'confidence': conf,
        'correct': is_correct
    })
    
    icon = "✅" if is_correct else "❌"
    print(f"\n📊 RESULT:")
    print(f"   Predicted: {'Normal' if pred == 1 else 'Earthquake'} ({conf:.1f}%) {icon}")
    print(f"   Expected: Normal")
    print(f"   Status: {'CORRECT' if is_correct else 'WRONG'}")

# ============================================================================
# ANALYSIS
# ============================================================================

print(f"\n{'='*70}")
print("RESULTS ANALYSIS")
print(f"{'='*70}")

correct_count = sum(1 for r in results if r['correct'])
total_count = len(results)

print(f"\n📊 OVERALL:")
print(f"   Total tests: {total_count}")
print(f"   Correct: {correct_count}")
print(f"   Accuracy: {correct_count/total_count*100:.1f}%")

# Show predictions
print(f"\n📊 PREDICTIONS:")
pred_counts = {}
for r in results:
    pred = r['predicted']
    pred_counts[pred] = pred_counts.get(pred, 0) + 1

for pred, count in pred_counts.items():
    print(f"   {pred}: {count}/{total_count} ({count/total_count*100:.1f}%)")

# Show confidence
avg_conf = np.mean([r['confidence'] for r in results])
print(f"\n📊 CONFIDENCE:")
print(f"   Average: {avg_conf:.1f}%")

# ============================================================================
# FINAL VERDICT
# ============================================================================

print(f"\n{'='*70}")
print("FINAL VERDICT")
print(f"{'='*70}")

print(f"\n🎯 CAN MODEL PREDICT NORMAL WITH TRAINING SPECTROGRAMS?")

if correct_count / total_count >= 0.9:
    print(f"   ✅ YES - Model can predict Normal correctly ({correct_count/total_count*100:.1f}%)")
    print(f"   ✅ Model is WORKING CORRECTLY")
    print(f"   🔴 Problem is PREPROCESSING MISMATCH")
    print(f"\n💡 CONCLUSION:")
    print(f"   Model learned correctly from training data")
    print(f"   Scanner preprocessing is DIFFERENT from training")
    print(f"   MUST fix scanner to match training preprocessing")
elif correct_count / total_count >= 0.7:
    print(f"   🟡 PARTIALLY - Model sometimes predicts Normal ({correct_count/total_count*100:.1f}%)")
    print(f"   ⚠️  Model has some issues but preprocessing is also a factor")
elif correct_count / total_count >= 0.5:
    print(f"   🟠 POOR - Model rarely predicts Normal ({correct_count/total_count*100:.1f}%)")
    print(f"   ⚠️  Model has significant issues")
else:
    print(f"   ❌ NO - Model CANNOT predict Normal ({correct_count/total_count*100:.1f}%)")
    print(f"   🔴 Model has FUNDAMENTAL PROBLEMS")
    print(f"   ⚠️  Even with correct preprocessing, model fails")

print(f"\n{'='*70}")
print("✅ TEST COMPLETE!")
print(f"{'='*70}")

# Save results
import json
with open('training_spectrogram_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: training_spectrogram_test_results.json")
