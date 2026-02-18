#!/usr/bin/env python3
"""
Evaluate Fixed Model on Test Set
Test model yang sudah di-train dengan fixed split (no leakage)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("EVALUATE FIXED MODEL ON TEST SET")
print("="*70)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")

# ============================================================================
# STEP 1: Load Model and Configuration
# ============================================================================

print(f"\n📊 Step 1: Loading model and configuration...")

# Find latest experiment
exp_dir = Path('experiments_fixed')
experiments = sorted(exp_dir.glob('exp_*'))
if not experiments:
    print("❌ No experiments found!")
    exit(1)

latest_exp = experiments[-1]
print(f"✅ Latest experiment: {latest_exp.name}")

# Load configuration
with open(latest_exp / 'config.json', 'r') as f:
    config = json.load(f)

# Load class mappings
with open(latest_exp / 'class_mappings.json', 'r') as f:
    class_mappings = json.load(f)

magnitude_classes = class_mappings['magnitude_classes']
azimuth_classes = class_mappings['azimuth_classes']

print(f"✅ Configuration loaded")
print(f"   Magnitude classes: {len(magnitude_classes)}")
print(f"   Azimuth classes: {len(azimuth_classes)}")

# ============================================================================
# STEP 2: Load Test Data
# ============================================================================

print(f"\n📊 Step 2: Loading test data...")

# Load test split
test_df = pd.read_csv('dataset_unified/metadata/test_split.csv')

# Encode labels
magnitude_to_idx = {cls: idx for idx, cls in enumerate(magnitude_classes)}
azimuth_to_idx = {cls: idx for idx, cls in enumerate(azimuth_classes)}

test_df['magnitude_label'] = test_df['magnitude_class'].map(magnitude_to_idx)
test_df['azimuth_label'] = test_df['azimuth_class'].map(azimuth_to_idx)

print(f"✅ Test data loaded: {len(test_df)} samples")

# ============================================================================
# STEP 3: Create Dataset
# ============================================================================

print(f"\n📊 Step 3: Creating test dataset...")

class EarthquakeDataset(Dataset):
    def __init__(self, dataframe, dataset_dir, transform=None):
        self.df = dataframe
        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        
        # Multiple source directories
        self.source_dirs = {
            'v2.1_original': Path('dataset_spectrogram_ssh_v22') / 'spectrograms',
            'augmented': Path('dataset_augmented') / 'spectrograms',
            'quiet_days.csv': Path('dataset_normal') / 'spectrograms'
        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Determine source directory
        dataset_source = row.get('dataset_source', 'v2.1_original')
        source_dir = self.source_dirs.get(dataset_source, self.source_dirs['v2.1_original'])
        
        # Handle filename
        filename = row['spectrogram_file']
        
        # Try multiple filename variations
        possible_paths = [
            source_dir / filename,
            source_dir / filename.replace('_normal_', '_'),
            source_dir / filename.replace('_aug', ''),
        ]
        
        # Find first existing file
        spec_path = None
        for path in possible_paths:
            if path.exists():
                spec_path = path
                break
        
        # Try other source directories
        if spec_path is None:
            for other_dir in self.source_dirs.values():
                for path_variant in [filename, filename.replace('_normal_', '_'), filename.replace('_aug', '')]:
                    test_path = other_dir / path_variant
                    if test_path.exists():
                        spec_path = test_path
                        break
                if spec_path is not None:
                    break
        
        if spec_path is None or not spec_path.exists():
            raise FileNotFoundError(f"Cannot find: {filename}")
        
        image = Image.open(spec_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        mag_label = row['magnitude_label']
        azi_label = row['azimuth_label']
        
        return image, (mag_label, azi_label)

# Create dataset
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = EarthquakeDataset(test_df, config['dataset_dir'], test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"✅ Test dataset created")

# ============================================================================
# STEP 4: Load Model
# ============================================================================

print(f"\n📊 Step 4: Loading trained model...")

class MultiTaskVGG16(nn.Module):
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

model = MultiTaskVGG16(len(magnitude_classes), len(azimuth_classes))

# Load checkpoint
checkpoint = torch.load(latest_exp / 'best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✅ Model loaded from: {latest_exp / 'best_model.pth'}")

# ============================================================================
# STEP 5: Evaluate on Test Set
# ============================================================================

print(f"\n📊 Step 5: Evaluating on test set...")

all_mag_preds = []
all_mag_labels = []
all_azi_preds = []
all_azi_labels = []

with torch.no_grad():
    for images, (mag_labels, azi_labels) in tqdm(test_loader, desc='Testing'):
        images = images.to(device)
        
        mag_outputs, azi_outputs = model(images)
        
        _, mag_predicted = torch.max(mag_outputs, 1)
        _, azi_predicted = torch.max(azi_outputs, 1)
        
        all_mag_preds.extend(mag_predicted.cpu().numpy())
        all_mag_labels.extend(mag_labels.numpy())
        all_azi_preds.extend(azi_predicted.cpu().numpy())
        all_azi_labels.extend(azi_labels.numpy())

print(f"✅ Evaluation complete")

# ============================================================================
# STEP 6: Calculate Metrics
# ============================================================================

print(f"\n📊 Step 6: Calculating metrics...")

# Magnitude metrics
mag_accuracy = (np.array(all_mag_preds) == np.array(all_mag_labels)).mean() * 100

# Azimuth metrics
azi_accuracy = (np.array(all_azi_preds) == np.array(all_azi_labels)).mean() * 100

# Normal class accuracy (magnitude)
normal_idx = magnitude_to_idx.get('Normal', 0)
normal_mask = np.array(all_mag_labels) == normal_idx
if normal_mask.sum() > 0:
    normal_accuracy = (np.array(all_mag_preds)[normal_mask] == np.array(all_mag_labels)[normal_mask]).mean() * 100
else:
    normal_accuracy = 0.0

print(f"\n{'='*70}")
print("TEST SET RESULTS")
print(f"{'='*70}")

print(f"\n📊 Overall Metrics:")
print(f"   Magnitude Accuracy: {mag_accuracy:.2f}%")
print(f"   Azimuth Accuracy: {azi_accuracy:.2f}%")
print(f"   Normal Class Accuracy: {normal_accuracy:.2f}%")

# ============================================================================
# STEP 7: Detailed Reports
# ============================================================================

print(f"\n📊 Step 7: Generating detailed reports...")

# Magnitude classification report
print(f"\n{'='*70}")
print("MAGNITUDE CLASSIFICATION REPORT")
print(f"{'='*70}")
unique_mag_labels = sorted(set(all_mag_labels))
mag_class_names = [magnitude_classes[i] for i in unique_mag_labels]
print(classification_report(all_mag_labels, all_mag_preds, labels=unique_mag_labels, target_names=mag_class_names, zero_division=0))

# Azimuth classification report
print(f"\n{'='*70}")
print("AZIMUTH CLASSIFICATION REPORT")
print(f"{'='*70}")
unique_azi_labels = sorted(set(all_azi_labels))
azi_class_names = [azimuth_classes[i] for i in unique_azi_labels]
print(classification_report(all_azi_labels, all_azi_preds, labels=unique_azi_labels, target_names=azi_class_names, zero_division=0))

# ============================================================================
# STEP 8: Confusion Matrices
# ============================================================================

print(f"\n📊 Step 8: Creating confusion matrices...")

# Create output directory
output_dir = latest_exp / 'test_results'
output_dir.mkdir(exist_ok=True)

# Magnitude confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm_mag = confusion_matrix(all_mag_labels, all_mag_preds)
sns.heatmap(cm_mag, annot=True, fmt='d', cmap='Blues', 
            xticklabels=magnitude_classes, yticklabels=magnitude_classes, ax=axes[0])
axes[0].set_title(f'Magnitude Confusion Matrix\nAccuracy: {mag_accuracy:.2f}%')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Azimuth confusion matrix
cm_azi = confusion_matrix(all_azi_labels, all_azi_preds)
sns.heatmap(cm_azi, annot=True, fmt='d', cmap='Greens',
            xticklabels=azimuth_classes, yticklabels=azimuth_classes, ax=axes[1])
axes[1].set_title(f'Azimuth Confusion Matrix\nAccuracy: {azi_accuracy:.2f}%')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
print(f"✅ Confusion matrices saved: {output_dir / 'confusion_matrices.png'}")

# ============================================================================
# STEP 9: Save Results
# ============================================================================

print(f"\n📊 Step 9: Saving results...")

results = {
    'test_samples': len(test_df),
    'magnitude_accuracy': float(mag_accuracy),
    'azimuth_accuracy': float(azi_accuracy),
    'normal_class_accuracy': float(normal_accuracy),
    'magnitude_classes': magnitude_classes,
    'azimuth_classes': azimuth_classes
}

with open(output_dir / 'test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved: {output_dir / 'test_results.json'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")

print(f"\n✅ EVALUATION COMPLETE!")

print(f"\n📊 Test Set Performance:")
print(f"   Total Samples: {len(test_df)}")
print(f"   Magnitude Accuracy: {mag_accuracy:.2f}%")
print(f"   Azimuth Accuracy: {azi_accuracy:.2f}%")
print(f"   Normal Class Accuracy: {normal_accuracy:.2f}%")

print(f"\n📁 Files Created:")
print(f"   1. {output_dir / 'confusion_matrices.png'}")
print(f"   2. {output_dir / 'test_results.json'}")

print(f"\n🎯 Success Criteria:")
if mag_accuracy >= 90:
    print(f"   ✅ Magnitude Accuracy >= 90%: {mag_accuracy:.2f}%")
else:
    print(f"   ⚠️  Magnitude Accuracy < 90%: {mag_accuracy:.2f}%")

if azi_accuracy >= 50:
    print(f"   ✅ Azimuth Accuracy >= 50%: {azi_accuracy:.2f}%")
else:
    print(f"   ⚠️  Azimuth Accuracy < 50%: {azi_accuracy:.2f}%")

if normal_accuracy >= 60:
    print(f"   ✅ Normal Class Accuracy >= 60%: {normal_accuracy:.2f}%")
else:
    print(f"   ⚠️  Normal Class Accuracy < 60%: {normal_accuracy:.2f}%")

print(f"\n{'='*70}")
print("✅ EVALUATION COMPLETE!")
print(f"{'='*70}")
