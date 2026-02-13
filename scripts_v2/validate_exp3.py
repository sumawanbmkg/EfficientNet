"""
Experiment 3 Validation Script (FIXED)
=====================================
Evaluate the newly trained Exp 3 model (Modern Data + Balanced) 
using a hierarchical validation approach.
"""

import sys
import os
import torch
import pandas as pd
import json
from pathlib import Path

# Add root project to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autoupdate_pipeline.src.trainer_v2 import HierarchicalEfficientNet, HierarchicalEarthquakeDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score

# Config
MODEL_PATH = 'experiments_v2/experiment_3/best_model.pth'
TEST_META = 'dataset_experiment_3/final_metadata/test_exp3.csv'
DATA_ROOT = 'dataset_experiment_3'
OUTPUT_FILE = 'experiments_v2/experiment_3/validation_report_exp3.json'

def validate():
    print(f"Validating Model: {MODEL_PATH}")
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # In trainer_v2, the model takes num_mag_classes, num_azi_classes
    model = HierarchicalEfficientNet(num_mag_classes=4, num_azi_classes=9)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # Load Data
    test_df = pd.read_csv(TEST_META)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = HierarchicalEarthquakeDataset(test_df, DATA_ROOT, transform=transform)
    # n_workers=0 to avoid pickling issues in validation
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_mag_preds = []
    all_mag_true = []
    
    print("Running inference...")
    with torch.no_grad():
        for images, is_pre, mag_labels, azi_labels in loader:
            images = images.to(device)
            mag_labels = mag_labels.to(device)
            
            # Forward pass returns binary_logits, mag_logits, azi_logits
            _, mag_logits, _ = model(images)
            
            _, preds = torch.max(mag_logits, 1)
            all_mag_preds.extend(preds.cpu().numpy())
            all_mag_true.extend(mag_labels.cpu().numpy())
            
    # Map back to names
    mag_classes = ['Normal', 'Moderate', 'Medium', 'Large']
    y_true = [mag_classes[i] for i in all_mag_true]
    y_pred = [mag_classes[i] for i in all_mag_preds]
    
    report = classification_report(y_true, y_pred, output_dict=True, labels=mag_classes)
    
    print("\nClassification Report (Experiment 3):")
    print(classification_report(y_true, y_pred, labels=mag_classes))
    
    # Save Report
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Report saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    validate()
