"""
Hierarchical Inference Module
=============================
Performs inference using trained hierarchical models.

Flow:
1. Stage 1: Is this Normal or Precursor?
2. If Precursor → Stage 2: What magnitude? (Moderate/Medium/Large)
3. If Precursor → Stage 3: What direction? (8 azimuths)
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
import numpy as np

class HierarchicalEfficientNet(nn.Module):
    """EfficientNet-B0 for hierarchical classification"""
    
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class HierarchicalConvNeXt(nn.Module):
    """ConvNeXt-Tiny for hierarchical classification"""
    
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False)
        num_features = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class HierarchicalPredictor:
    """
    Hierarchical prediction using 3-stage models.
    
    Usage:
        predictor = HierarchicalPredictor('experiments_hierarchical/efficientnet_hierarchical_xxx')
        result = predictor.predict(image_path)
        # or
        result = predictor.predict_tensor(image_tensor)
    """
    
    def __init__(self, model_dir, device=None):
        self.model_dir = model_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            self.config = json.load(f)
        
        # Load class mappings
        with open(os.path.join(model_dir, 'class_mappings.json'), 'r') as f:
            self.mappings = json.load(f)
        
        # Reverse mappings for prediction
        self.binary_classes = {v: k for k, v in self.mappings['binary'].items()}
        self.magnitude_classes = {v: k for k, v in self.mappings['magnitude'].items()}
        self.azimuth_classes = {v: k for k, v in self.mappings['azimuth'].items()}
        
        # Determine backbone
        self.backbone = self.config.get('backbone', 'efficientnet')
        
        # Load models
        self._load_models()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded hierarchical {self.backbone} models from {model_dir}")
    
    def _create_model(self, num_classes):
        """Create model based on backbone type"""
        if self.backbone == 'efficientnet':
            return HierarchicalEfficientNet(num_classes)
        else:
            return HierarchicalConvNeXt(num_classes)
    
    def _load_models(self):
        """Load all three stage models"""
        # Stage 1: Binary
        self.model_s1 = self._create_model(2).to(self.device)
        self.model_s1.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'stage1_binary_best.pth'),
                      map_location=self.device)
        )
        self.model_s1.eval()
        
        # Stage 2: Magnitude
        self.model_s2 = self._create_model(3).to(self.device)
        self.model_s2.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'stage2_magnitude_best.pth'),
                      map_location=self.device)
        )
        self.model_s2.eval()
        
        # Stage 3: Azimuth
        self.model_s3 = self._create_model(8).to(self.device)
        self.model_s3.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'stage3_azimuth_best.pth'),
                      map_location=self.device)
        )
        self.model_s3.eval()
    
    def predict(self, image_path):
        """Predict from image path"""
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self.predict_tensor(tensor)
    
    def predict_tensor(self, tensor):
        """
        Predict from tensor.
        
        Returns:
            dict with keys:
                - is_precursor: bool
                - binary_confidence: float (0-1)
                - magnitude: str or None
                - magnitude_probs: dict or None
                - azimuth: str or None
                - azimuth_probs: dict or None
        """
        with torch.no_grad():
            # Stage 1: Binary classification
            output_s1 = self.model_s1(tensor)
            probs_s1 = torch.softmax(output_s1, dim=1)[0]
            pred_s1 = output_s1.argmax(dim=1).item()
            
            is_precursor = pred_s1 == 1
            binary_conf = probs_s1[pred_s1].item()
            
            result = {
                'is_precursor': is_precursor,
                'binary_class': self.binary_classes[pred_s1],
                'binary_confidence': binary_conf,
                'binary_probs': {
                    'Normal': probs_s1[0].item(),
                    'Precursor': probs_s1[1].item()
                },
                'magnitude': None,
                'magnitude_probs': None,
                'azimuth': None,
                'azimuth_probs': None
            }
            
            if is_precursor:
                # Stage 2: Magnitude classification
                output_s2 = self.model_s2(tensor)
                probs_s2 = torch.softmax(output_s2, dim=1)[0]
                pred_s2 = output_s2.argmax(dim=1).item()
                
                result['magnitude'] = self.magnitude_classes[pred_s2]
                result['magnitude_confidence'] = probs_s2[pred_s2].item()
                result['magnitude_probs'] = {
                    self.magnitude_classes[i]: probs_s2[i].item()
                    for i in range(3)
                }
                
                # Stage 3: Azimuth classification
                output_s3 = self.model_s3(tensor)
                probs_s3 = torch.softmax(output_s3, dim=1)[0]
                pred_s3 = output_s3.argmax(dim=1).item()
                
                result['azimuth'] = self.azimuth_classes[pred_s3]
                result['azimuth_confidence'] = probs_s3[pred_s3].item()
                result['azimuth_probs'] = {
                    self.azimuth_classes[i]: probs_s3[i].item()
                    for i in range(8)
                }
            
            return result
    
    def predict_batch(self, image_paths):
        """Predict for multiple images"""
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results

def find_latest_model(backbone='efficientnet'):
    """Find the latest trained hierarchical model"""
    base_dir = 'experiments_hierarchical'
    if not os.path.exists(base_dir):
        return None
    
    dirs = [d for d in os.listdir(base_dir) 
            if d.startswith(f'{backbone}_hierarchical_')]
    
    if not dirs:
        return None
    
    dirs.sort(reverse=True)
    return os.path.join(base_dir, dirs[0])

# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Path to model directory')
    parser.add_argument('--backbone', type=str, default='efficientnet',
                        choices=['efficientnet', 'convnext'])
    parser.add_argument('--image', type=str, default=None,
                        help='Path to test image')
    args = parser.parse_args()
    
    # Find model
    model_dir = args.model_dir or find_latest_model(args.backbone)
    
    if model_dir is None:
        print(f"No trained {args.backbone} hierarchical model found!")
        print("Run: python run_hierarchical_training.py first")
        exit(1)
    
    print(f"Using model: {model_dir}")
    
    # Create predictor
    predictor = HierarchicalPredictor(model_dir)
    
    # Test prediction
    if args.image:
        result = predictor.predict(args.image)
    else:
        # Find a test image
        import pandas as pd
        metadata = pd.read_csv('dataset_unified/metadata/unified_metadata.csv')
        test_row = metadata.iloc[0]
        test_path = os.path.join('dataset_unified', test_row['relative_path'])
        print(f"\nTest image: {test_path}")
        print(f"Ground truth: {test_row['magnitude_class']}, {test_row.get('azimuth_class', 'N/A')}")
        result = predictor.predict(test_path)
    
    # Print result
    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Is Precursor: {result['is_precursor']}")
    print(f"Binary Confidence: {result['binary_confidence']*100:.1f}%")
    print(f"Binary Probs: Normal={result['binary_probs']['Normal']*100:.1f}%, "
          f"Precursor={result['binary_probs']['Precursor']*100:.1f}%")
    
    if result['is_precursor']:
        print(f"\nMagnitude: {result['magnitude']} ({result['magnitude_confidence']*100:.1f}%)")
        print(f"Magnitude Probs: {', '.join([f'{k}={v*100:.1f}%' for k,v in result['magnitude_probs'].items()])}")
        print(f"\nAzimuth: {result['azimuth']} ({result['azimuth_confidence']*100:.1f}%)")
        print(f"Azimuth Probs: {', '.join([f'{k}={v*100:.1f}%' for k,v in result['azimuth_probs'].items()])}")
