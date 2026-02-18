"""
Custom Scanner - Real AI Inference Module
Untuk demonstrasi model research ke profesor

Author: Earthquake Prediction Research Team
Date: February 14, 2026
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL ARCHITECTURE (Copy from trainer_v2.py)
# ============================================================================

class HierarchicalEfficientNet(nn.Module):
    """
    EfficientNet-B0 with Hierarchical Heads:
    1. Binary Head (Gatekeeper)
    2. Magnitude Head (Conditional)
    3. Azimuth Head (Conditional)
    """
    def __init__(self, num_mag_classes=4, num_azi_classes=9, pretrained=False):
        super().__init__()
        # Load Backbone
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Feature Size
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # Remove default classifier
        
        # Shared Neck
        self.neck = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU()
        )
        
        # 1. Binary Head (Probabilitas Prekursor)
        self.binary_head = nn.Linear(256, 2)  # [Normal, Precursor]
        
        # 2. Magnitude Head
        self.mag_head = nn.Linear(256, num_mag_classes)
        
        # 3. Azimuth Head
        self.azi_head = nn.Linear(256, num_azi_classes)
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        embedding = self.neck(features)
        
        # Multi-head outputs
        binary_out = self.binary_head(embedding)
        mag_out = self.mag_head(embedding)
        azi_out = self.azi_head(embedding)
        
        return {
            'binary': binary_out,
            'magnitude': mag_out,
            'azimuth': azi_out,
            'embedding': embedding
        }

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def load_model(model_path, device='cpu'):
    """
    Load trained model from checkpoint
    
    Args:
        model_path: Path to .pth file
        device: 'cpu' or 'cuda'
    
    Returns:
        model: Loaded model in eval mode
    """
    try:
        # Initialize model
        model = HierarchicalEfficientNet(
            num_mag_classes=4,
            num_azi_classes=9,
            pretrained=False
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        logger.info(f"✅ Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def preprocess_image(image_path_or_pil, image_size=224):
    """
    Preprocess image for model input
    
    Args:
        image_path_or_pil: Path to image or PIL Image object
        image_size: Target size (default 224 for EfficientNet)
    
    Returns:
        tensor: Preprocessed image tensor [1, 3, H, W]
    """
    # Load image if path provided
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil).convert('RGB')
    else:
        image = image_path_or_pil.convert('RGB')
    
    # Define transforms (ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply transforms and add batch dimension
    tensor = transform(image).unsqueeze(0)
    
    return tensor

def run_inference(model, image_tensor, device='cpu'):
    """
    Run inference on preprocessed image
    
    Args:
        model: Loaded model
        image_tensor: Preprocessed image tensor
        device: 'cpu' or 'cuda'
    
    Returns:
        dict: Inference results with probabilities and predictions
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Get probabilities
    binary_probs = torch.softmax(outputs['binary'], dim=1)[0]
    mag_probs = torch.softmax(outputs['magnitude'], dim=1)[0]
    azi_probs = torch.softmax(outputs['azimuth'], dim=1)[0]
    
    # Get predictions
    binary_pred = torch.argmax(binary_probs).item()
    mag_pred = torch.argmax(mag_probs).item()
    azi_pred = torch.argmax(azi_probs).item()
    
    # Class names
    mag_classes = ['Normal', 'Moderate', 'Medium', 'Large']
    azi_classes = ['Normal', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    
    results = {
        'binary': {
            'prediction': 'Precursor' if binary_pred == 1 else 'Normal',
            'confidence': binary_probs[binary_pred].item() * 100,
            'prob_normal': binary_probs[0].item() * 100,
            'prob_precursor': binary_probs[1].item() * 100
        },
        'magnitude': {
            'prediction': mag_classes[mag_pred],
            'confidence': mag_probs[mag_pred].item() * 100,
            'probabilities': {
                mag_classes[i]: mag_probs[i].item() * 100 
                for i in range(len(mag_classes))
            }
        },
        'azimuth': {
            'prediction': azi_classes[azi_pred],
            'confidence': azi_probs[azi_pred].item() * 100,
            'probabilities': {
                azi_classes[i]: azi_probs[i].item() * 100 
                for i in range(len(azi_classes))
            }
        }
    }
    
    return results

def predict_single_image(model_path, image_path, device='cpu'):
    """
    Complete inference pipeline for single image
    
    Args:
        model_path: Path to model checkpoint
        image_path: Path to spectrogram image
        device: 'cpu' or 'cuda'
    
    Returns:
        dict: Complete inference results
    """
    # Load model
    model = load_model(model_path, device)
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    
    # Run inference
    results = run_inference(model, image_tensor, device)
    
    return results

# ============================================================================
# STREAMLIT INTEGRATION HELPERS
# ============================================================================

def format_results_for_dashboard(results):
    """
    Format inference results for Custom Scanner display
    
    Args:
        results: Output from run_inference()
    
    Returns:
        dict: Formatted results for dashboard
    """
    binary = results['binary']
    magnitude = results['magnitude']
    
    # Determine detection status
    if binary['prediction'] == 'Precursor':
        detected = "YES"
        prob = binary['confidence']
        est_mag = magnitude['prediction']
    else:
        detected = "NO"
        prob = binary['prob_normal']
        est_mag = "Normal"
    
    return {
        'detected': detected,
        'prob': prob,
        'est_mag': est_mag,
        'binary_confidence': binary['confidence'],
        'magnitude_confidence': magnitude['confidence'],
        'magnitude_probs': magnitude['probabilities'],
        'azimuth': results['azimuth']['prediction'],
        'azimuth_confidence': results['azimuth']['confidence']
    }

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    model_path = "experiments_v2/hierarchical/best_model.pth"
    image_path = "dataset_consolidation/spectrograms/sample.png"
    
    # Run inference
    results = predict_single_image(model_path, image_path, device='cpu')
    
    # Print results
    print("=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"\nBinary Classification:")
    print(f"  Prediction: {results['binary']['prediction']}")
    print(f"  Confidence: {results['binary']['confidence']:.2f}%")
    
    print(f"\nMagnitude Classification:")
    print(f"  Prediction: {results['magnitude']['prediction']}")
    print(f"  Confidence: {results['magnitude']['confidence']:.2f}%")
    
    print(f"\nAzimuth Classification:")
    print(f"  Prediction: {results['azimuth']['prediction']}")
    print(f"  Confidence: {results['azimuth']['confidence']:.2f}%")
    print("=" * 60)
