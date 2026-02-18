#!/usr/bin/env python3
"""
Test CNN Model Loading
Simple test to verify model loading functionality
"""

import torch
import sys
import os
from multi_task_cnn_model import create_model

def test_model_loading():
    """Test model loading functionality"""
    print("Testing CNN Model Loading...")
    print("="*50)
    
    # Model config (from experiments)
    config = {
        'backbone': 'resnet50',
        'pretrained': True,
        'num_magnitude_classes': 5,
        'num_azimuth_classes': 8,
        'dropout_rate': 0.5,
        'learn_weights': True
    }
    
    print(f"Config: {config}")
    
    try:
        # Create model
        print("\n1. Creating model...")
        model, criterion = create_model(config)
        print(f"✅ Model created: {type(model).__name__}")
        
        # Check model path
        model_path = "experiments/exp_20260129_160807/best_model.pth"
        print(f"\n2. Checking model path: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"✅ Model file exists: {os.path.getsize(model_path)} bytes")
        
        # Load checkpoint
        print(f"\n3. Loading checkpoint...")
        device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=device)
        
        print(f"✅ Checkpoint loaded")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Load state dict
        print(f"\n4. Loading state dict...")
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model state dict loaded")
            if 'best_val_loss' in checkpoint:
                print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Direct state dict loaded")
        
        # Set to eval mode
        model.eval()
        print(f"✅ Model set to eval mode")
        
        # Test forward pass
        print(f"\n5. Testing forward pass...")
        batch_size = 1
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            mag_logits, az_logits = model(x)
        
        print(f"✅ Forward pass successful")
        print(f"Magnitude logits shape: {mag_logits.shape}")
        print(f"Azimuth logits shape: {az_logits.shape}")
        
        # Test predictions
        print(f"\n6. Testing predictions...")
        mag_pred = mag_logits.argmax(dim=1).item()
        az_pred = az_logits.argmax(dim=1).item()
        
        magnitude_classes = ['Small', 'Moderate', 'Medium', 'Large', 'Major']
        azimuth_classes = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        print(f"✅ Predictions successful")
        print(f"Magnitude prediction: {magnitude_classes[mag_pred]} (class {mag_pred})")
        print(f"Azimuth prediction: {azimuth_classes[az_pred]} (class {az_pred})")
        
        print(f"\n🎉 ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_model_loading()
    sys.exit(0 if success else 1)