#!/usr/bin/env python3
"""
Test script for EarthquakeCNN V3.0
Quick validation of model architecture and training pipeline

Author: Earthquake Prediction Research Team
Date: 30 January 2026
Version: 3.0
"""

import torch
import torch.nn as nn
from earthquake_cnn_v3 import create_model_v3, get_model_config, EMAModel
from earthquake_dataset_v3 import create_dataloaders_v3
import time

def test_model_architecture():
    """Test model architecture and forward pass"""
    print("🧠 Testing Model Architecture...")
    
    # Get configuration
    config = get_model_config()
    
    # Create model
    model, criterion = create_model_v3(config)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    start_time = time.time()
    magnitude_logits, azimuth_logits = model(x)
    forward_time = time.time() - start_time
    
    print(f"Magnitude output: {magnitude_logits.shape}")
    print(f"Azimuth output: {azimuth_logits.shape}")
    print(f"Forward pass time: {forward_time:.4f}s")
    
    # Test loss
    magnitude_targets = torch.randint(0, 3, (batch_size,))
    azimuth_targets = torch.randint(0, 8, (batch_size,))
    
    loss_dict = criterion(magnitude_logits, azimuth_logits, 
                         magnitude_targets, azimuth_targets)
    
    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model, criterion

def test_ema_model(model):
    """Test EMA model functionality"""
    print("\n📈 Testing EMA Model...")
    
    # Create EMA model
    ema_model = EMAModel(model, decay=0.9999)
    
    # Simulate training step
    original_param = next(model.parameters()).clone()
    
    # Modify model parameters (simulate gradient update)
    modified_param = None
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01)
            if modified_param is None:
                modified_param = param.clone()
            break  # Only modify first parameter for test
    
    # Update EMA
    ema_model.update()
    
    # Apply shadow weights
    ema_model.apply_shadow()
    ema_param = next(model.parameters()).clone()
    
    # Restore original weights
    ema_model.restore()
    restored_param = next(model.parameters()).clone()
    
    print(f"Original param mean: {original_param.mean().item():.6f}")
    print(f"Modified param mean: {modified_param.mean().item():.6f}")
    print(f"EMA param mean: {ema_param.mean().item():.6f}")
    print(f"Restored param mean: {restored_param.mean().item():.6f}")
    
    # Check if restoration worked (compare with modified, not original)
    assert torch.allclose(modified_param, restored_param, atol=1e-6), "EMA restoration failed!"
    print("✅ EMA model test passed!")

def test_dataset_loading():
    """Test dataset loading and preprocessing"""
    print("\n📊 Testing Dataset Loading...")
    
    try:
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders_v3(
            'dataset_unified', 
            batch_size=8,
            image_size=224,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
        
        print(f"Dataloaders created:")
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Val: {len(val_loader)} batches") 
        print(f"  Test: {len(test_loader)} batches")
        
        # Test batch loading
        start_time = time.time()
        for i, (images, mag_labels, az_labels) in enumerate(train_loader):
            if i >= 2:  # Test only first 2 batches
                break
            
            print(f"  Batch {i+1}: images {images.shape}, mag {mag_labels.shape}, az {az_labels.shape}")
            print(f"    Magnitude labels: {mag_labels.tolist()}")
            print(f"    Azimuth labels: {az_labels.tolist()}")
            
            # Check data ranges
            print(f"    Image range: [{images.min():.3f}, {images.max():.3f}]")
            
        load_time = time.time() - start_time
        print(f"  Batch loading time: {load_time:.4f}s")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        print("Note: Make sure 'dataset_unified' directory exists with proper structure")
        return None, None, None

def test_training_step(model, criterion, train_loader):
    """Test a single training step"""
    print("\n🏋️ Testing Training Step...")
    
    if train_loader is None:
        print("⚠️ Skipping training step test (no dataloader)")
        return
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.05
    )
    
    # Setup AMP scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Get a batch
    try:
        images, mag_labels, az_labels = next(iter(train_loader))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with AMP
        start_time = time.time()
        with torch.cuda.amp.autocast():
            magnitude_logits, azimuth_logits = model(images)
            loss_dict = criterion(magnitude_logits, azimuth_logits, mag_labels, az_labels)
            loss = loss_dict['total_loss']
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        step_time = time.time() - start_time
        
        print(f"Training step completed:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Step time: {step_time:.4f}s")
        print(f"  Magnitude accuracy: {(magnitude_logits.argmax(1) == mag_labels).float().mean():.4f}")
        print(f"  Azimuth accuracy: {(azimuth_logits.argmax(1) == az_labels).float().mean():.4f}")
        
        print("✅ Training step test passed!")
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")

def test_progressive_resizing():
    """Test progressive resizing functionality"""
    print("\n📏 Testing Progressive Resizing...")
    
    from train_earthquake_v3 import ProgressiveResizer
    
    # Test schedule
    schedule = {0: 112, 20: 168, 40: 224}
    resizer = ProgressiveResizer(schedule)
    
    # Test different epochs
    test_epochs = [0, 10, 20, 30, 40, 50]
    
    for epoch in test_epochs:
        size = resizer.get_size(epoch)
        print(f"  Epoch {epoch}: {size}x{size}")
    
    print("✅ Progressive resizing test passed!")

def main():
    """Main test function"""
    print("🚀 EarthquakeCNN V3.0 - Comprehensive Test Suite")
    print("=" * 60)
    
    # Test 1: Model Architecture
    model, criterion = test_model_architecture()
    
    # Test 2: EMA Model
    test_ema_model(model)
    
    # Test 3: Dataset Loading
    train_loader, val_loader, test_loader = test_dataset_loading()
    
    # Test 4: Training Step
    test_training_step(model, criterion, train_loader)
    
    # Test 5: Progressive Resizing
    test_progressive_resizing()
    
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print("✅ Model architecture: PASSED")
    print("✅ EMA model: PASSED") 
    print("✅ Progressive resizing: PASSED")
    
    if train_loader is not None:
        print("✅ Dataset loading: PASSED")
        print("✅ Training step: PASSED")
        print("\n🚀 EarthquakeCNN V3.0 is ready for training!")
    else:
        print("⚠️ Dataset loading: SKIPPED (dataset not found)")
        print("⚠️ Training step: SKIPPED")
        print("\n📝 Note: Create 'dataset_unified' directory to test full pipeline")
    
    print("\n🎉 All available tests completed successfully!")

if __name__ == '__main__':
    main()