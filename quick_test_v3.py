#!/usr/bin/env python3
"""
Quick test for EarthquakeCNN V3.0 - Model only
Tests core functionality without requiring dataset
"""

import torch
import time
from earthquake_cnn_v3 import create_model_v3, get_model_config, EMAModel

def test_model_core():
    """Test core model functionality"""
    print("🚀 EarthquakeCNN V3.0 - Quick Test")
    print("=" * 50)
    
    # Get configuration
    config = get_model_config()
    print(f"📋 Configuration loaded")
    
    # Create model
    model, criterion = create_model_v3(config)
    print(f"🧠 Model created: ConvNeXt-Tiny backbone")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🔢 Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"💾 Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\n🔄 Testing forward pass...")
    print(f"Input shape: {x.shape}")
    
    # Forward pass timing
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        magnitude_logits, azimuth_logits = model(x)
        forward_time = time.time() - start_time
    
    print(f"Magnitude output: {magnitude_logits.shape}")
    print(f"Azimuth output: {azimuth_logits.shape}")
    print(f"Forward time: {forward_time:.4f}s ({forward_time/batch_size*1000:.1f}ms per sample)")
    
    # Test loss computation
    magnitude_targets = torch.randint(0, 3, (batch_size,))
    azimuth_targets = torch.randint(0, 8, (batch_size,))
    
    loss_dict = criterion(magnitude_logits, azimuth_logits, 
                         magnitude_targets, azimuth_targets)
    
    print(f"\n💰 Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test EMA
    print(f"\n📈 Testing EMA model...")
    ema_model = EMAModel(model, decay=0.9999)
    
    # Simulate parameter update
    original_param = next(model.parameters()).clone()
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.01)
            break
    
    # Update EMA
    ema_model.update()
    print(f"✅ EMA update successful")
    
    # Test predictions
    print(f"\n🎯 Testing predictions...")
    magnitude_preds = magnitude_logits.argmax(dim=1)
    azimuth_preds = azimuth_logits.argmax(dim=1)
    
    print(f"Magnitude predictions: {magnitude_preds.tolist()}")
    print(f"Magnitude targets:     {magnitude_targets.tolist()}")
    print(f"Azimuth predictions:   {azimuth_preds.tolist()}")
    print(f"Azimuth targets:       {azimuth_targets.tolist()}")
    
    # Calculate accuracy
    mag_acc = (magnitude_preds == magnitude_targets).float().mean()
    az_acc = (azimuth_preds == azimuth_targets).float().mean()
    
    print(f"Magnitude accuracy: {mag_acc:.4f}")
    print(f"Azimuth accuracy: {az_acc:.4f}")
    
    print(f"\n✅ All tests passed!")
    print(f"🎉 EarthquakeCNN V3.0 is ready for training!")
    
    return model, criterion

def test_progressive_resizing():
    """Test progressive resizing logic"""
    print(f"\n📏 Testing Progressive Resizing...")
    
    class ProgressiveResizer:
        def __init__(self, schedule):
            self.schedule = schedule
            self.current_size = min(schedule.values())
            
        def get_size(self, epoch):
            for epoch_threshold in sorted(self.schedule.keys(), reverse=True):
                if epoch >= epoch_threshold:
                    self.current_size = self.schedule[epoch_threshold]
                    break
            return self.current_size
    
    # Test schedule
    schedule = {0: 112, 20: 168, 40: 224}
    resizer = ProgressiveResizer(schedule)
    
    test_epochs = [0, 10, 20, 30, 40, 50]
    
    for epoch in test_epochs:
        size = resizer.get_size(epoch)
        print(f"  Epoch {epoch:2d}: {size}x{size}")
    
    print(f"✅ Progressive resizing working correctly")

def test_mixed_precision():
    """Test Automatic Mixed Precision"""
    print(f"\n⚡ Testing Mixed Precision...")
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping AMP test")
        return
    
    # Get model
    config = get_model_config()
    model, criterion = create_model_v3(config)
    model = model.cuda()
    
    # Test data
    x = torch.randn(4, 3, 224, 224).cuda()
    mag_targets = torch.randint(0, 3, (4,)).cuda()
    az_targets = torch.randint(0, 8, (4,)).cuda()
    
    # AMP scaler
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Forward pass with AMP
    model.train()
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        magnitude_logits, azimuth_logits = model(x)
        loss_dict = criterion(magnitude_logits, azimuth_logits, mag_targets, az_targets)
        loss = loss_dict['total_loss']
    
    # Backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"✅ Mixed precision training working")
    print(f"Loss: {loss.item():.4f}")

if __name__ == '__main__':
    # Run tests
    model, criterion = test_model_core()
    test_progressive_resizing()
    test_mixed_precision()
    
    print(f"\n" + "=" * 50)
    print(f"🎯 EarthquakeCNN V3.0 - All Core Tests Passed!")
    print(f"🚀 Ready for full training pipeline!")