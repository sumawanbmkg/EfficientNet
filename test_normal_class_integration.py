#!/usr/bin/env python3
"""
Test script to verify Normal class integration in V3.0

This script verifies:
1. Dataset includes Normal class
2. Model is configured for 4 magnitude + 9 azimuth classes
3. Class weights are computed correctly
4. Data splits maintain Normal class distribution

Author: Earthquake Prediction Research Team
Date: 31 January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch

from earthquake_cnn_v3 import create_model_v3, get_model_config
from earthquake_dataset_v3 import EarthquakeDatasetV3


def test_dataset_normal_class():
    """Test that dataset includes Normal class"""
    print("=" * 80)
    print("🧪 TEST 1: Dataset Normal Class Integration")
    print("=" * 80)
    
    # Load metadata
    metadata_path = 'dataset_unified/metadata/unified_metadata.csv'
    metadata = pd.read_csv(metadata_path)
    
    print(f"\n📊 Total records: {len(metadata)}")
    
    # Check magnitude classes
    print(f"\n📈 Magnitude classes:")
    magnitude_counts = metadata['magnitude_class'].value_counts()
    for cls, count in magnitude_counts.items():
        percentage = (count / len(metadata)) * 100
        print(f"  {cls:12s}: {count:4d} ({percentage:5.1f}%)")
    
    # Check azimuth classes
    print(f"\n🧭 Azimuth classes:")
    azimuth_counts = metadata['azimuth_class'].value_counts()
    for cls, count in azimuth_counts.items():
        percentage = (count / len(metadata)) * 100
        print(f"  {cls:12s}: {count:4d} ({percentage:5.1f}%)")
    
    # Verify Normal class exists
    assert 'Normal' in magnitude_counts.index, "❌ Normal class not found in magnitude!"
    assert 'Normal' in azimuth_counts.index, "❌ Normal class not found in azimuth!"
    
    print(f"\n✅ TEST 1 PASSED: Normal class found in both magnitude and azimuth")
    print(f"   - Magnitude Normal: {magnitude_counts['Normal']} samples")
    print(f"   - Azimuth Normal: {azimuth_counts['Normal']} samples")
    
    return True


def test_model_configuration():
    """Test that model is configured for correct number of classes"""
    print("\n" + "=" * 80)
    print("🧪 TEST 2: Model Configuration")
    print("=" * 80)
    
    # Get model config
    config = get_model_config()
    
    print(f"\n🔧 Model configuration:")
    print(f"  Magnitude classes: {config['num_magnitude_classes']}")
    print(f"  Azimuth classes: {config['num_azimuth_classes']}")
    
    # Verify correct number of classes
    assert config['num_magnitude_classes'] == 4, \
        f"❌ Expected 4 magnitude classes, got {config['num_magnitude_classes']}"
    assert config['num_azimuth_classes'] == 9, \
        f"❌ Expected 9 azimuth classes, got {config['num_azimuth_classes']}"
    
    # Create model
    model, criterion = create_model_v3(config)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    magnitude_logits, azimuth_logits = model(x)
    
    print(f"\n🧠 Model architecture:")
    print(f"  Input shape: {x.shape}")
    print(f"  Magnitude output: {magnitude_logits.shape}")
    print(f"  Azimuth output: {azimuth_logits.shape}")
    
    # Verify output shapes
    assert magnitude_logits.shape == (batch_size, 4), \
        f"❌ Expected magnitude output shape (4, 4), got {magnitude_logits.shape}"
    assert azimuth_logits.shape == (batch_size, 9), \
        f"❌ Expected azimuth output shape (4, 9), got {azimuth_logits.shape}"
    
    print(f"\n✅ TEST 2 PASSED: Model configured correctly")
    print(f"   - Magnitude output: {magnitude_logits.shape[1]} classes")
    print(f"   - Azimuth output: {azimuth_logits.shape[1]} classes")
    
    return True


def test_dataset_loader():
    """Test that dataset loader handles Normal class correctly"""
    print("\n" + "=" * 80)
    print("🧪 TEST 3: Dataset Loader")
    print("=" * 80)
    
    # Create dataset
    dataset = EarthquakeDatasetV3(
        'dataset_unified',
        split='train',
        image_size=224
    )
    
    print(f"\n📦 Dataset created:")
    print(f"  Total samples: {len(dataset)}")
    
    # Check class mappings
    print(f"\n🏷️ Magnitude class mapping:")
    for cls, idx in sorted(dataset.magnitude_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx}: {cls}")
    
    print(f"\n🏷️ Azimuth class mapping:")
    for cls, idx in sorted(dataset.azimuth_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx}: {cls}")
    
    # Verify Normal class in mappings
    assert 'Normal' in dataset.magnitude_to_idx, \
        "❌ Normal class not found in magnitude mapping!"
    assert 'Normal' in dataset.azimuth_to_idx, \
        "❌ Normal class not found in azimuth mapping!"
    
    # Verify correct number of classes
    assert len(dataset.magnitude_to_idx) == 4, \
        f"❌ Expected 4 magnitude classes, got {len(dataset.magnitude_to_idx)}"
    assert len(dataset.azimuth_to_idx) == 9, \
        f"❌ Expected 9 azimuth classes, got {len(dataset.azimuth_to_idx)}"
    
    # Get class distribution
    class_counts = dataset.get_class_counts()
    
    print(f"\n📊 Class distribution in training set:")
    print(f"  Magnitude:")
    for cls, count in sorted(class_counts['magnitude'].items()):
        print(f"    {cls:12s}: {count:4d}")
    print(f"  Azimuth:")
    for cls, count in sorted(class_counts['azimuth'].items()):
        print(f"    {cls:12s}: {count:4d}")
    
    # Test loading a sample
    image, mag_label, az_label = dataset[0]
    
    print(f"\n🖼️ Sample data:")
    print(f"  Image shape: {image.shape}")
    print(f"  Magnitude label: {mag_label} ({dataset.idx_to_magnitude[mag_label]})")
    print(f"  Azimuth label: {az_label} ({dataset.idx_to_azimuth[az_label]})")
    
    print(f"\n✅ TEST 3 PASSED: Dataset loader handles Normal class correctly")
    print(f"   - Magnitude classes: {len(dataset.magnitude_to_idx)}")
    print(f"   - Azimuth classes: {len(dataset.azimuth_to_idx)}")
    
    return True


def test_class_weights():
    """Test that class weights are computed correctly"""
    print("\n" + "=" * 80)
    print("🧪 TEST 4: Class Weight Computation")
    print("=" * 80)
    
    # Load metadata
    metadata_path = 'dataset_unified/metadata/unified_metadata.csv'
    metadata = pd.read_csv(metadata_path)
    
    # Clean data
    metadata = metadata.dropna(subset=['magnitude_class', 'azimuth_class'])
    metadata['magnitude_class'] = metadata['magnitude_class'].astype(str)
    metadata['azimuth_class'] = metadata['azimuth_class'].astype(str)
    metadata = metadata[
        (metadata['magnitude_class'] != 'nan') & 
        (metadata['azimuth_class'] != 'nan')
    ]
    
    # Get labels
    magnitude_labels = metadata['magnitude_class'].astype(str).values
    azimuth_labels = metadata['azimuth_class'].astype(str).values
    
    # Get unique classes
    magnitude_classes = np.unique(magnitude_labels)
    azimuth_classes = np.unique(azimuth_labels)
    
    print(f"\n📊 Detected classes:")
    print(f"  Magnitude: {magnitude_classes}")
    print(f"  Azimuth: {azimuth_classes}")
    
    # Verify Normal class is detected
    assert 'Normal' in magnitude_classes, "❌ Normal not detected in magnitude classes!"
    assert 'Normal' in azimuth_classes, "❌ Normal not detected in azimuth classes!"
    
    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    
    magnitude_weights = compute_class_weight(
        'balanced', classes=magnitude_classes, y=magnitude_labels
    )
    azimuth_weights = compute_class_weight(
        'balanced', classes=azimuth_classes, y=azimuth_labels
    )
    
    print(f"\n⚖️ Computed class weights:")
    print(f"  Magnitude:")
    for cls, weight in zip(magnitude_classes, magnitude_weights):
        print(f"    {cls:12s}: {weight:6.2f}")
    
    print(f"  Azimuth:")
    for cls, weight in zip(azimuth_classes, azimuth_weights):
        print(f"    {cls:12s}: {weight:6.2f}")
    
    # Verify Normal class has reasonable weight
    normal_mag_idx = np.where(magnitude_classes == 'Normal')[0][0]
    normal_az_idx = np.where(azimuth_classes == 'Normal')[0][0]
    
    normal_mag_weight = magnitude_weights[normal_mag_idx]
    normal_az_weight = azimuth_weights[normal_az_idx]
    
    print(f"\n🎯 Normal class weights:")
    print(f"  Magnitude: {normal_mag_weight:.2f}")
    print(f"  Azimuth: {normal_az_weight:.2f}")
    
    # Verify weights are reasonable (not too high or too low)
    # Note: Normal is a common class, so weight can be < 1.0
    assert 0.1 < normal_mag_weight < 5.0, \
        f"❌ Normal magnitude weight seems unreasonable: {normal_mag_weight}"
    assert 0.1 < normal_az_weight < 5.0, \
        f"❌ Normal azimuth weight seems unreasonable: {normal_az_weight}"
    
    print(f"\n✅ TEST 4 PASSED: Class weights computed correctly")
    print(f"   - Normal magnitude weight: {normal_mag_weight:.2f}")
    print(f"   - Normal azimuth weight: {normal_az_weight:.2f}")
    
    return True


def test_data_splits():
    """Test that data splits maintain Normal class distribution"""
    print("\n" + "=" * 80)
    print("🧪 TEST 5: Data Split Distribution")
    print("=" * 80)
    
    # Create datasets for all splits
    train_dataset = EarthquakeDatasetV3('dataset_unified', split='train')
    val_dataset = EarthquakeDatasetV3('dataset_unified', split='val')
    test_dataset = EarthquakeDatasetV3('dataset_unified', split='test')
    
    print(f"\n📊 Split sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} samples")
    
    # Get class distributions
    train_counts = train_dataset.get_class_counts()
    val_counts = val_dataset.get_class_counts()
    test_counts = test_dataset.get_class_counts()
    
    print(f"\n📈 Normal class distribution:")
    train_normal = train_counts['magnitude'].get('Normal', 0)
    val_normal = val_counts['magnitude'].get('Normal', 0)
    test_normal = test_counts['magnitude'].get('Normal', 0)
    total_normal = train_normal + val_normal + test_normal
    
    print(f"  Train: {train_normal} ({train_normal/total_normal*100:.1f}%)")
    print(f"  Val:   {val_normal} ({val_normal/total_normal*100:.1f}%)")
    print(f"  Test:  {test_normal} ({test_normal/total_normal*100:.1f}%)")
    print(f"  Total: {total_normal}")
    
    # Verify Normal class exists in all splits
    assert train_normal > 0, "❌ No Normal samples in training set!"
    assert val_normal > 0, "❌ No Normal samples in validation set!"
    assert test_normal > 0, "❌ No Normal samples in test set!"
    
    # Verify reasonable distribution (train should be ~68%, val ~17%, test ~15%)
    train_ratio = train_normal / total_normal
    val_ratio = val_normal / total_normal
    test_ratio = test_normal / total_normal
    
    assert 0.60 < train_ratio < 0.75, \
        f"❌ Train ratio seems off: {train_ratio:.2%}"
    assert 0.10 < val_ratio < 0.25, \
        f"❌ Val ratio seems off: {val_ratio:.2%}"
    assert 0.10 < test_ratio < 0.25, \
        f"❌ Test ratio seems off: {test_ratio:.2%}"
    
    print(f"\n✅ TEST 5 PASSED: Data splits maintain Normal class distribution")
    print(f"   - Train: {train_ratio:.1%}")
    print(f"   - Val: {val_ratio:.1%}")
    print(f"   - Test: {test_ratio:.1%}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🚀 NORMAL CLASS INTEGRATION TEST SUITE")
    print("=" * 80)
    print("\nTesting Normal class integration in EarthquakeCNN V3.0")
    print("This verifies that the model is ready to train with Normal class included.\n")
    
    tests = [
        ("Dataset Normal Class", test_dataset_normal_class),
        ("Model Configuration", test_model_configuration),
        ("Dataset Loader", test_dataset_loader),
        ("Class Weights", test_class_weights),
        ("Data Splits", test_data_splits)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\n{'=' * 80}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Normal class integration is complete and working correctly.")
        print("✅ Model is ready to train with Normal class included.")
        print("\n🚀 Next step: Run training with:")
        print("   python train_earthquake_v3.py")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed!")
        print("Please fix the issues before training.")
    
    print("=" * 80)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
