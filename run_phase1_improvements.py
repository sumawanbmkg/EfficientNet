#!/usr/bin/env python3
"""
Quick Start Script for Phase 1 Improvements
Runs all Phase 1 improvements in sequence

Steps:
1. Check dependencies
2. Run SMOTE augmentation
3. Test balanced focal loss
4. Train model with improvements
5. Compare with baseline

Author: Earthquake Prediction Research Team
Date: February 1, 2026
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required packages are installed"""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'sklearn',
        'PIL',
        'imblearn',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"  ✅ {package}")
        except ImportError:
            logger.error(f"  ❌ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"\n❌ Missing packages: {missing_packages}")
        logger.error("Please install: pip install -r requirements_v4.txt")
        return False
    
    logger.info("✅ All dependencies installed!")
    return True


def check_dataset():
    """Check if dataset exists"""
    logger.info("\n🔍 Checking dataset...")
    
    dataset_path = Path('dataset_unified')
    metadata_path = dataset_path / 'metadata' / 'unified_metadata.csv'
    
    if not dataset_path.exists():
        logger.error(f"❌ Dataset not found: {dataset_path}")
        return False
    
    if not metadata_path.exists():
        logger.error(f"❌ Metadata not found: {metadata_path}")
        return False
    
    logger.info(f"✅ Dataset found: {dataset_path}")
    return True


def run_smote_augmentation():
    """Run SMOTE augmentation"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: SMOTE DATA AUGMENTATION")
    logger.info("=" * 60)
    
    try:
        subprocess.run([sys.executable, 'implement_smote_augmentation.py'], check=True)
        logger.info("✅ SMOTE augmentation completed!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ SMOTE augmentation failed: {e}")
        return False


def test_balanced_focal_loss():
    """Test balanced focal loss"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: TEST BALANCED FOCAL LOSS")
    logger.info("=" * 60)
    
    try:
        subprocess.run([sys.executable, 'implement_balanced_focal_loss.py'], check=True)
        logger.info("✅ Balanced focal loss test passed!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Balanced focal loss test failed: {e}")
        return False


def train_with_improvements():
    """Train model with improvements"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: TRAIN MODEL WITH IMPROVEMENTS")
    logger.info("=" * 60)
    
    try:
        subprocess.run([sys.executable, 'train_with_improvements_v4.py'], check=True)
        logger.info("✅ Training completed!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training failed: {e}")
        return False


def compare_with_baseline():
    """Compare results with baseline"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: COMPARE WITH BASELINE")
    logger.info("=" * 60)
    
    # Check if baseline results exist
    baseline_path = Path('experiments_v3')
    improved_path = Path('experiments_v4')
    
    if not baseline_path.exists():
        logger.warning("⚠️  Baseline results not found")
        logger.info("Run baseline training first: python train_earthquake_v3.py")
        return False
    
    if not improved_path.exists():
        logger.error("❌ Improved results not found")
        return False
    
    # Find latest experiments
    baseline_exps = sorted(baseline_path.glob('exp_v3_*'))
    improved_exps = sorted(improved_path.glob('exp_v4_*'))
    
    if not baseline_exps or not improved_exps:
        logger.warning("⚠️  No experiments found for comparison")
        return False
    
    # Load and compare results
    import pandas as pd
    
    baseline_history = pd.read_csv(baseline_exps[-1] / 'training_history.csv')
    improved_history = pd.read_csv(improved_exps[-1] / 'training_history.csv')
    
    logger.info("\n📊 PERFORMANCE COMPARISON")
    logger.info("=" * 60)
    
    # Best validation F1 scores
    baseline_mag_f1 = baseline_history['val_magnitude_f1'].max()
    baseline_az_f1 = baseline_history['val_azimuth_f1'].max()
    
    improved_mag_f1 = improved_history['val_magnitude_f1'].max()
    improved_az_f1 = improved_history['val_azimuth_f1'].max()
    
    logger.info(f"\n🎯 Magnitude F1:")
    logger.info(f"  Baseline: {baseline_mag_f1:.4f}")
    logger.info(f"  Improved: {improved_mag_f1:.4f}")
    logger.info(f"  Gain: {(improved_mag_f1 - baseline_mag_f1):.4f} ({(improved_mag_f1/baseline_mag_f1 - 1)*100:+.1f}%)")
    
    logger.info(f"\n🧭 Azimuth F1:")
    logger.info(f"  Baseline: {baseline_az_f1:.4f}")
    logger.info(f"  Improved: {improved_az_f1:.4f}")
    logger.info(f"  Gain: {(improved_az_f1 - baseline_az_f1):.4f} ({(improved_az_f1/baseline_az_f1 - 1)*100:+.1f}%)")
    
    # Check if target improvement achieved
    mag_improvement = (improved_mag_f1 / baseline_mag_f1 - 1) * 100
    az_improvement = (improved_az_f1 / baseline_az_f1 - 1) * 100
    
    logger.info(f"\n🎯 TARGET CHECK:")
    logger.info(f"  Magnitude improvement: {mag_improvement:+.1f}% (target: +5%)")
    logger.info(f"  Azimuth improvement: {az_improvement:+.1f}% (target: +10%)")
    
    if mag_improvement >= 5 and az_improvement >= 10:
        logger.info("  ✅ TARGET ACHIEVED!")
    else:
        logger.info("  ⚠️  Target not fully achieved, but progress made")
    
    return True


def main():
    """Main function"""
    logger.info("🚀 PHASE 1 IMPROVEMENTS - QUICK START")
    logger.info("=" * 60)
    logger.info("This script will:")
    logger.info("1. Check dependencies")
    logger.info("2. Run SMOTE augmentation")
    logger.info("3. Test balanced focal loss")
    logger.info("4. Train model with improvements")
    logger.info("5. Compare with baseline")
    logger.info("=" * 60)
    
    # Step 0: Check dependencies
    if not check_dependencies():
        logger.error("\n❌ Dependency check failed!")
        logger.error("Install dependencies: pip install -r requirements_v4.txt")
        return
    
    # Step 0.5: Check dataset
    if not check_dataset():
        logger.error("\n❌ Dataset check failed!")
        logger.error("Make sure dataset_unified/ exists with metadata")
        return
    
    # Step 1: SMOTE augmentation
    logger.info("\n" + "=" * 60)
    logger.info("Starting Phase 1 improvements...")
    logger.info("=" * 60)
    
    if not run_smote_augmentation():
        logger.error("\n❌ SMOTE augmentation failed!")
        logger.error("Check implement_smote_augmentation.py for errors")
        return
    
    # Step 2: Test balanced focal loss
    if not test_balanced_focal_loss():
        logger.error("\n❌ Balanced focal loss test failed!")
        return
    
    # Step 3: Train with improvements
    if not train_with_improvements():
        logger.error("\n❌ Training failed!")
        return
    
    # Step 4: Compare with baseline
    compare_with_baseline()
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ PHASE 1 IMPROVEMENTS COMPLETED!")
    logger.info("=" * 60)
    logger.info("\n📁 Output directories:")
    logger.info("  - dataset_smote/: SMOTE-augmented dataset")
    logger.info("  - experiments_v4/: Training results")
    logger.info("\n🎯 Next steps:")
    logger.info("  1. Review training curves in experiments_v4/")
    logger.info("  2. Analyze performance improvements")
    logger.info("  3. Proceed to Phase 2 (Architecture improvements)")
    logger.info("\n📚 Documentation:")
    logger.info("  - MODEL_IMPROVEMENT_RECOMMENDATIONS_Q1.md")
    logger.info("  - QUICK_IMPLEMENTATION_GUIDE.md")
    logger.info("  - RINGKASAN_REKOMENDASI_PENINGKATAN_MODEL.md")


if __name__ == '__main__':
    main()
