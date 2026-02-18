#!/usr/bin/env python3
"""
Script untuk menjalankan FULL TRAINING (30 epochs) Phase 1
Setelah quick test (5 epochs) berhasil

Author: AI Assistant
Date: 1 Februari 2026
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_quick_test_results():
    """Check if quick test (5 epochs) was successful"""
    logger.info("🔍 Checking quick test results...")
    
    # Find latest experiment
    exp_dir = Path('experiments_v4')
    if not exp_dir.exists():
        logger.error("❌ No experiments_v4 directory found!")
        return False
    
    # Get latest experiment
    experiments = sorted(exp_dir.glob('exp_v4_phase1_*_smote'))
    if not experiments:
        logger.error("❌ No Phase 1 experiments found!")
        return False
    
    latest_exp = experiments[-1]
    logger.info(f"📁 Latest experiment: {latest_exp.name}")
    
    # Check training history
    history_file = latest_exp / 'training_history.csv'
    if not history_file.exists():
        logger.error("❌ No training history found!")
        return False
    
    # Read last epoch results
    import pandas as pd
    df = pd.read_csv(history_file)
    
    if len(df) == 0:
        logger.error("❌ Training history is empty!")
        return False
    
    last_epoch = df.iloc[-1]
    
    logger.info(f"\n📊 Quick Test Results (Epoch {int(last_epoch['epoch'])}):")
    logger.info(f"   Val Magnitude F1: {last_epoch['val_magnitude_f1']:.4f}")
    logger.info(f"   Val Azimuth F1: {last_epoch['val_azimuth_f1']:.4f}")
    logger.info(f"   Val Loss: {last_epoch['val_loss']:.4f}")
    
    # Check if results are good
    mag_f1 = last_epoch['val_magnitude_f1']
    azi_f1 = last_epoch['val_azimuth_f1']
    
    # Criteria for success
    success = True
    
    if mag_f1 < 0.50:
        logger.warning(f"⚠️ Magnitude F1 ({mag_f1:.4f}) is too low (expected > 0.50)")
        success = False
    
    if azi_f1 < 0.20:
        logger.warning(f"⚠️ Azimuth F1 ({azi_f1:.4f}) is too low (expected > 0.20)")
        success = False
    
    # Check if F1 is improving (not stuck)
    if len(df) >= 3:
        # Check last 3 epochs
        last_3_mag = df['val_magnitude_f1'].tail(3).values
        last_3_azi = df['val_azimuth_f1'].tail(3).values
        
        # Check if stuck (all same value)
        if len(set(last_3_mag)) == 1:
            logger.warning(f"⚠️ Magnitude F1 is stuck at {last_3_mag[0]:.4f}")
            success = False
        
        if len(set(last_3_azi)) == 1 and last_3_azi[0] < 0.1:
            logger.warning(f"⚠️ Azimuth F1 is stuck at {last_3_azi[0]:.4f}")
            success = False
    
    if success:
        logger.info("\n✅ Quick test SUCCESSFUL! Ready for full training.")
        logger.info(f"   Magnitude F1: {mag_f1:.4f} (good!)")
        logger.info(f"   Azimuth F1: {azi_f1:.4f} (improving!)")
    else:
        logger.error("\n❌ Quick test FAILED! Need to investigate.")
        logger.error("   Please check training logs and fix issues before full training.")
    
    return success

def update_config_for_full_training():
    """Update train_with_improvements_v4.py to use 30 epochs"""
    logger.info("\n🔧 Updating configuration for full training...")
    
    script_path = Path('train_with_improvements_v4.py')
    
    # Read current content
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace epochs: 5 with epochs: 30
    if "'epochs': 5" in content:
        content = content.replace("'epochs': 5", "'epochs': 30")
        logger.info("   ✅ Changed epochs from 5 to 30")
    elif "'epochs': 30" in content:
        logger.info("   ℹ️ Already set to 30 epochs")
    else:
        logger.error("   ❌ Could not find epochs configuration!")
        return False
    
    # Write back
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info("   ✅ Configuration updated successfully")
    return True

def run_full_training():
    """Run full training with 30 epochs"""
    logger.info("\n🚀 Starting FULL TRAINING (30 epochs)...")
    logger.info("=" * 60)
    logger.info("⏱️ Estimated time: ~6 hours")
    logger.info("📁 Results will be saved to: experiments_v4/")
    logger.info("=" * 60)
    
    # Run training
    import subprocess
    
    log_file = 'training_phase1_full.log'
    
    cmd = f'python train_with_improvements_v4.py 2>&1 | tee {log_file}'
    
    logger.info(f"\n📝 Training log: {log_file}")
    logger.info("🔄 Training started...\n")
    
    # Run command
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        logger.info("\n✅ FULL TRAINING COMPLETED SUCCESSFULLY!")
    else:
        logger.error(f"\n❌ Training failed with return code: {result.returncode}")
        return False
    
    return True

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("🚀 PHASE 1 FULL TRAINING LAUNCHER")
    logger.info("=" * 60)
    logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Step 1: Check quick test results
    logger.info("STEP 1: Checking quick test results...")
    if not check_quick_test_results():
        logger.error("\n❌ Quick test did not pass. Aborting full training.")
        logger.error("   Please fix issues and try again.")
        sys.exit(1)
    
    # Step 2: Update configuration
    logger.info("\nSTEP 2: Updating configuration for full training...")
    if not update_config_for_full_training():
        logger.error("\n❌ Failed to update configuration. Aborting.")
        sys.exit(1)
    
    # Step 3: Confirm with user
    logger.info("\n" + "=" * 60)
    logger.info("⚠️ READY TO START FULL TRAINING")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("  - Epochs: 30")
    logger.info("  - Estimated time: ~6 hours")
    logger.info("  - Dataset: SMOTE-augmented (2,144 samples)")
    logger.info("  - Model: ConvNeXt-Tiny with Phase 1 improvements")
    logger.info("")
    logger.info("Hyperparameters:")
    logger.info("  - Learning Rate: 1e-4")
    logger.info("  - Focal Gamma: 2.0")
    logger.info("  - Focal Beta: 0.999")
    logger.info("  - Label Smoothing: 0.05")
    logger.info("=" * 60)
    
    response = input("\n▶️ Start full training? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        logger.info("❌ Training cancelled by user.")
        sys.exit(0)
    
    # Step 4: Run full training
    logger.info("\nSTEP 3: Running full training...")
    if not run_full_training():
        logger.error("\n❌ Full training failed!")
        sys.exit(1)
    
    # Step 5: Show final results
    logger.info("\n" + "=" * 60)
    logger.info("✅ FULL TRAINING COMPLETED!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Check results: python check_phase1_results.py")
    logger.info("2. View training history: experiments_v4/[latest]/training_history.csv")
    logger.info("3. Evaluate model: python evaluate_model.py")
    logger.info("")
    logger.info("Expected results:")
    logger.info("  - Magnitude F1: 0.85-0.90 (target: 0.92)")
    logger.info("  - Azimuth F1: 0.60-0.70 (target: 0.68)")
    logger.info("=" * 60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Training interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
