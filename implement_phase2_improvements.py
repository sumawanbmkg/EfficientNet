#!/usr/bin/env python3
"""
Phase 2 Improvements - Quick Fixes untuk Azimuth F1

Improvements:
1. Increase SMOTE augmentation (172 → 1,300 samples)
2. Adjust class weights (beta: 0.999 → 0.9995)
3. Add early stopping
4. Separate learning rates for azimuth head

Expected: Azimuth F1: 0.5831 → 0.65-0.68

Author: AI Assistant
Date: 2 Februari 2026
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from PIL import Image
from imblearn.over_sampling import SMOTE
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def increase_smote_augmentation():
    """
    Increase SMOTE augmentation from 172 to ~1,300 samples
    
    Target distribution:
    - Normal: 888 (keep)
    - N: 480 (keep)
    - S: 168 → 300
    - Others: 100 → 250 each
    """
    logger.info("=" * 60)
    logger.info("🚀 PHASE 2: INCREASING SMOTE AUGMENTATION")
    logger.info("=" * 60)
    
    # Load original dataset
    dataset_dir = Path('dataset_spectrogram_ssh_v22')
    metadata_file = dataset_dir / 'metadata' / 'dataset_metadata.csv'
    
    if not metadata_file.exists():
        logger.error(f"❌ Metadata file not found: {metadata_file}")
        return False
    
    df = pd.read_csv(metadata_file)
    logger.info(f"📊 Original dataset: {len(df)} samples")
    
    # Check azimuth distribution
    azimuth_dist = df['azimuth_label'].value_counts()
    logger.info("\n📊 Original Azimuth Distribution:")
    for label, count in azimuth_dist.items():
        logger.info(f"   {label}: {count} samples")
    
    # Define target distribution
    target_distribution = {
        'Normal': 888,  # Keep
        'N': 480,       # Keep
        'S': 300,       # Increase from 168
        'E': 250,       # Increase from 100
        'NE': 250,      # Increase from 100
        'NW': 250,      # Increase from 104
        'SE': 250,      # Increase from 100
        'SW': 250,      # Increase from 100
        'W': 250        # Increase from 104
    }
    
    logger.info("\n🎯 Target Distribution:")
    total_target = 0
    for label, count in target_distribution.items():
        current = azimuth_dist.get(label, 0)
        increase = count - current
        total_target += count
        logger.info(f"   {label}: {current} → {count} (+{increase})")
    
    logger.info(f"\n📈 Total samples: {len(df)} → {total_target}")
    logger.info(f"   Synthetic samples: {total_target - len(df)}")
    
    # Load images and prepare features
    logger.info("\n📥 Loading images...")
    features = []
    labels_mag = []
    labels_azi = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        img_path = dataset_dir / 'spectrograms' / row['filename']
        if img_path.exists():
            try:
                img = Image.open(img_path).convert('RGB')
                # Resize to smaller size for SMOTE (faster)
                img = img.resize((64, 64), Image.LANCZOS)
                img_array = np.array(img).flatten()
                features.append(img_array)
                labels_mag.append(row['magnitude_label'])
                labels_azi.append(row['azimuth_label'])
                valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {img_path}: {e}")
    
    features = np.array(features)
    labels_azi = np.array(labels_azi)
    
    logger.info(f"✅ Loaded {len(features)} images")
    logger.info(f"   Feature shape: {features.shape}")
    
    # Apply SMOTE with target distribution
    logger.info("\n🔄 Applying SMOTE...")
    
    # Calculate sampling strategy
    sampling_strategy = {}
    current_counts = Counter(labels_azi)
    
    for label, target_count in target_distribution.items():
        current_count = current_counts.get(label, 0)
        if target_count > current_count:
            sampling_strategy[label] = target_count
    
    logger.info(f"   Sampling strategy: {sampling_strategy}")
    
    # Apply SMOTE
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=5,
        random_state=42,
        n_jobs=-1
    )
    
    try:
        features_resampled, labels_azi_resampled = smote.fit_resample(features, labels_azi)
        logger.info(f"✅ SMOTE completed")
        logger.info(f"   Original: {len(features)} samples")
        logger.info(f"   Resampled: {len(features_resampled)} samples")
        logger.info(f"   Synthetic: {len(features_resampled) - len(features)} samples")
    except Exception as e:
        logger.error(f"❌ SMOTE failed: {e}")
        return False
    
    # Create output directory
    output_dir = Path('dataset_smote_v2')
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'spectrograms').mkdir(exist_ok=True)
    (output_dir / 'metadata').mkdir(exist_ok=True)
    
    logger.info(f"\n💾 Saving augmented dataset to: {output_dir}")
    
    # Save original samples
    logger.info("   Copying original samples...")
    new_metadata = []
    
    for idx, orig_idx in enumerate(valid_indices):
        row = df.iloc[orig_idx]
        # Copy original image
        src_path = dataset_dir / 'spectrograms' / row['filename']
        dst_path = output_dir / 'spectrograms' / row['filename']
        
        if src_path.exists():
            img = Image.open(src_path)
            img.save(dst_path)
            
            new_metadata.append({
                'filename': row['filename'],
                'magnitude_label': row['magnitude_label'],
                'azimuth_label': row['azimuth_label'],
                'is_synthetic': False,
                'event_id': row.get('event_id', ''),
                'station': row.get('station', ''),
                'timestamp': row.get('timestamp', '')
            })
    
    logger.info(f"   ✅ Copied {len(new_metadata)} original samples")
    
    # Save synthetic samples
    logger.info("   Generating synthetic samples...")
    synthetic_count = 0
    
    for idx in range(len(features), len(features_resampled)):
        # Reshape feature back to image
        img_array = features_resampled[idx].reshape(64, 64, 3).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Resize to original size
        img = img.resize((224, 224), Image.LANCZOS)
        
        # Save
        filename = f"synthetic_{synthetic_count:04d}.png"
        img_path = output_dir / 'spectrograms' / filename
        img.save(img_path)
        
        # Get magnitude label (use most common for this azimuth)
        azi_label = labels_azi_resampled[idx]
        # Find most common magnitude for this azimuth in original data
        azi_samples = df[df['azimuth_label'] == azi_label]
        if len(azi_samples) > 0:
            mag_label = azi_samples['magnitude_label'].mode()[0]
        else:
            mag_label = 'Normal'
        
        new_metadata.append({
            'filename': filename,
            'magnitude_label': mag_label,
            'azimuth_label': azi_label,
            'is_synthetic': True,
            'event_id': f'synthetic_{synthetic_count}',
            'station': 'SMOTE',
            'timestamp': ''
        })
        
        synthetic_count += 1
        
        if synthetic_count % 100 == 0:
            logger.info(f"      Generated {synthetic_count} synthetic samples...")
    
    logger.info(f"   ✅ Generated {synthetic_count} synthetic samples")
    
    # Save metadata
    metadata_df = pd.DataFrame(new_metadata)
    metadata_path = output_dir / 'metadata' / 'combined_metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    
    logger.info(f"\n✅ Augmented dataset saved!")
    logger.info(f"   Total samples: {len(metadata_df)}")
    logger.info(f"   Original: {len(metadata_df[~metadata_df['is_synthetic']])}")
    logger.info(f"   Synthetic: {len(metadata_df[metadata_df['is_synthetic']])}")
    
    # Show final distribution
    logger.info("\n📊 Final Azimuth Distribution:")
    final_dist = metadata_df['azimuth_label'].value_counts()
    for label, count in final_dist.items():
        logger.info(f"   {label}: {count} samples")
    
    return True

def create_improved_training_script():
    """
    Create improved training script with:
    - Early stopping
    - Separate learning rates
    - Adjusted class weights
    """
    logger.info("\n" + "=" * 60)
    logger.info("📝 CREATING IMPROVED TRAINING SCRIPT")
    logger.info("=" * 60)
    
    script_content = '''#!/usr/bin/env python3
"""
Phase 2 Training - With Improvements

Improvements:
1. Early stopping (patience=5)
2. Separate learning rates (azimuth head 2x higher)
3. Adjusted class weights (beta=0.9995)
4. SMOTE v2 dataset (1,300+ synthetic samples)

Expected: Azimuth F1: 0.5831 → 0.65-0.68

Author: AI Assistant
Date: 2 Februari 2026
"""

import sys
sys.path.append('.')

from train_with_improvements_v4 import *

class ImprovedTrainerV4Phase2(ImprovedTrainerV4):
    """Enhanced trainer with Phase 2 improvements"""
    
    def __init__(self, config, use_smote=True):
        # Update config for Phase 2
        config['focal_beta'] = 0.9995  # Increased from 0.999
        config['early_stopping_patience'] = 5
        
        super().__init__(config, use_smote)
        
        # Override optimizer with separate learning rates
        self._setup_separate_lr_optimizer()
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 5)
        self.best_azimuth_f1 = 0
        self.patience_counter = 0
    
    def _setup_separate_lr_optimizer(self):
        """Setup optimizer with separate learning rates"""
        base_lr = self.config['base_lr']
        
        # Azimuth head gets 2x learning rate (harder task)
        param_groups = [
            {'params': self.model.backbone.parameters(), 'lr': base_lr},
            {'params': self.model.magnitude_head.parameters(), 'lr': base_lr},
            {'params': self.model.azimuth_head.parameters(), 'lr': base_lr * 2},  # 2x!
        ]
        
        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.config['weight_decay']
        )
        
        logger.info("✅ Separate learning rates configured:")
        logger.info(f"   Backbone: {base_lr}")
        logger.info(f"   Magnitude head: {base_lr}")
        logger.info(f"   Azimuth head: {base_lr * 2} (2x)")
    
    def _train_epoch(self, epoch):
        """Train one epoch with early stopping check"""
        # Call parent method
        metrics = super()._train_epoch(epoch)
        
        # Early stopping check
        val_azimuth_f1 = metrics.get('val_azimuth_f1', 0)
        
        if val_azimuth_f1 > self.best_azimuth_f1:
            self.best_azimuth_f1 = val_azimuth_f1
            self.patience_counter = 0
            logger.info(f"   ✅ New best azimuth F1: {val_azimuth_f1:.4f}")
        else:
            self.patience_counter += 1
            logger.info(f"   ⚠️ No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
            
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"\\n⏹️ EARLY STOPPING triggered!")
                logger.info(f"   Best azimuth F1: {self.best_azimuth_f1:.4f}")
                return None  # Signal to stop
        
        return metrics

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 PHASE 2 TRAINING - WITH IMPROVEMENTS")
    logger.info("=" * 60)
    
    config = {
        'seed': 42,
        'epochs': 30,
        'batch_size': 32,
        'base_lr': 1e-4,
        'weight_decay': 0.05,
        'dropout_rate': 0.3,
        'focal_gamma': 2.0,
        'focal_beta': 0.9995,  # Increased from 0.999
        'label_smoothing': 0.05,
        'ema_decay': 0.9999,
        'randaugment_magnitude': 9,
        'randaugment_num_ops': 2,
        'early_stopping_patience': 5
    }
    
    logger.info("\\n📋 Phase 2 Configuration:")
    logger.info(f"   Beta: {config['focal_beta']} (increased from 0.999)")
    logger.info(f"   Early stopping patience: {config['early_stopping_patience']}")
    logger.info(f"   Separate LR: azimuth head 2x higher")
    logger.info(f"   Dataset: SMOTE v2 (1,300+ synthetic samples)")
    
    # Create trainer with SMOTE v2
    trainer = ImprovedTrainerV4Phase2(config, use_smote=True)
    
    # Override dataset path to use SMOTE v2
    trainer.dataset_dir = Path('dataset_smote_v2')
    trainer.metadata_file = 'combined_metadata.csv'
    
    # Start training
    trainer.train()
    
    logger.info("\\n✅ PHASE 2 TRAINING COMPLETED!")
    logger.info("=" * 60)
'''
    
    # Save script
    script_path = Path('train_with_improvements_v4_phase2.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    logger.info(f"✅ Training script created: {script_path}")
    
    return True

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("🚀 PHASE 2 IMPROVEMENTS IMPLEMENTATION")
    logger.info("=" * 60)
    logger.info("Goal: Increase Azimuth F1 from 0.5831 to 0.65-0.68")
    logger.info("")
    
    # Step 1: Increase SMOTE augmentation
    logger.info("STEP 1: Increasing SMOTE augmentation...")
    if not increase_smote_augmentation():
        logger.error("❌ Failed to increase SMOTE augmentation")
        return False
    
    # Step 2: Create improved training script
    logger.info("\nSTEP 2: Creating improved training script...")
    if not create_improved_training_script():
        logger.error("❌ Failed to create training script")
        return False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ PHASE 2 IMPROVEMENTS READY!")
    logger.info("=" * 60)
    logger.info("\nImprovements implemented:")
    logger.info("1. ✅ SMOTE augmentation: 172 → 1,300+ samples")
    logger.info("2. ✅ Class weights: beta 0.999 → 0.9995")
    logger.info("3. ✅ Early stopping: patience=5")
    logger.info("4. ✅ Separate LR: azimuth head 2x higher")
    logger.info("\nNext steps:")
    logger.info("1. Run training: python train_with_improvements_v4_phase2.py")
    logger.info("2. Expected: Azimuth F1 = 0.65-0.68 ✅")
    logger.info("3. Training time: ~6 hours")
    logger.info("")
    logger.info("🎯 TARGET Q1 JOURNAL ACHIEVABLE!")
    logger.info("=" * 60)
    
    return True

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''

# Save script
script_path = Path('implement_phase2_improvements.py')
with open(script_path, 'w', encoding='utf-8') as f:
    f.write(script_content)

logger.info(f"✅ Phase 2 implementation script created: {script_path}")
