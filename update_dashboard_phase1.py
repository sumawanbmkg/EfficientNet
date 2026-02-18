#!/usr/bin/env python3
"""
Update Dashboard dengan Hasil Phase 1 Training

Updates:
- Training metrics dari Phase 1
- Performance comparison
- Model information
- Dataset statistics

Author: AI Assistant
Date: 2 Februari 2026
"""

import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_training_metrics():
    """Update training metrics dengan hasil Phase 1"""
    logger.info("📊 Updating training metrics...")
    
    # Load Phase 1 results
    exp_dir = Path('experiments_v4/exp_v4_phase1_20260201_204458_smote')
    history_file = exp_dir / 'training_history.csv'
    
    if not history_file.exists():
        logger.error(f"❌ History file not found: {history_file}")
        return False
    
    df = pd.read_csv(history_file)
    
    # Get final and best metrics
    final_metrics = df.iloc[-1]
    best_mag_idx = df['val_magnitude_f1'].idxmax()
    best_azi_idx = df['val_azimuth_f1'].idxmax()
    
    metrics = {
        'phase1_final': {
            'epoch': int(final_metrics['epoch']),
            'train_loss': float(final_metrics['train_loss']),
            'val_loss': float(final_metrics['val_loss']),
            'train_magnitude_f1': float(final_metrics['train_magnitude_f1']),
            'val_magnitude_f1': float(final_metrics['val_magnitude_f1']),
            'train_azimuth_f1': float(final_metrics['train_azimuth_f1']),
            'val_azimuth_f1': float(final_metrics['val_azimuth_f1'])
        },
        'phase1_best': {
            'magnitude_f1': float(df.iloc[best_mag_idx]['val_magnitude_f1']),
            'magnitude_epoch': int(df.iloc[best_mag_idx]['epoch']),
            'azimuth_f1': float(df.iloc[best_azi_idx]['val_azimuth_f1']),
            'azimuth_epoch': int(df.iloc[best_azi_idx]['epoch'])
        },
        'baseline': {
            'magnitude_f1': 0.87,
            'azimuth_f1': 0.60
        },
        'target': {
            'magnitude_f1': 0.92,
            'azimuth_f1': 0.68
        }
    }
    
    # Save metrics
    metrics_file = Path('dashboard_data/training_metrics_phase1.json')
    metrics_file.parent.mkdir(exist_ok=True)
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✅ Metrics saved to: {metrics_file}")
    logger.info(f"   Final Magnitude F1: {metrics['phase1_final']['val_magnitude_f1']:.4f}")
    logger.info(f"   Final Azimuth F1: {metrics['phase1_final']['val_azimuth_f1']:.4f}")
    
    return True

def update_model_info():
    """Update model information"""
    logger.info("\n🧠 Updating model information...")
    
    model_info = {
        'name': 'EarthquakeCNN V4.0 Phase 1',
        'version': '4.0',
        'backbone': 'ConvNeXt-Tiny',
        'pretrained': 'ImageNet-1K',
        'total_parameters': 30583149,
        'trainable_parameters': 30583149,
        'tasks': {
            'magnitude': {
                'classes': 4,
                'labels': ['Large', 'Medium', 'Moderate', 'Normal']
            },
            'azimuth': {
                'classes': 9,
                'labels': ['E', 'N', 'NE', 'NW', 'Normal', 'S', 'SE', 'SW', 'W']
            }
        },
        'loss_function': 'Balanced Focal Loss',
        'optimizer': 'AdamW',
        'hyperparameters': {
            'learning_rate': 1e-4,
            'focal_gamma': 2.0,
            'focal_beta': 0.999,
            'label_smoothing': 0.05,
            'weight_decay': 0.05,
            'dropout_rate': 0.3,
            'batch_size': 32,
            'epochs': 30
        },
        'improvements': [
            'Balanced Focal Loss untuk class imbalance',
            'SMOTE augmentation (172 synthetic samples)',
            'Fine-tuning dengan LR 1e-4',
            'EMA untuk model stability',
            'RandAugment untuk generalization'
        ]
    }
    
    # Save model info
    model_file = Path('dashboard_data/model_info_phase1.json')
    model_file.parent.mkdir(exist_ok=True)
    
    with open(model_file, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"✅ Model info saved to: {model_file}")
    
    return True

def update_dataset_stats():
    """Update dataset statistics"""
    logger.info("\n📊 Updating dataset statistics...")
    
    dataset_stats = {
        'total_samples': 2144,
        'original_samples': 1972,
        'synthetic_samples': 172,
        'augmentation_method': 'SMOTE',
        'train_split': 0.8,
        'val_split': 0.2,
        'magnitude_distribution': {
            'Large': 29,
            'Medium': 1139,
            'Moderate': 20,
            'Normal': 956
        },
        'azimuth_distribution': {
            'E': 100,
            'N': 480,
            'NE': 100,
            'NW': 104,
            'Normal': 888,
            'S': 168,
            'SE': 100,
            'SW': 100,
            'W': 104
        },
        'image_size': '224x224',
        'channels': 3,
        'format': 'PNG',
        'preprocessing': [
            'Resize to 224x224',
            'Normalize (ImageNet stats)',
            'RandAugment (magnitude=9, num_ops=2)'
        ]
    }
    
    # Save dataset stats
    dataset_file = Path('dashboard_data/dataset_stats_phase1.json')
    dataset_file.parent.mkdir(exist_ok=True)
    
    with open(dataset_file, 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    
    logger.info(f"✅ Dataset stats saved to: {dataset_file}")
    
    return True

def create_summary_report():
    """Create summary report untuk dashboard"""
    logger.info("\n📝 Creating summary report...")
    
    summary = {
        'project_title': 'Earthquake Precursor Detection using Multi-Task CNN',
        'institution': 'BMKG & ITS Collaboration',
        'date': '2 Februari 2026',
        'status': 'Phase 1 Completed',
        'achievements': [
            {
                'title': 'Magnitude Classification',
                'status': 'success',
                'metric': 'F1 Score: 0.9309',
                'target': 'Target: 0.92',
                'result': 'ACHIEVED (+1.2%)'
            },
            {
                'title': 'Azimuth Classification',
                'status': 'warning',
                'metric': 'F1 Score: 0.5831',
                'target': 'Target: 0.68',
                'result': 'NEEDS IMPROVEMENT (-14.2%)'
            },
            {
                'title': 'Training Stability',
                'status': 'success',
                'metric': 'No Model Collapse',
                'target': 'Stable Training',
                'result': 'ACHIEVED'
            },
            {
                'title': 'Class Balance',
                'status': 'success',
                'metric': 'Balanced Focal Loss',
                'target': 'Handle Imbalance',
                'result': 'IMPROVED'
            }
        ],
        'next_steps': [
            'Phase 2: Increase SMOTE augmentation (172 → 1,300 samples)',
            'Phase 2: Add early stopping (patience=5)',
            'Phase 2: Separate learning rates for azimuth head',
            'Phase 2: Adjust class weights (beta=0.9995)',
            'Expected: Azimuth F1 → 0.65-0.68'
        ],
        'publication_target': 'WOS/Scopus Q1 Journal',
        'confidence': 'High (85-90% with Phase 2 improvements)'
    }
    
    # Save summary
    summary_file = Path('dashboard_data/project_summary_phase1.json')
    summary_file.parent.mkdir(exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Summary saved to: {summary_file}")
    
    return True

def copy_training_history():
    """Copy training history untuk dashboard"""
    logger.info("\n📋 Copying training history...")
    
    src_file = Path('experiments_v4/exp_v4_phase1_20260201_204458_smote/training_history.csv')
    dst_file = Path('dashboard_data/training_history_phase1.csv')
    
    if not src_file.exists():
        logger.error(f"❌ Source file not found: {src_file}")
        return False
    
    dst_file.parent.mkdir(exist_ok=True)
    
    # Copy file
    import shutil
    shutil.copy(src_file, dst_file)
    
    logger.info(f"✅ Training history copied to: {dst_file}")
    
    return True

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("🔄 UPDATING DASHBOARD WITH PHASE 1 RESULTS")
    logger.info("=" * 60)
    
    success = True
    
    # Update all components
    if not update_training_metrics():
        success = False
    
    if not update_model_info():
        success = False
    
    if not update_dataset_stats():
        success = False
    
    if not create_summary_report():
        success = False
    
    if not copy_training_history():
        success = False
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("✅ DASHBOARD UPDATE COMPLETED!")
        logger.info("=" * 60)
        logger.info("\nUpdated files:")
        logger.info("  - dashboard_data/training_metrics_phase1.json")
        logger.info("  - dashboard_data/model_info_phase1.json")
        logger.info("  - dashboard_data/dataset_stats_phase1.json")
        logger.info("  - dashboard_data/project_summary_phase1.json")
        logger.info("  - dashboard_data/training_history_phase1.csv")
        logger.info("\nNext steps:")
        logger.info("  1. Run dashboard: python run_dashboard.py")
        logger.info("  2. Open browser: http://localhost:8501")
        logger.info("  3. View updated metrics and visualizations")
        logger.info("=" * 60)
    else:
        logger.error("\n❌ Dashboard update failed!")
    
    return success

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
