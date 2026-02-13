"""
Run Experiment 3 Training (Hierarchical EfficientNet)
====================================================
Using the newly homogenized and SMOTE-balanced dataset.
"""

import sys
import os
import logging
from pathlib import Path
import shutil

# Add root project to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autoupdate_pipeline.src.trainer_v2 import ModelTrainerV2

# Configuration
TRAIN_META = 'dataset_experiment_3/final_metadata/train_exp3.csv'
VAL_META = 'dataset_experiment_3/final_metadata/val_exp3.csv'
DATASET_ROOT = 'dataset_experiment_3'
OUTPUT_DIR = 'experiments_v2/experiment_3'
EPOCHS = 40
PATIENCE = 8

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*50)
    logger.info("EXPERIMENT 3: RUN TRAINING (EFFICIENTNET-B0 + HOMOGENIZED DATA)")
    logger.info("="*50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize Trainer
    trainer = ModelTrainerV2(config={'output_dir': OUTPUT_DIR}, device_str='auto')
    
    logger.info(f"Training Meta: {TRAIN_META}")
    logger.info(f"Validation Meta: {VAL_META}")
    logger.info(f"Dataset Root: {DATASET_ROOT}")
    
    try:
        model, best_score = trainer.train(
            train_metadata_path=TRAIN_META,
            val_metadata_path=VAL_META,
            dataset_root=DATASET_ROOT,
            epochs=EPOCHS,
            early_stopping_patience=PATIENCE
        )
        
        logger.info("="*50)
        logger.info(f"EXP 3 SUCCESS! Best Score: {best_score:.4f}")
        
        # Save results summary
        summary = {
            'best_val_score': best_score,
            'dataset': 'Experiment 3 (2,600 Train Samples)',
            'base_model': 'EfficientNet-B0',
            'homogenization': '2023-2025 Normal + Historical Large'
        }
        import json
        with open(os.path.join(OUTPUT_DIR, 'exp3_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
        # The trainer saves models in its own way, but we ensure output location
        if os.path.exists('best_hierarchical_model.pth'):
            shutil.move('best_hierarchical_model.pth', os.path.join(OUTPUT_DIR, 'best_model.pth'))
            logger.info("Model preserved in experiments_v2/experiment_3/best_model.pth")

    except Exception as e:
        logger.error(f"Experiment 3 failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
