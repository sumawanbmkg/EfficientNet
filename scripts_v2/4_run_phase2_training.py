"""
Script Phase 2 - Step 4: Run Hierarchical Training
Mengeksekusi training menggunakan ModelTrainerV2 dari Auto-Update Pipeline.
Ini adalah script 'jembatan' yang menghubungkan dataset phase 2 dengan otak training baru.

Input:
- dataset_smote_train/augmented_train_metadata.csv (Training Data)
- dataset_consolidation/metadata/split_val.csv (Validation Data)

Output:
- experiments_v2/hierarchical/ (Saved Model & Logs)
"""

import sys
import os
import logging
from pathlib import Path

# Tambahkan root project ke path agar bisa import autoupdate_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autoupdate_pipeline.src.trainer_v2 import ModelTrainerV2

# Konfigurasi
TRAIN_META = 'dataset_smote_train/augmented_train_metadata.csv'
VAL_META = 'dataset_consolidation/metadata/split_val.csv'
DATASET_ROOT = 'dataset_consolidation' # Root folder gambar (termasuk yang di link dari SMOTE)
OUTPUT_DIR = 'experiments_v2/hierarchical'
EPOCHS = 50
PATIENCE = 10

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*50)
    logger.info("PHASE 2 - STEP 4: RUN TRAINING")
    logger.info("="*50)
    
    # 1. Validasi Input
    if not os.path.exists(TRAIN_META):
        logger.error(f"Train metadata not found: {TRAIN_META}. Run Step 3 first.")
        return
    if not os.path.exists(VAL_META):
        logger.error(f"Val metadata not found: {VAL_META}. Run Step 2 first.")
        return
        
    # 2. Setup Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 3. Inisialisasi Trainer
    trainer = ModelTrainerV2(config={'output_dir': OUTPUT_DIR}, device_str='auto')
    
    # 4. Mulai Training
    # Perhatikan: Dataset Root untuk SMOTE mungkin ada di 'dataset_smote_train'
    # Trainer V2 didesain pintar mencari file gambar.
    # Namun untuk amannya, kita pass working dir project sebagai root, 
    # karena path di metadata relatif terhadap project root (e.g. 'dataset_smote_train/spectrograms/...')
    
    logger.info(f"Training Data: {TRAIN_META}")
    logger.info(f"Validation Data: {VAL_META}")
    logger.info(f"Dataset Root: {os.getcwd()}") 
    
    try:
        model, best_score = trainer.train(
            train_metadata_path=TRAIN_META,
            val_metadata_path=VAL_META,
            dataset_root=os.getcwd(), # Pass current dir as root
            epochs=EPOCHS,
            early_stopping_patience=PATIENCE
        )
        
        logger.info("="*50)
        logger.info(f"TRAINING SUCCESS! Best Val F1-Mag: {best_score:.4f}")
        logger.info(f"Model saved in: best_hierarchical_model.pth")
        
        # Pindahkan model ke folder experiment
        import shutil
        shutil.move('best_hierarchical_model.pth', os.path.join(OUTPUT_DIR, 'best_model.pth'))
        logger.info(f"Model moved to: {os.path.join(OUTPUT_DIR, 'best_model.pth')}")
        
    except Exception as e:
        logger.error(f"Training Failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
