
import os
import subprocess
import time
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_step(command, description):
    logger.info(f"STARTING STEP: {description}")
    try:
        # Menjalankan script python dengan subprocess
        result = subprocess.run(["python"] + command.split(), check=True, capture_output=True, text=True)
        logger.info(f"COMPLETED: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {description}")
        logger.error(e.stdout)
        logger.error(e.stderr)
        return False

def main():
    logger.info("=== AFK MASTER ORCHESTRATOR STARTED ===")
    
    # 1. MONITORING PHASE
    logger.info("Waiting for scanners to complete...")
    # Kita cek metadata.csv sebagai indikator selesai
    normal_meta = "dataset_normal_new/metadata.csv"
    moderate_meta = "dataset_moderate/metadata.csv"
    
    timeout = 3600 * 3 # 3 jam max wait
    start_time = time.time()
    
    while True:
        if os.path.exists(normal_meta) and os.path.exists(moderate_meta):
            # Cek apakah file sudah tidak bertambah ukurannya (selesai ditulis)
            size1 = os.path.getsize(normal_meta)
            time.sleep(30)
            size2 = os.path.getsize(normal_meta)
            if size1 == size2 and size1 > 0:
                logger.info("Scanners confirmed finished.")
                break
        
        if time.time() - start_time > timeout:
            logger.error("Timeout waiting for scanners.")
            return

        time.sleep(60) # Cek tiap menit
    
    # 2. PIPELINE EXECUTION
    pipeline = [
        ("scripts_v2/1_merge_datasets.py", "Dataset Consolidation"),
        ("scripts_v2/2_create_split.py", "Data Splitting (Leakage-Free)"),
        ("scripts_v2/3_apply_balancing.py", "SMOTE Augmentation"),
        ("scripts_v2/4_run_phase2_training.py", "Model Retraining")
    ]
    
    for cmd, desc in pipeline:
        if not run_step(cmd, desc):
            logger.error(f"Pipeline ABORTED at {desc}")
            break
            
    logger.info("=== ALL PROCESSES COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
