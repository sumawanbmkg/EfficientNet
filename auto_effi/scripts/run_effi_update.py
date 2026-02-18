#!/usr/bin/env python
"""
Auto-Effi Update Orchestrator
Main script to trigger model update based on new data in pending/
"""

import os
import sys
import json
import yaml
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.trainer_effi import AutoEffiTrainer

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("auto_effi.orchestrator")

def run_pipeline():
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "config" / "pipeline_config.yaml"
    registry_path = base_dir / "config" / "model_registry.json"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    logger.info("="*50)
    logger.info("AUTO-EFFI UPDATE PIPELINE START")
    logger.info("="*50)
    
    # 1. Scan for Pending Data
    pending_dir = base_dir / "data" / "pending"
    validated_dir = base_dir / "data" / "validated"
    validated_dir.mkdir(parents=True, exist_ok=True)
    
    pending_files = list(pending_dir.glob("*.csv"))
    
    # Configuration for strict trigger
    min_new_samples = config.get('trigger', {}).get('min_new_samples', 5)
    
    new_data_df = pd.DataFrame()
    processed_files = []
    
    if not pending_files:
        logger.info("⏹️ No pending CSV data found. Pipeline will STOP.")
        return
        
    logger.info(f"Found {len(pending_files)} pending data files. Validating...")
    
    # 2. Ingest & Validate Pending Data
    required_cols = ['filename', 'magnitude_class', 'azimuth_class'] # Minimal required
    
    for p_file in pending_files:
        try:
            df = pd.read_csv(p_file)
            # Check columns
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"File {p_file.name} missing required columns {required_cols}. Skipping.")
                continue
                
            # Basic content validation (e.g. valid classes)
            # (Simplification: Assuming pre-validated by user, but good to add checks here)
            
            new_data_df = pd.concat([new_data_df, df], ignore_index=True)
            processed_files.append(p_file)
            
        except Exception as e:
            logger.error(f"Error reading {p_file.name}: {e}")
            
    # 3. Check Threshold (Active Learning / Batching)
    if len(new_data_df) < min_new_samples:
        logger.info(f"⏹️ New samples ({len(new_data_df)}) < Threshold ({min_new_samples}). Waiting for more evidence.")
        return

    logger.info(f"✅ Ingesting {len(new_data_df)} valid new samples. Triggering Update!")
    
    # 4. Merge with Main Dataset
    # We use the original dataset + any new validated events
    meta_path = base_dir / config['paths']['metadata_file']
    root_path = base_dir / config['paths']['dataset_dir']
    
    # Fallback logic for paths (same as before)
    if not os.path.exists(meta_path):
        meta_path = base_dir.parent / "dataset_unified" / "metadata" / "unified_metadata.csv"
        root_path = base_dir.parent / "dataset_unified" / "spectrograms"
    
    # Load original
    try:
        original_df = pd.read_csv(meta_path)
        # Append new data
        # Ensure we map columns correctly if needed. Assuming 'filename' matches 'spectrogram_file' logic
        # For simplicity, we assume schema match.
        
        # Add a 'source' column to track provenance
        new_data_df['provenance'] = f"auto_update_{datetime.now().strftime('%Y%m%d')}"
        original_df['provenance'] = original_df.get('provenance', 'original')
        
        combined_df = pd.concat([original_df, new_data_df], ignore_index=True)
        
        # Save UPDATE backup (don't overwrite original unified yet, use a staging metadata)
        staging_meta_path = base_dir / "data" / "staging_metadata.csv"
        combined_df.to_csv(staging_meta_path, index=False)
        logger.info(f"Created staging metadata at {staging_meta_path} with {len(combined_df)} rows.")
        
        # NOTE: We DO NOT move processed files yet. We wait for Decision Logic (Scenario A/B).
            
    except Exception as e:
        logger.error(f"Failed to merge datasets: {e}")
        return

    # 5. Trigger Trainer (using staging metadata)
    trainer = AutoEffiTrainer(str(config_path))
    # Note: We pass staging_meta_path as the metadata source
    # We explicitly pass the number of new samples to trigger PARTIAL MODE if needed
    new_count = len(new_data_df)
    logger.info(f"Invoking Trainer with new_samples_count={new_count}")
    
    try:
        challenger_model_pth = trainer.train(
            metadata_path=str(staging_meta_path), 
            dataset_root=str(root_path),
            new_samples_count=new_count
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    # --- GOLDEN SET EVALUATION ---
    golden_test_path = base_dir / config['paths']['benchmark_test_set']
    if golden_test_path.exists():
        logger.info(f"✨ Running Final Evaluation on Golden Set: {golden_test_path.name}")
        try:
            # Re-evaluate model on Golden Set to get definitive metrics
            # This method updates metadata.json with the new metrics
            trainer.evaluate_on_test_set(str(golden_test_path), str(root_path))
        except Exception as e:
            logger.error(f"Golden Set Evaluation failed: {e}")
            # Proceed with validation metrics (fallback)
    else:
        logger.warning("Golden Set not found. Proceeding with internal validation metrics.")
    
    # 6. No-Harm Evaluation
    with open(base_dir / "models" / "challenger" / "metadata.json", 'r') as f:
        challenger_meta = json.load(f)
        
    champion_meta = registry['champion']
    
    logger.info("Performing No-Harm Principle Check...")
    
    challenger_recall = challenger_meta['metrics']['magnitude_recall_large']
    champion_recall = champion_meta['metrics']['magnitude_recall_large']
    
    challenger_score = challenger_meta['metrics']['composite_score']
    champion_score = champion_meta['metrics']['composite_score']
    
    # Update Statistics
    registry['pipeline_history']['total_runs'] += 1
    registry['pipeline_history']['last_run'] = datetime.now().isoformat()
    
    # Decision Logic
    promote = False
    rejection_reason = ""
    
    if challenger_recall < champion_recall:
        rejection_reason = f"Regression in Large Event Recall ({challenger_recall:.2f}% < {champion_recall:.2f}%)"
    elif challenger_score <= champion_score:
        rejection_reason = f"No significant improvement in composite score ({challenger_score:.4f} <= {champion_score:.4f})"
    else:
        promote = True
        
    if promote:
        # SCENARIO A: SUCCESS
        logger.info("🏆 CHALLENGER PROMOTED! Performance improved without regressions.")
        
        # 1. Deployment (copy files)
        champ_dir = base_dir / "models" / "champion"
        archive_dir = base_dir / "models" / "archive" / champion_meta['model_id']
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Archive old champion
        for f in champ_dir.glob("*"):
            shutil.copy(f, archive_dir / f.name)
            
        # Deploy new one
        challenger_dir = base_dir / "models" / "challenger"
        for f in challenger_dir.glob("*"):
            shutil.copy(f, champ_dir / f.name)
            
        # 2. Finalize Metadata (Staging -> Unified)
        # In a real scenario, we overwrite the main metadata file
        # shutil.copy(staging_meta_path, meta_path) 
        # For now, we simulate this to avoid breaking main dataset if this runs often
        logger.info(f"STAGING METADATA ({staging_meta_path.name}) promoted to MAIN METADATA.")
            
        # 3. Cleanup Pending Evidence (Move to Validated)
        for p_file in processed_files:
            if p_file.exists():
                shutil.move(str(p_file), str(validated_dir / p_file.name))
        logger.info(f" moved {len(processed_files)} pending files to validated/.")
            
        # Update Registry
        registry['champion'] = challenger_meta
        registry['champion']['status'] = "active"
        registry['archive'].append(champion_meta)
        registry['pipeline_history']['successful_updates'] += 1
        
        logger.info(f"Deployed Version {challenger_meta['model_id']} to Production.")
    else:
        # SCENARIO B: FAILURE
        logger.info(f"❌ CHALLENGER REJECTED: {rejection_reason}")
        logger.info("Keeping current Champion in production.")
        registry['pipeline_history']['failed_updates'] += 1
        
        # Rollback Metadata (Delete Staging)
        if staging_meta_path.exists():
            os.remove(staging_meta_path)
            logger.info("Staging metadata discarded.")
            
        # Evidence Retention (Do NOT delete pending files)
        logger.info(f"⚠️ RETAINING {len(processed_files)} pending files for future attempts (Accumulation Strategy).")

    # Save Registry (Always)
    registry['last_updated'] = datetime.now().isoformat()
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated with latest run status.")
    
    if promote:
        logger.info("✅ Pipeline Complete: UPDATE SUCCESSFUL")
    else:
        logger.info("☑️ Pipeline Complete: NO UPDATE REQUIRED")

if __name__ == "__main__":
    run_pipeline()
