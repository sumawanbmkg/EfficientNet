import sys
import os
import pandas as pd
import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from trainer_effi import AutoEffiTrainer

def run_loso():
    METADATA_PATH = r"d:\multi\auto_effi\data\unified_metadata.csv"
    DATASET_ROOT = r"d:\multi\dataset_fix" 
    RESULTS_DIR = r"d:\multi\publication_efficientnet\loso_results"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"📂 Loading Metadata from {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    
    # Filter valid stations (e.g. at least 20 samples to be meaningful)
    # Reducing threshold to ensure we cover enough ground, but skipping very sparse ones
    station_counts = df['station'].value_counts()
    valid_stations = station_counts[station_counts >= 15].index.tolist()
    
    print(f"🔄 Starting LOSO Validation on {len(valid_stations)} stations...")
    print(f"Stations: {valid_stations}")
    
    summary_metrics = []
    
    for i, station in enumerate(tqdm(valid_stations, desc="Processing Stations")):
        print(f"\n\n==================================================")
        print(f"🧪 [{i+1}/{len(valid_stations)}] Validating Hold-Out Station: {station}")
        print(f"==================================================")
        
        # Output directory for this fold
        station_dir = os.path.join(RESULTS_DIR, station)
        
        # Check if already done
        meta_path = os.path.join(station_dir, "metadata.json")
        if os.path.exists(meta_path):
            print(f"⏩ Skipping {station}, results already exist.")
            try:
                with open(meta_path, 'r') as f:
                    res = json.load(f)
                    m = res.get('metrics', {})
                    m['station'] = station
                    summary_metrics.append(m)
            except Exception as e:
                print(f"⚠️ Error reading existing metadata for {station}: {e}")
            continue
            
        # Split Data (Logic: Train on ALL others, Test on Target Station)
        test_df = df[df['station'] == station].copy()
        train_df = df[df['station'] != station].copy()
        
        print(f"📊 Split: Train={len(train_df)} | Test (Station {station})={len(test_df)}")
        
        # Instantiate Trainer with custom output dir
        # Fix path issue: config is in auto_effi/config
        config_path = r"d:\multi\auto_effi\config\pipeline_config.yaml"
        trainer = AutoEffiTrainer(config_path=config_path, output_dir=station_dir)
        
        # Optimization for LOSO: 
        # We reduce epochs slightly to make 20+ folds feasible.
        # Standard config is 50. We'll use 12 (5 Frozen + 7 Unfrozen).
        trainer.training_config['epochs'] = 12
        
        start_time = time.time()
        
        try:
            # Force Full Mode (>50 samples) to enable Phase 2 finetuning
            metrics = trainer.train(
                metadata_path=METADATA_PATH,
                dataset_root=DATASET_ROOT,
                new_samples_count=100, 
                train_df=train_df,
                val_df=test_df
            )
            
            if metrics:
                metrics['station'] = station
                metrics['train_time'] = time.time() - start_time
                summary_metrics.append(metrics)
                
        except Exception as e:
            print(f"❌ Failed on station {station}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save Final Summary
    if summary_metrics:
        summary_df = pd.DataFrame(summary_metrics)
        # Reorder columns
        cols = ['station', 'magnitude_accuracy', 'azimuth_accuracy', 'magnitude_recall_large', 'composite_score']
        # Add other cols if they exist
        cols += [c for c in summary_df.columns if c not in cols]
        try:
            summary_df = summary_df[cols]
        except KeyError:
            pass # Keep original order if cols missing
            
        summary_csv = os.path.join(RESULTS_DIR, "loso_final_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"\n✅ LOSO Validation Complete.")
        print(f"💾 Summary saved to: {summary_csv}")
        print("\nTop 5 Stations by Composite Score:")
        print(summary_df.sort_values('composite_score', ascending=False).head(5))
        
        print("\nOverall Average Performance:")
        print(summary_df.mean(numeric_only=True))
    else:
        print("\n❌ No metrics collected.")

if __name__ == "__main__":
    run_loso()
