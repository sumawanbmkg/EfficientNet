"""
Script Phase 2 - Step 2: Create Stratified Split
Membagi dataset menjadi Training, Validation, dan Test sets dengan prinsip:
1. Zero Data Leakage: Split berdasarkan EVENT ID (Station + Date), bukan random image.
2. Stratified: Memastikan distribusi Magnitude (terutama Large) seimbang di setiap set.
3. Hierarchical Ready: Menambahkan kolom 'is_precursor' untuk training tahap binary.

Output:
- dataset_consolidation/metadata/split_train.csv
- dataset_consolidation/metadata/split_val.csv
- dataset_consolidation/metadata/split_test.csv
"""

import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import StratifiedGroupKFold

# Konfigurasi
INPUT_FILE = 'dataset_consolidation/metadata.csv'
OUTPUT_DIR = 'dataset_consolidation/metadata'
RANDOM_SEED = 42

# Rasio Split (approximate karena GroupKFold)
# Target: Train 70%, Val 15%, Test 15%
N_SPLITS = 6  # 1/6 ~= 16.6% untuk Test/Val

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*50)
    logger.info("PHASE 2 - STEP 2: CREATE STRATIFIED SPLIT")
    logger.info("="*50)
    
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"Loaded {len(df)} samples")
    
    # 2. Preprocessing untuk Split
    # Buat kolom 'event_group' sebagai identifier unik untuk GroupKFold
    # Format: STATION_YYYYMMDD (abaikan jam agar semua jam dalam satu hari masuk grup sama)
    df['date_str'] = df['date'].astype(str)
    df['event_group'] = df['station'] + '_' + df['date_str']
    
    # Stratification Target: Magnitude Base Class
    # Kita ingin memastikan kelas 'Large', 'Medium', 'Moderate', 'Normal' tersebar merata
    # Handle kelas 'Major' -> gabung ke 'Large'
    # Handle kelas 'Small' -> gabung ke 'Normal' (jika ada)
    
    def normalize_mag_class(cls):
        if pd.isna(cls): return 'Normal'
        cls = str(cls).capitalize()
        if cls == 'Major': return 'Large'
        if cls == 'Small': return 'Normal'
        if cls not in ['Large', 'Medium', 'Moderate', 'Normal']: return 'Normal'
        return cls

    df['stratify_label'] = df['magnitude_class'].apply(normalize_mag_class)
    
    # Tambahkan Binary Label (untuk Hierarchical Training)
    df['is_precursor'] = (df['stratify_label'] != 'Normal').astype(int)
    
    logger.info("Stratification Class Distribution:")
    logger.info(df['stratify_label'].value_counts())

    # 3. Splitting Strategy
    # Kita butuh 3 set: Train, Val, Test.
    # GroupKFold tidak support 3-way split langsung dengan stratifikasi sempurna.
    # Trik:
    # Split 1: Train+Val (85%) vs Test (15%)
    # Split 2: Train (70%) vs Val (15%) dari sisa data
    
    # Setup GroupKFold
    gkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=RANDOM_SEED)
    
    # Split 1: Pisahkan Test Set
    X = df.index
    y = df['stratify_label']
    groups = df['event_group']
    
    train_val_idx, test_idx = next(gkf.split(X, y, groups))
    
    df_train_val = df.iloc[train_val_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # Split 2: Pisahkan Train dan Val dari df_train_val
    # Rasio sisa: Val harus ~15% dari total awal, atau ~17.6% dari Train+Val
    # Gunakan 5 split di sisa data (1/5 = 20% dari 85% ~= 17% total)
    gkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    X_inner = df_train_val.index
    y_inner = df_train_val['stratify_label']
    groups_inner = df_train_val['event_group']
    
    train_idx, val_idx = next(gkf_inner.split(X_inner, y_inner, groups_inner))
    
    # Karena idx di sini relative terhadap df_train_val, kita harus mapping balik atau
    # lebih mudah langsung ambil dari dataframe hasil split
    df_train = df_train_val.iloc[train_idx].copy()
    df_val = df_train_val.iloc[val_idx].copy()
    
    # 4. Validasi Hasil Split (Cek Leakage)
    train_groups = set(df_train['event_group'])
    val_groups = set(df_val['event_group'])
    test_groups = set(df_test['event_group'])
    
    leak_tv = train_groups.intersection(val_groups)
    leak_vt = val_groups.intersection(test_groups)
    leak_tt = train_groups.intersection(test_groups)
    
    if leak_tv or leak_vt or leak_tt:
        logger.error("CRITICAL: Data Leakage Detected!")
        logger.error(f"Train-Val Leak: {len(leak_tv)}")
        logger.error(f"Val-Test Leak: {len(leak_vt)}")
        logger.error(f"Train-Test Leak: {len(leak_tt)}")
        return
    else:
        logger.info("Split VALID - No Data Leakage between sets.")
        
    # 5. Save Splits
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_train.to_csv(os.path.join(OUTPUT_DIR, 'split_train.csv'), index=False)
    df_val.to_csv(os.path.join(OUTPUT_DIR, 'split_val.csv'), index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, 'split_test.csv'), index=False)
    
    logger.info("="*50)
    logger.info("SPLIT SUMMARY")
    logger.info(f"Total Samples: {len(df)}")
    logger.info(f"Train Set: {len(df_train)} ({len(df_train)/len(df):.1%}) - {len(train_groups)} Events")
    logger.info(f"Val Set:   {len(df_val)} ({len(df_val)/len(df):.1%}) - {len(val_groups)} Events")
    logger.info(f"Test Set:  {len(df_test)} ({len(df_test)/len(df):.1%}) - {len(test_groups)} Events")
    
    logger.info("\nLarge Class Distribution:")
    tr_l = len(df_train[df_train['stratify_label'] == 'Large'])
    va_l = len(df_val[df_val['stratify_label'] == 'Large'])
    te_l = len(df_test[df_test['stratify_label'] == 'Large'])
    logger.info(f"Train: {tr_l}")
    logger.info(f"Val:   {va_l}")
    logger.info(f"Test:  {te_l}")

if __name__ == "__main__":
    main()
