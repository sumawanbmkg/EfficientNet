"""
Experiment 3 - Step 2: Finalize Dataset (Split & SMOTE)
======================================================
1. Split consolidated metadata into Train/Val/Test (Zero Data Leakage).
2. Apply SMOTE to the Training set to balance classes.
3. Save final artifacts for training.
"""

import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from sklearn.model_selection import StratifiedGroupKFold

# Configuration
INPUT_FILE = 'dataset_experiment_3/metadata_raw_exp3.csv'
BASE_IMG_DIR = 'dataset_experiment_3'
OUTPUT_DIR = 'dataset_experiment_3/final_metadata'
SMOTE_IMG_DIR = 'dataset_experiment_3/spectrograms' # Unified folder
RANDOM_SEED = 42

# Target samples in TRAINING set after SMOTE
# Experiment 3 targets: balance minority classes
TARGET_TRAIN_SAMPLES = {
    'Large': 600,
    'Medium': 600,
    'Moderate': 600
}

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_image(path):
    try:
        if not os.path.isabs(path):
            path = os.path.join(BASE_IMG_DIR, path)
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float32)
    except:
        return None

def save_image(arr, path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def extract_features_simple(images):
    features = []
    for img in images:
        feats = [
            img[:,:,0].mean(), img[:,:,0].std(),
            img[:,:,1].mean(), img[:,:,1].std(),
            img[:,:,2].mean(), img[:,:,2].std()
        ]
        features.append(feats)
    return np.array(features)

def find_neighbors(features, query_idx, k=5):
    diff = features - features[query_idx]
    dist = np.linalg.norm(diff, axis=1)
    sorted_idx = np.argsort(dist)
    return sorted_idx[1:k+1]

def apply_smote(df_train, class_name, target_count):
    df_class = df_train[df_train['magnitude_class'] == class_name]
    current_count = len(df_class)
    needed = target_count - current_count
    
    if needed <= 0:
        return []

    logger.info(f"Generating {needed} SMOTE samples for {class_name}...")
    
    images = []
    valid_rows = []
    for idx, row in df_class.iterrows():
        img = load_image(row['filepath'])
        if img is not None:
            images.append(img)
            valid_rows.append(row)
            
    if len(images) < 2:
        return []
        
    features = extract_features_simple(images)
    new_rows = []
    
    for i in tqdm(range(needed), desc=f"SMOTE {class_name}"):
        p_idx = np.random.randint(len(images))
        parent_img = images[p_idx]
        
        neighbors = find_neighbors(features, p_idx, k=min(5, len(images)-1))
        n_idx = np.random.choice(neighbors)
        neighbor_img = images[n_idx]
        
        alpha = np.random.uniform(0.3, 0.7)
        # Resize to fixed 224x224 to avoid broadcast errors and match model input
        p_img_resized = np.array(Image.fromarray(parent_img.astype(np.uint8)).resize((224, 224)), dtype=np.float32)
        n_img_resized = np.array(Image.fromarray(neighbor_img.astype(np.uint8)).resize((224, 224)), dtype=np.float32)
        
        synthetic_img = p_img_resized * alpha + n_img_resized * (1 - alpha)
        
        # Save synthetic image
        fname = f"SMOTE_Exp3_{class_name}_{i:04d}.png"
        fpath = os.path.join(SMOTE_IMG_DIR, fname)
        save_image(synthetic_img, fpath)
        
        # Metadata inheritance
        new_row = valid_rows[p_idx].copy()
        new_row['filename'] = fname
        new_row['consolidation_path'] = f"spectrograms/{fname}"
        new_row['is_synthetic'] = 1
        new_row['event_group'] = f"SYNTH_{class_name}_{i}"
        new_rows.append(new_row)
        
    return new_rows

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SMOTE_IMG_DIR, exist_ok=True)
    
    df = pd.read_csv(INPUT_FILE)
    df['event_group'] = df['station'] + '_' + df['date'].astype(str)
    df['is_synthetic'] = 0
    df['is_precursor'] = (df['magnitude_class'] != 'Normal').astype(int)
    # Trainer expects consolidation_path
    df['consolidation_path'] = df['filepath']

    logger.info(f"Initial Distribution:\n{df['magnitude_class'].value_counts()}")

    # 1. Stratified Group Split
    gkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    X = df.index
    y = df['magnitude_class']
    groups = df['event_group']
    
    # Split Test (10%)
    train_val_idx, test_idx = next(gkf.split(X, y, groups))
    df_train_val = df.iloc[train_val_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # Split Val (10% of total ~= 11.1% of remaining)
    gkf_inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=RANDOM_SEED)
    train_idx, val_idx = next(gkf_inner.split(df_train_val.index, df_train_val['magnitude_class'], df_train_val['event_group']))
    
    df_train = df_train_val.iloc[train_idx].copy()
    df_val = df_train_val.iloc[val_idx].copy()
    
    logger.info(f"Split Summary: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # 2. Apply SMOTE to Train
    augmented_train_rows = []
    # Add original training images
    augmented_train_rows.append(df_train)
    
    for cls, target in TARGET_TRAIN_SAMPLES.items():
        new_samples = apply_smote(df_train, cls, target)
        if new_samples:
            augmented_train_rows.append(pd.DataFrame(new_samples))
            
    df_train_final = pd.concat(augmented_train_rows, ignore_index=True)
    
    # 3. Save
    df_train_final.to_csv(os.path.join(OUTPUT_DIR, 'train_exp3.csv'), index=False)
    df_val.to_csv(os.path.join(OUTPUT_DIR, 'val_exp3.csv'), index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, 'test_exp3.csv'), index=False)
    
    logger.info(f"Final Train Distribution:\n{df_train_final['magnitude_class'].value_counts()}")
    logger.info(f"Files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
