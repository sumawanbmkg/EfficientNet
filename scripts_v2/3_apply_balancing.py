"""
Script Phase 2 - Step 3: Apply Advanced Balancing (SMOTE)
Mengimplementasikan augmentasi data sintetis (SMOTE) khusus untuk Training Set.
Hanya diterapkan pada kelas 'Large' dan 'Medium' untuk memperbaiki imbalance.

Logika:
1. Load split_train.csv
2. Identifikasi kelas minoritas (Large)
3. Generate image sintetis (interpolasi spektrogram)
4. Simpan hasil di dataset_smote_train/
5. Update metadata training baru (augmented_train.csv)
"""

import os
import shutil
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter
import logging

# Konfigurasi
INPUT_META = 'dataset_consolidation/metadata/split_train.csv'
OUTPUT_DIR = 'dataset_smote_train'
INPUT_IMG_DIR = 'dataset_consolidation'

# Target jumlah sampel per kelas (minimum)
# Jika sampel asli < target, akan digenerate sampai target tercapai
TARGET_SAMPLES = {
    'Large': 500,    # Target utama kita
    'Medium': 500,   # Penyeimbang
    'Moderate': 500  # Penyeimbang
}

# Parameter SMOTE
K_NEIGHBORS = 5

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_image(path):
    try:
        if not os.path.isabs(path):
            path = os.path.join(INPUT_IMG_DIR, path)
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None

def save_image(arr, path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def extract_features_simple(images):
    # Fitur sederhana untuk NN: Mean & Std dari setiap channel
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
    # Euclidean distance simple
    diff = features - features[query_idx]
    dist = np.linalg.norm(diff, axis=1)
    # Sort, skip self (dist=0)
    sorted_idx = np.argsort(dist)
    return sorted_idx[1:k+1]

def generate_smote(df_class, class_name, target_count):
    current_count = len(df_class)
    needed = target_count - current_count
    
    if needed <= 0:
        logger.info(f"  Class {class_name}: {current_count} >= {target_count}, skipping SMOTE.")
        return []

    logger.info(f"  Class {class_name}: Generating {needed} synthetic samples (Base: {current_count})")
    
    # 1. Load All Images in Memory (Warning: Memory intensive if images are huge)
    images = []
    valid_indices = []
    
    for idx, row in df_class.iterrows():
        img = load_image(row['consolidation_path'])
        if img is not None:
            images.append(img)
            valid_indices.append(idx)
            
    if len(images) < 2:
        logger.warning(f"  Not enough samples for SMOTE in {class_name}!")
        return []
        
    features = extract_features_simple(images)
    new_metadata = []
    
    # 2. Generate
    for i in tqdm(range(needed), desc=f"  SMOTE {class_name}"):
        # Pick random parent
        parent_local_idx = np.random.randint(len(images))
        parent_img = images[parent_local_idx]
        
        # Find neighbors
        neighbors = find_neighbors(features, parent_local_idx, k=min(K_NEIGHBORS, len(images)-1))
        neighbor_local_idx = np.random.choice(neighbors)
        neighbor_img = images[neighbor_local_idx]
        
        # Ensure same shape (resize if needed)
        if parent_img.shape != neighbor_img.shape:
            from PIL import Image as PILImage
            target_shape = parent_img.shape[:2]
            neighbor_pil = PILImage.fromarray(neighbor_img.astype(np.uint8))
            neighbor_pil = neighbor_pil.resize((target_shape[1], target_shape[0]))
            neighbor_img = np.array(neighbor_pil, dtype=np.float32)
        
        # Interpolate
        alpha = np.random.uniform(0.3, 0.7)
        try:
            synthetic_img = parent_img * alpha + neighbor_img * (1 - alpha)
        except Exception as e:
            logger.warning(f"  Failed to interpolate: {e}, skipping")
            continue
        
        # Save
        filename = f"SMOTE_{class_name}_{i:04d}.png"
        save_path = os.path.join(OUTPUT_DIR, 'spectrograms', filename)
        save_image(synthetic_img, save_path)
        
        # Metadata
        # Wariskan metadata dari Parent utama, tapi beri tanda sintetis
        parent_row = df_class.loc[valid_indices[parent_local_idx]].copy()
        parent_row['original_relative_path'] = 'SYNTHETIC'
        parent_row['consolidation_path'] = f"spectrograms/{filename}"
        parent_row['is_synthetic'] = 1
        parent_row['smote_alpha'] = alpha
        # ID Event baru agar tidak dianggap duplikat saat training cek
        parent_row['event_group'] = f"SYNTH_{class_name}_{i}" 
        
        new_metadata.append(parent_row)
        
    return new_metadata

def main():
    logger.info("="*50)
    logger.info("PHASE 2 - STEP 3: APPLY SMOTE BALANCING")
    logger.info("="*50)
    
    if not os.path.exists(INPUT_META):
        logger.error(f"Input not found: {INPUT_META}. Run Step 2 first.")
        return

    # Prepare Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'spectrograms'), exist_ok=True)
    
    df = pd.read_csv(INPUT_META)
    
    # Filter class 'Major' digabung ke 'Large' jika belum (harusnya sudah di Step 2)
    # Tapi kita pastikan lagi
    # 'stratify_label' sudah dibuat di step 2
    
    augmented_rows = []
    
    # 1. Copy Data Asli (Bisa copy file fisik atau cuma reference path)
    # Untuk efisiensi, kita referensikan path di consolidation folder, 
    # kecuali jika perlu dipindah semua.
    # Agar aman & isolated, kita buat symlink atau copy.
    # Disini kita hanya update metadata pathnya.
    
    logger.info("Processing Original Data...")
    df['is_synthetic'] = 0
    df['smote_alpha'] = 0.0
    # Add prefix path agar loader nanti tau ini ada di consolidation/
    # Atau kita atur loader agar pintar. 
    # Opsi terbaik: Biarkan path relative, tapi nanti training script harus tau logic foldernya.
    # Untuk simplifikasi SEKARANG: Kita copy semua training image ke folder smote.
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Copying Originals"):
        src = os.path.join(INPUT_IMG_DIR, row['consolidation_path'])
        dst_name = os.path.basename(row['consolidation_path'])
        dst = os.path.join(OUTPUT_DIR, 'spectrograms', dst_name)
        
        if not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
            except:
                continue
                
        row['consolidation_path'] = f"spectrograms/{dst_name}"
        augmented_rows.append(row)
        
    # 2. Generate SMOTE per Class
    unique_classes = df['magnitude_class'].unique()
    
    for cls in unique_classes:
        if cls in TARGET_SAMPLES:
            df_class = df[df['magnitude_class'] == cls]
            new_rows = generate_smote(df_class, cls, TARGET_SAMPLES[cls])
            augmented_rows.extend(new_rows)
            
    # 3. Save Final Augmented Metadata
    final_df = pd.DataFrame(augmented_rows)
    out_path = os.path.join(OUTPUT_DIR, 'augmented_train_metadata.csv')
    final_df.to_csv(out_path, index=False)
    
    logger.info("="*50)
    logger.info("BALANCING COMPLETE")
    logger.info("Original Train Size: {}".format(len(df)))
    logger.info("Augmented Train Size: {}".format(len(final_df)))
    logger.info("\nNew Class Distribution:")
    logger.info(final_df['stratify_label'].value_counts())
    logger.info(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
