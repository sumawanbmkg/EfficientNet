"""
SMOTE Implementation for Spectrogram Image Dataset
Generates synthetic samples for minority classes (Moderate, Large)
using image-based SMOTE technique.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import shutil

# Configuration
DATASET_PATH = 'dataset_unified'
METADATA_FILE = os.path.join(DATASET_PATH, 'metadata', 'unified_metadata.csv')
OUTPUT_DIR = 'dataset_smote'
TARGET_SAMPLES_PER_CLASS = 200  # Target for minority classes

# SMOTE parameters
K_NEIGHBORS = 5  # Number of neighbors for SMOTE


def load_image_as_array(img_path):
    """Load image and convert to numpy array"""
    img = Image.open(img_path).convert('RGB')
    return np.array(img, dtype=np.float32)


def save_array_as_image(arr, save_path):
    """Save numpy array as image"""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(save_path)


def find_k_neighbors(target_idx, same_class_indices, features, k=5):
    """Find k nearest neighbors based on image features"""
    target_feat = features[target_idx]
    distances = []
    
    for idx in same_class_indices:
        if idx != target_idx:
            dist = np.linalg.norm(target_feat - features[idx])
            distances.append((idx, dist))
    
    distances.sort(key=lambda x: x[1])
    return [idx for idx, _ in distances[:k]]


def generate_synthetic_image(img1, img2, alpha=None):
    """Generate synthetic image by interpolating between two images"""
    if alpha is None:
        alpha = np.random.uniform(0.3, 0.7)
    
    synthetic = img1 * alpha + img2 * (1 - alpha)
    return synthetic


def extract_simple_features(img_array):
    """Extract simple features for neighbor finding"""
    # Use downsampled mean values as features
    h, w, c = img_array.shape
    block_h, block_w = h // 8, w // 8
    features = []
    
    for i in range(8):
        for j in range(8):
            block = img_array[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            features.extend([block.mean(), block.std()])
    
    return np.array(features)


def apply_smote_to_class(class_data, class_name, dataset_path, output_dir, 
                         target_count, k_neighbors=5):
    """Apply SMOTE to generate synthetic samples for a class"""
    
    current_count = len(class_data)
    samples_needed = target_count - current_count
    
    if samples_needed <= 0:
        print(f"  {class_name}: Already has {current_count} samples, no SMOTE needed")
        return []
    
    print(f"  {class_name}: {current_count} -> {target_count} (generating {samples_needed} synthetic)")
    
    # Load all images and extract features
    images = []
    features = []
    
    print(f"    Loading images...")
    for _, row in tqdm(class_data.iterrows(), total=len(class_data), desc="    Loading"):
        img_path = os.path.join(dataset_path, row['unified_path'])
        if os.path.exists(img_path):
            img_arr = load_image_as_array(img_path)
            images.append(img_arr)
            features.append(extract_simple_features(img_arr))
    
    if len(images) < 2:
        print(f"    Not enough images for SMOTE")
        return []
    
    features = np.array(features)
    indices = list(range(len(images)))
    
    # Generate synthetic samples
    synthetic_samples = []
    print(f"    Generating synthetic samples...")
    
    for i in tqdm(range(samples_needed), desc="    SMOTE"):
        # Randomly select a sample
        idx = np.random.choice(indices)
        
        # Find k neighbors
        neighbors = find_k_neighbors(idx, indices, features, k=min(k_neighbors, len(indices)-1))
        
        if not neighbors:
            continue
        
        # Randomly select one neighbor
        neighbor_idx = np.random.choice(neighbors)
        
        # Generate synthetic image
        alpha = np.random.uniform(0.3, 0.7)
        synthetic_img = generate_synthetic_image(images[idx], images[neighbor_idx], alpha)
        
        # Create metadata for synthetic sample
        base_row = class_data.iloc[idx % len(class_data)].copy()
        synthetic_samples.append({
            'image': synthetic_img,
            'base_row': base_row,
            'alpha': alpha,
            'source_idx': idx,
            'neighbor_idx': neighbor_idx
        })
    
    return synthetic_samples


def main():
    print("=" * 70)
    print("SMOTE DATASET GENERATION")
    print("=" * 70)
    print(f"Target samples per minority class: {TARGET_SAMPLES_PER_CLASS}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'spectrograms'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'spectrograms', 'original'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'spectrograms', 'smote'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'metadata'), exist_ok=True)
    
    # Load metadata
    print("\nLoading metadata...")
    train_data = pd.read_csv(os.path.join(DATASET_PATH, 'metadata', 'train_split.csv'))
    val_data = pd.read_csv(os.path.join(DATASET_PATH, 'metadata', 'val_split.csv'))
    
    # Filter precursor samples
    valid_mag_classes = ['Moderate', 'Medium', 'Large']
    train_precursor = train_data[train_data['magnitude_class'].isin(valid_mag_classes)].copy()
    
    print(f"\nOriginal class distribution:")
    mag_dist = Counter(train_precursor['magnitude_class'])
    for cls, count in sorted(mag_dist.items()):
        print(f"  {cls}: {count}")
    
    # Apply SMOTE to minority classes
    print("\n" + "=" * 70)
    print("APPLYING SMOTE TO MINORITY CLASSES")
    print("=" * 70)
    
    all_synthetic = []
    
    for mag_class in ['Moderate', 'Large']:
        class_data = train_precursor[train_precursor['magnitude_class'] == mag_class]
        synthetic = apply_smote_to_class(
            class_data, mag_class, DATASET_PATH, OUTPUT_DIR,
            TARGET_SAMPLES_PER_CLASS, K_NEIGHBORS
        )
        all_synthetic.extend([(s, mag_class) for s in synthetic])
    
    # Save synthetic images and create metadata
    print("\n" + "=" * 70)
    print("SAVING SYNTHETIC SAMPLES")
    print("=" * 70)
    
    synthetic_metadata = []
    
    for i, (sample, mag_class) in enumerate(tqdm(all_synthetic, desc="Saving")):
        # Generate filename
        filename = f"SMOTE_{mag_class}_{i:04d}.png"
        save_path = os.path.join(OUTPUT_DIR, 'spectrograms', 'smote', filename)
        
        # Save image
        save_array_as_image(sample['image'], save_path)
        
        # Create metadata entry
        base_row = sample['base_row']
        new_row = {
            'station': base_row['station'],
            'date': base_row['date'],
            'hour': base_row.get('hour', 0),
            'azimuth': base_row.get('azimuth', 0),
            'magnitude': base_row.get('magnitude', 0),
            'azimuth_class': base_row['azimuth_class'],
            'magnitude_class': mag_class,
            'unified_path': f'spectrograms/smote/{filename}',
            'augmentation_type': 'smote',
            'augmentation_id': i,
            'source_alpha': sample['alpha'],
            'is_synthetic': True
        }
        synthetic_metadata.append(new_row)
    
    # Copy original images
    print("\nCopying original images...")
    original_metadata = []
    
    for _, row in tqdm(train_precursor.iterrows(), total=len(train_precursor), desc="Copying"):
        src_path = os.path.join(DATASET_PATH, row['unified_path'])
        if os.path.exists(src_path):
            filename = os.path.basename(row['unified_path'])
            dst_path = os.path.join(OUTPUT_DIR, 'spectrograms', 'original', filename)
            shutil.copy2(src_path, dst_path)
            
            new_row = row.to_dict()
            new_row['unified_path'] = f'spectrograms/original/{filename}'
            new_row['is_synthetic'] = False
            original_metadata.append(new_row)
    
    # Combine metadata
    all_metadata = original_metadata + synthetic_metadata
    combined_df = pd.DataFrame(all_metadata)
    
    # Save metadata
    combined_df.to_csv(os.path.join(OUTPUT_DIR, 'metadata', 'smote_train.csv'), index=False)
    
    # Copy validation data
    print("\nCopying validation data...")
    val_precursor = val_data[val_data['magnitude_class'].isin(valid_mag_classes)].copy()
    val_metadata = []
    
    for _, row in tqdm(val_precursor.iterrows(), total=len(val_precursor), desc="Copying val"):
        src_path = os.path.join(DATASET_PATH, row['unified_path'])
        if os.path.exists(src_path):
            filename = os.path.basename(row['unified_path'])
            dst_path = os.path.join(OUTPUT_DIR, 'spectrograms', 'original', filename)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            
            new_row = row.to_dict()
            new_row['unified_path'] = f'spectrograms/original/{filename}'
            new_row['is_synthetic'] = False
            val_metadata.append(new_row)
    
    val_df = pd.DataFrame(val_metadata)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'metadata', 'smote_val.csv'), index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SMOTE DATASET GENERATION COMPLETE")
    print("=" * 70)
    
    print(f"\nNew class distribution (train):")
    new_dist = Counter(combined_df['magnitude_class'])
    for cls, count in sorted(new_dist.items()):
        print(f"  {cls}: {count}")
    
    print(f"\nTotal train samples: {len(combined_df)}")
    print(f"  - Original: {len(original_metadata)}")
    print(f"  - Synthetic (SMOTE): {len(synthetic_metadata)}")
    print(f"\nValidation samples: {len(val_df)}")
    
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'original_distribution': dict(mag_dist),
        'new_distribution': dict(new_dist),
        'total_train': len(combined_df),
        'original_samples': len(original_metadata),
        'synthetic_samples': len(synthetic_metadata),
        'validation_samples': len(val_df),
        'target_per_class': TARGET_SAMPLES_PER_CLASS,
        'k_neighbors': K_NEIGHBORS
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metadata', 'smote_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n[OK] SMOTE dataset ready for training!")


if __name__ == '__main__':
    main()
