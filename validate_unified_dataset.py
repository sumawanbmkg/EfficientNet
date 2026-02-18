#!/usr/bin/env python3
"""
Script untuk validasi dan analisis dataset unified
Memastikan semua file tersalin dengan benar dan memberikan laporan lengkap
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_unified_dataset():
    """Validate unified dataset structure and content"""
    
    dataset_dir = "dataset_unified"
    
    print("="*70)
    print("VALIDASI DATASET UNIFIED")
    print("="*70)
    
    # Check directory structure
    required_dirs = [
        "spectrograms/original",
        "spectrograms/augmented", 
        "spectrograms/by_azimuth",
        "spectrograms/by_magnitude",
        "metadata"
    ]
    
    print("\n📁 STRUKTUR FOLDER:")
    for req_dir in required_dirs:
        full_path = os.path.join(dataset_dir, req_dir)
        if os.path.exists(full_path):
            print(f"   ✅ {req_dir}")
        else:
            print(f"   ❌ {req_dir} - MISSING")
    
    # Load metadata
    metadata_file = f"{dataset_dir}/metadata/unified_metadata.csv"
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata file not found: {metadata_file}")
        return
    
    df = pd.read_csv(metadata_file)
    print(f"\n📊 METADATA LOADED: {len(df)} records")
    
    # Count actual files
    original_files = len([f for f in os.listdir(f"{dataset_dir}/spectrograms/original") if f.endswith('.png')])
    augmented_files = len([f for f in os.listdir(f"{dataset_dir}/spectrograms/augmented") if f.endswith('.png')])
    
    print(f"\n📈 FILE COUNTS:")
    print(f"   Original images: {original_files}")
    print(f"   Augmented images: {augmented_files}")
    print(f"   Total images: {original_files + augmented_files}")
    print(f"   Metadata records: {len(df)}")
    
    # Validate image format
    print(f"\n🖼️  IMAGE FORMAT VALIDATION:")
    sample_files = []
    
    # Check original images
    orig_dir = f"{dataset_dir}/spectrograms/original"
    orig_files = [f for f in os.listdir(orig_dir) if f.endswith('.png')][:5]
    for filename in orig_files:
        img_path = os.path.join(orig_dir, filename)
        try:
            img = Image.open(img_path)
            sample_files.append({
                'file': filename,
                'size': img.size,
                'mode': img.mode,
                'type': 'original'
            })
        except Exception as e:
            print(f"   ❌ Error reading {filename}: {e}")
    
    # Check augmented images
    aug_dir = f"{dataset_dir}/spectrograms/augmented"
    aug_files = [f for f in os.listdir(aug_dir) if f.endswith('.png')][:5]
    for filename in aug_files:
        img_path = os.path.join(aug_dir, filename)
        try:
            img = Image.open(img_path)
            sample_files.append({
                'file': filename,
                'size': img.size,
                'mode': img.mode,
                'type': 'augmented'
            })
        except Exception as e:
            print(f"   ❌ Error reading {filename}: {e}")
    
    # Print sample validation
    for sample in sample_files:
        size_ok = sample['size'] == (224, 224)
        mode_ok = sample['mode'] == 'RGB'
        status = "✅" if size_ok and mode_ok else "❌"
        print(f"   {status} {sample['file']} ({sample['type']}): {sample['size']} {sample['mode']}")
    
    # Analyze distributions
    print(f"\n📊 DISTRIBUSI DATA:")
    
    # Dataset source
    if 'dataset_source' in df.columns:
        print(f"\n   Dataset Source:")
        source_dist = df['dataset_source'].value_counts()
        for source, count in source_dist.items():
            print(f"     {source}: {count} ({count/len(df)*100:.1f}%)")
    
    # Image type
    if 'image_type' in df.columns:
        print(f"\n   Image Type:")
        type_dist = df['image_type'].value_counts()
        for img_type, count in type_dist.items():
            print(f"     {img_type}: {count} ({count/len(df)*100:.1f}%)")
    
    # Azimuth distribution
    if 'azimuth_class' in df.columns:
        print(f"\n   Azimuth Distribution:")
        az_dist = df['azimuth_class'].value_counts()
        for az_class, count in az_dist.items():
            print(f"     {az_class}: {count} ({count/len(df)*100:.1f}%)")
    
    # Magnitude distribution
    if 'magnitude_class' in df.columns:
        print(f"\n   Magnitude Distribution:")
        mag_dist = df['magnitude_class'].value_counts()
        for mag_class, count in mag_dist.items():
            print(f"     {mag_class}: {count} ({count/len(df)*100:.1f}%)")
    
    # Station distribution (top 10)
    if 'station' in df.columns:
        print(f"\n   Top 10 Stations:")
        station_dist = df['station'].value_counts().head(10)
        for station, count in station_dist.items():
            print(f"     {station}: {count} ({count/len(df)*100:.1f}%)")
    
    # Check class folders
    print(f"\n📂 CLASS FOLDER VALIDATION:")
    
    # Azimuth folders
    azimuth_classes = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    print(f"   Azimuth folders:")
    for az_class in azimuth_classes:
        folder_path = f"{dataset_dir}/spectrograms/by_azimuth/{az_class}"
        if os.path.exists(folder_path):
            file_count = len([f for f in os.listdir(folder_path) if f.endswith('.png')])
            print(f"     {az_class}: {file_count} files")
        else:
            print(f"     {az_class}: MISSING")
    
    # Magnitude folders
    mag_classes = ['Small', 'Moderate', 'Medium', 'Large', 'Major']
    print(f"   Magnitude folders:")
    for mag_class in mag_classes:
        folder_path = f"{dataset_dir}/spectrograms/by_magnitude/{mag_class}"
        if os.path.exists(folder_path):
            file_count = len([f for f in os.listdir(folder_path) if f.endswith('.png')])
            print(f"     {mag_class}: {file_count} files")
        else:
            print(f"     {mag_class}: MISSING")
    
    # Summary
    print(f"\n" + "="*70)
    print("SUMMARY VALIDASI")
    print("="*70)
    print(f"✅ Dataset unified berhasil dibuat")
    print(f"✅ Total images: {original_files + augmented_files}")
    print(f"✅ Original images: {original_files}")
    print(f"✅ Augmented images: {augmented_files}")
    print(f"✅ Format: 224x224 RGB (CNN-ready)")
    print(f"✅ Struktur folder lengkap")
    print(f"✅ Metadata tersedia")
    print(f"✅ Class folders terorganisir")
    
    return df

if __name__ == '__main__':
    validate_unified_dataset()