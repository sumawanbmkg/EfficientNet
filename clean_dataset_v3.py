#!/usr/bin/env python3
"""
Dataset Cleaning Script for EarthquakeCNN V3.0
Cleans and validates the unified dataset before training

Author: Earthquake Prediction Research Team
Date: 30 January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def clean_unified_dataset(dataset_dir='dataset_unified'):
    """
    Clean the unified dataset by removing invalid entries
    
    Args:
        dataset_dir: Path to dataset directory
    """
    print("🧹 Cleaning Unified Dataset...")
    
    # Load metadata
    metadata_path = Path(dataset_dir) / 'metadata' / 'unified_metadata.csv'
    
    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        return
    
    print(f"📊 Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    print(f"Original dataset size: {len(df)}")
    
    # Check for missing values
    print(f"\n🔍 Checking for missing values...")
    missing_counts = df.isnull().sum()
    critical_columns = ['magnitude_class', 'azimuth_class', 'unified_path']
    
    for col in critical_columns:
        if col in missing_counts and missing_counts[col] > 0:
            print(f"  {col}: {missing_counts[col]} missing values")
    
    # Clean data
    print(f"\n🧹 Cleaning data...")
    
    # Remove rows with NaN in critical columns
    df_clean = df.dropna(subset=critical_columns)
    print(f"After removing NaN values: {len(df_clean)}")
    
    # Convert to string to ensure consistent data types
    df_clean['magnitude_class'] = df_clean['magnitude_class'].astype(str)
    df_clean['azimuth_class'] = df_clean['azimuth_class'].astype(str)
    
    # Remove any rows with 'nan' string values
    df_clean = df_clean[
        (df_clean['magnitude_class'] != 'nan') & 
        (df_clean['azimuth_class'] != 'nan')
    ]
    print(f"After removing 'nan' strings: {len(df_clean)}")
    
    # Check if image files exist
    print(f"\n📁 Checking image file existence...")
    dataset_path = Path(dataset_dir)
    missing_files = []
    
    for idx, row in df_clean.iterrows():
        image_path = dataset_path / row['unified_path']
        if not image_path.exists():
            missing_files.append(idx)
    
    if missing_files:
        print(f"Found {len(missing_files)} missing image files")
        df_clean = df_clean.drop(missing_files)
        print(f"After removing missing files: {len(df_clean)}")
    else:
        print("✅ All image files exist")
    
    # Analyze class distribution
    print(f"\n📈 Class Distribution Analysis:")
    
    magnitude_counts = df_clean['magnitude_class'].value_counts()
    azimuth_counts = df_clean['azimuth_class'].value_counts()
    
    print(f"\nMagnitude Classes:")
    for cls, count in magnitude_counts.items():
        print(f"  {cls}: {count} samples")
    
    print(f"\nAzimuth Classes:")
    for cls, count in azimuth_counts.items():
        print(f"  {cls}: {count} samples")
    
    # Check for extreme imbalance
    mag_ratio = magnitude_counts.max() / magnitude_counts.min()
    az_ratio = azimuth_counts.max() / azimuth_counts.min()
    
    print(f"\nClass Imbalance Ratios:")
    print(f"  Magnitude: {mag_ratio:.1f}:1")
    print(f"  Azimuth: {az_ratio:.1f}:1")
    
    if az_ratio > 50:
        print(f"⚠️ Extreme azimuth imbalance detected! Consider using Focal Loss.")
    
    # Save cleaned dataset
    if len(df_clean) < len(df):
        backup_path = metadata_path.with_suffix('.backup.csv')
        print(f"\n💾 Backing up original to: {backup_path}")
        df.to_csv(backup_path, index=False)
        
        print(f"💾 Saving cleaned dataset to: {metadata_path}")
        df_clean.to_csv(metadata_path, index=False)
        
        print(f"✅ Dataset cleaned: {len(df)} -> {len(df_clean)} samples")
    else:
        print(f"✅ Dataset is already clean: {len(df_clean)} samples")
    
    return df_clean

def validate_dataset_structure(dataset_dir='dataset_unified'):
    """
    Validate dataset directory structure
    
    Args:
        dataset_dir: Path to dataset directory
    """
    print(f"\n🔍 Validating Dataset Structure...")
    
    dataset_path = Path(dataset_dir)
    
    # Check required directories
    required_dirs = [
        'spectrograms',
        'spectrograms/original',
        'metadata'
    ]
    
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - MISSING")
    
    # Check required files
    required_files = [
        'metadata/unified_metadata.csv'
    ]
    
    for file_name in required_files:
        file_path = dataset_path / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - MISSING")
    
    # Count image files
    spectrograms_dir = dataset_path / 'spectrograms'
    if spectrograms_dir.exists():
        png_files = list(spectrograms_dir.rglob('*.png'))
        print(f"📊 Found {len(png_files)} PNG files")
    
    return True

def create_training_splits_info(df_clean):
    """
    Create information about training splits
    
    Args:
        df_clean: Cleaned dataframe
    """
    print(f"\n📊 Training Split Information:")
    
    from sklearn.model_selection import train_test_split
    
    # Stratified split based on azimuth (more imbalanced)
    try:
        train_val_data, test_data = train_test_split(
            df_clean,
            test_size=0.15,
            stratify=df_clean['azimuth_class'],
            random_state=42
        )
        
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=0.2,  # 0.2 of 0.85 = 0.17 of total
            stratify=train_val_data['azimuth_class'],
            random_state=42
        )
        
        print(f"  Train: {len(train_data)} samples ({len(train_data)/len(df_clean)*100:.1f}%)")
        print(f"  Val:   {len(val_data)} samples ({len(val_data)/len(df_clean)*100:.1f}%)")
        print(f"  Test:  {len(test_data)} samples ({len(test_data)/len(df_clean)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Cannot create stratified splits: {e}")
        print(f"Will use random splits during training")
        return False

def main():
    """Main cleaning function"""
    print("🚀 EarthquakeCNN V3.0 - Dataset Cleaning")
    print("=" * 50)
    
    # Validate structure
    validate_dataset_structure()
    
    # Clean dataset
    df_clean = clean_unified_dataset()
    
    if df_clean is not None:
        # Create split info
        create_training_splits_info(df_clean)
        
        print(f"\n✅ Dataset cleaning completed!")
        print(f"📊 Final dataset: {len(df_clean)} samples")
        print(f"🚀 Ready for training with EarthquakeCNN V3.0")
    else:
        print(f"\n❌ Dataset cleaning failed!")
        print(f"Please check dataset structure and try again")

if __name__ == '__main__':
    main()