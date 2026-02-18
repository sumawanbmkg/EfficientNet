"""
Fix Unified Metadata

Script untuk memperbaiki metadata dengan hanya menyimpan entry yang file spectrogramnya ada.
"""

import os
import pandas as pd
from pathlib import Path

def fix_metadata():
    """Filter metadata untuk hanya file yang ada."""
    
    # Paths
    metadata_path = Path('dataset_unified/metadata/unified_metadata.csv')
    spectrograms_dir = Path('dataset_unified/spectrograms')
    
    print("=" * 60)
    print("FIXING UNIFIED METADATA")
    print("=" * 60)
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    print(f"\nOriginal metadata: {len(df)} rows")
    
    # Get all spectrogram files that exist
    existing_files = set()
    
    # Scan all subdirectories
    for root, dirs, files in os.walk(spectrograms_dir):
        for f in files:
            if f.endswith('.png'):
                existing_files.add(f)
    
    print(f"Found {len(existing_files)} spectrogram files on disk")
    
    # Check which files in metadata exist
    valid_rows = []
    missing_files = []
    
    for idx, row in df.iterrows():
        filename = row.get('spectrogram_file')
        if pd.isna(filename):
            missing_files.append(('NaN filename', idx))
            continue
            
        if filename in existing_files:
            valid_rows.append(idx)
        else:
            missing_files.append((filename, idx))
    
    print(f"\nValid rows (file exists): {len(valid_rows)}")
    print(f"Missing files: {len(missing_files)}")
    
    # Show sample of missing files
    if missing_files:
        print("\nSample missing files (first 10):")
        for fname, idx in missing_files[:10]:
            print(f"  - {fname}")
    
    # Create filtered dataframe
    df_valid = df.loc[valid_rows].copy()
    
    # Ensure filename column is set correctly
    df_valid['filename'] = df_valid['spectrogram_file']
    
    # Check class distribution
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION (after filtering)")
    print("=" * 60)
    
    print("\nMagnitude classes:")
    print(df_valid['magnitude_class'].value_counts())
    
    print("\nAzimuth classes:")
    print(df_valid['azimuth_class'].value_counts())
    
    # Backup original
    backup_path = metadata_path.with_suffix('.csv.backup')
    df.to_csv(backup_path, index=False)
    print(f"\nBackup saved to: {backup_path}")
    
    # Save fixed metadata
    df_valid.to_csv(metadata_path, index=False)
    print(f"Fixed metadata saved to: {metadata_path}")
    print(f"Final row count: {len(df_valid)}")
    
    print("\n" + "=" * 60)
    print("METADATA FIX COMPLETE")
    print("=" * 60)
    
    return df_valid

if __name__ == "__main__":
    fix_metadata()
