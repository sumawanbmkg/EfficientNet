"""
Script untuk merge spectrograms baru dari dataset_missing dan dataset_nowrec ke dataset_unified
"""
import os
import shutil
import pandas as pd
from pathlib import Path

def get_magnitude_category(magnitude):
    """Convert magnitude to category folder name"""
    if magnitude is None or pd.isna(magnitude):
        return "Medium"  # Default
    mag = float(magnitude)
    if mag < 5.0:
        return "Small"
    elif mag < 5.5:
        return "Moderate"
    elif mag < 6.0:
        return "Medium"
    elif mag < 7.0:
        return "Large"
    else:
        return "Major"

def merge_datasets():
    """Merge spectrograms dari dataset_missing dan dataset_nowrec ke dataset_unified"""
    
    unified_dir = Path("dataset_unified/spectrograms")
    unified_by_mag = unified_dir / "by_magnitude"
    unified_original = unified_dir / "original"
    
    missing_dir = Path("dataset_missing/spectrograms")
    nowrec_dir = Path("dataset_nowrec/spectrograms")
    
    # Track statistics
    stats = {
        'missing_copied': 0,
        'nowrec_copied': 0,
        'skipped_existing': 0,
        'errors': []
    }
    
    # Source magnitude folders (from new datasets)
    src_mag_folders = ['mag_5.0-5.9', 'mag_6.0-6.9', 'mag_7.0+']
    
    # Target magnitude categories (in unified)
    target_categories = ['Small', 'Moderate', 'Medium', 'Large', 'Major']
    
    # Ensure all target folders exist
    for cat in target_categories:
        (unified_by_mag / cat).mkdir(parents=True, exist_ok=True)
    unified_original.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MERGE DATASETS KE DATASET_UNIFIED")
    print("=" * 60)
    
    def copy_spectrograms(src_dir, source_name):
        """Copy spectrograms from source to unified"""
        copied = 0
        skipped = 0
        
        if not src_dir.exists():
            return copied, skipped
            
        for mag_folder in src_mag_folders:
            src_folder = src_dir / mag_folder
            if not src_folder.exists():
                continue
            
            # Map source folder to target category
            if mag_folder == 'mag_5.0-5.9':
                target_cat = 'Medium'  # 5.0-5.9 -> Medium/Moderate
            elif mag_folder == 'mag_6.0-6.9':
                target_cat = 'Large'
            else:  # mag_7.0+
                target_cat = 'Major'
            
            # Search recursively for PNG files (handles azi_* subfolders)
            for png_file in src_folder.rglob("*.png"):
                filename = png_file.name
                
                # Copy to by_magnitude
                dst_by_mag = unified_by_mag / target_cat / filename
                # Copy to original
                dst_original = unified_original / filename
                
                # Check if already exists in either location
                exists_in_by_mag = dst_by_mag.exists()
                exists_in_original = dst_original.exists()
                
                if not exists_in_by_mag and not exists_in_original:
                    try:
                        # Copy to by_magnitude
                        shutil.copy2(png_file, dst_by_mag)
                        # Also copy to original
                        shutil.copy2(png_file, dst_original)
                        copied += 1
                        print(f"  ✓ {filename} -> {target_cat}")
                    except Exception as e:
                        stats['errors'].append(f"Error copying {png_file}: {e}")
                else:
                    skipped += 1
        
        return copied, skipped
    
    # 1. Copy from dataset_missing
    print(f"\n[1] Copying from dataset_missing...")
    copied, skipped = copy_spectrograms(missing_dir, "missing")
    stats['missing_copied'] = copied
    stats['skipped_existing'] += skipped
    print(f"    Copied: {copied}, Skipped: {skipped}")
    
    # 2. Copy from dataset_nowrec
    print(f"\n[2] Copying from dataset_nowrec...")
    copied, skipped = copy_spectrograms(nowrec_dir, "nowrec")
    stats['nowrec_copied'] = copied
    stats['skipped_existing'] += skipped
    print(f"    Copied: {copied}, Skipped: {skipped}")
    
    # 3. Merge metadata
    print("\n[3] Merging metadata...")
    unified_meta = Path("dataset_unified/metadata")
    
    # Load existing unified metadata
    unified_meta_file = unified_meta / "unified_metadata.csv"
    if unified_meta_file.exists():
        existing_df = pd.read_csv(unified_meta_file)
        print(f"  Existing unified_metadata: {len(existing_df)} records")
    else:
        existing_df = pd.DataFrame()
    
    # Load new metadata and convert to unified format
    new_records = []
    
    missing_meta = Path("dataset_missing/metadata/processed_events.csv")
    if missing_meta.exists():
        missing_df = pd.read_csv(missing_meta)
        print(f"  Missing metadata: {len(missing_df)} records")
        new_records.append(missing_df)
    
    nowrec_meta = Path("dataset_nowrec/metadata/processed_events.csv")
    if nowrec_meta.exists():
        nowrec_df = pd.read_csv(nowrec_meta)
        print(f"  Nowrec metadata: {len(nowrec_df)} records")
        new_records.append(nowrec_df)
    
    if new_records:
        new_df = pd.concat(new_records, ignore_index=True)
        
        # Merge with existing
        if not existing_df.empty:
            # Find common columns
            common_cols = list(set(existing_df.columns) & set(new_df.columns))
            if common_cols:
                merged_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Remove duplicates
                if 'spectrogram_file' in merged_df.columns:
                    merged_df = merged_df.drop_duplicates(subset=['spectrogram_file'], keep='last')
                merged_df.to_csv(unified_meta_file, index=False)
                print(f"  Updated unified_metadata: {len(merged_df)} records")
    
    # 4. Count final spectrograms
    print("\n[4] Final count in dataset_unified...")
    
    # Count by_magnitude
    print("  by_magnitude/:")
    total_by_mag = 0
    for cat in target_categories:
        folder = unified_by_mag / cat
        if folder.exists():
            count = len(list(folder.glob("*.png")))
            total_by_mag += count
            print(f"    {cat}: {count} files")
    
    # Count original
    original_count = len(list(unified_original.glob("*.png"))) if unified_original.exists() else 0
    print(f"  original/: {original_count} files")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Copied from dataset_missing: {stats['missing_copied']}")
    print(f"Copied from dataset_nowrec:  {stats['nowrec_copied']}")
    print(f"Skipped (already exists):    {stats['skipped_existing']}")
    print(f"Total in by_magnitude:       {total_by_mag}")
    print(f"Total in original:           {original_count}")
    
    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for err in stats['errors']:
            print(f"  - {err}")
    
    return stats

if __name__ == "__main__":
    merge_datasets()
