#!/usr/bin/env python3
"""
Script untuk menggabungkan semua dataset yang ada ke dalam satu folder unified
Menggabungkan dataset dari berbagai sumber:
- dataset_spectrogram_ssh_v21/ (278 events, CNN-ready)
- dataset_augmented/ (1,084 images dengan augmentasi)
- dataset lainnya jika ada

Output: dataset_unified/ dengan struktur yang rapi dan metadata lengkap
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetConsolidator:
    """Consolidate multiple datasets into unified structure"""
    
    def __init__(self, output_dir='dataset_unified'):
        self.output_dir = output_dir
        self.consolidated_metadata = []
        self.stats = {
            'total_images_copied': 0,
            'total_original_images': 0,
            'total_augmented_images': 0,
            'datasets_processed': 0,
            'duplicates_skipped': 0
        }
        
        # Create output structure
        self._create_output_structure()
        
    def _create_output_structure(self):
        """Create unified dataset structure"""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/spectrograms",
            f"{self.output_dir}/spectrograms/original",
            f"{self.output_dir}/spectrograms/augmented", 
            f"{self.output_dir}/spectrograms/by_azimuth",
            f"{self.output_dir}/spectrograms/by_magnitude",
            f"{self.output_dir}/metadata",
            f"{self.output_dir}/logs"
        ]
        
        # Azimuth classes
        azimuth_classes = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        for az_class in azimuth_classes:
            dirs.append(f"{self.output_dir}/spectrograms/by_azimuth/{az_class}")
        
        # Magnitude classes
        mag_classes = ['Small', 'Moderate', 'Medium', 'Large', 'Major']
        for mag_class in mag_classes:
            dirs.append(f"{self.output_dir}/spectrograms/by_magnitude/{mag_class}")
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        logger.info(f"Created unified dataset structure in: {self.output_dir}")
    
    def process_v21_dataset(self):
        """Process dataset_spectrogram_ssh_v21 (original CNN-ready dataset)"""
        source_dir = "dataset_spectrogram_ssh_v21"
        
        if not os.path.exists(source_dir):
            logger.warning(f"Dataset v2.1 not found: {source_dir}")
            return
        
        logger.info("="*60)
        logger.info("PROCESSING DATASET SSH V2.1 (ORIGINAL)")
        logger.info("="*60)
        
        # Load metadata
        metadata_file = f"{source_dir}/metadata/dataset_metadata.csv"
        if os.path.exists(metadata_file):
            df = pd.read_csv(metadata_file)
            logger.info(f"Loaded {len(df)} records from v2.1 metadata")
            
            # Process each record
            for idx, row in df.iterrows():
                # Copy spectrogram image
                source_img = f"{source_dir}/spectrograms/{row['spectrogram_file']}"
                if os.path.exists(source_img):
                    # Copy to original folder
                    dest_img = f"{self.output_dir}/spectrograms/original/{row['spectrogram_file']}"
                    shutil.copy2(source_img, dest_img)
                    
                    # Copy to class folders
                    az_dest = f"{self.output_dir}/spectrograms/by_azimuth/{row['azimuth_class']}/{row['spectrogram_file']}"
                    mag_dest = f"{self.output_dir}/spectrograms/by_magnitude/{row['magnitude_class']}/{row['spectrogram_file']}"
                    shutil.copy2(source_img, az_dest)
                    shutil.copy2(source_img, mag_dest)
                    
                    # Add to consolidated metadata
                    metadata_record = row.to_dict()
                    metadata_record['dataset_source'] = 'v2.1_original'
                    metadata_record['image_type'] = 'original'
                    metadata_record['unified_path'] = f"spectrograms/original/{row['spectrogram_file']}"
                    self.consolidated_metadata.append(metadata_record)
                    
                    self.stats['total_images_copied'] += 1
                    self.stats['total_original_images'] += 1
                    
                    if idx % 50 == 0:
                        logger.info(f"Processed {idx+1}/{len(df)} v2.1 images...")
                else:
                    logger.warning(f"Image not found: {source_img}")
            
            logger.info(f"✅ Processed {len(df)} images from dataset v2.1")
            self.stats['datasets_processed'] += 1
        else:
            logger.error(f"Metadata file not found: {metadata_file}")
    
    def process_augmented_dataset(self):
        """Process dataset_augmented (augmented images)"""
        source_dir = "dataset_augmented"
        
        if not os.path.exists(source_dir):
            logger.warning(f"Augmented dataset not found: {source_dir}")
            return
        
        logger.info("="*60)
        logger.info("PROCESSING AUGMENTED DATASET")
        logger.info("="*60)
        
        # Load metadata if exists
        metadata_file = f"{source_dir}/metadata/dataset_metadata.csv"
        if os.path.exists(metadata_file):
            df = pd.read_csv(metadata_file)
            logger.info(f"Loaded {len(df)} records from augmented metadata")
            
            # Process each record
            for idx, row in df.iterrows():
                source_img = f"{source_dir}/spectrograms/{row['spectrogram_file']}"
                if os.path.exists(source_img):
                    # Determine if original or augmented
                    is_augmented = 'aug' in row['spectrogram_file']
                    
                    if is_augmented:
                        # Copy to augmented folder
                        dest_img = f"{self.output_dir}/spectrograms/augmented/{row['spectrogram_file']}"
                        shutil.copy2(source_img, dest_img)
                        
                        # Add to consolidated metadata
                        metadata_record = row.to_dict()
                        metadata_record['dataset_source'] = 'augmented'
                        metadata_record['image_type'] = 'augmented'
                        metadata_record['unified_path'] = f"spectrograms/augmented/{row['spectrogram_file']}"
                        self.consolidated_metadata.append(metadata_record)
                        
                        self.stats['total_augmented_images'] += 1
                    else:
                        # Skip original images (already processed from v2.1)
                        self.stats['duplicates_skipped'] += 1
                        continue
                    
                    self.stats['total_images_copied'] += 1
                    
                    if idx % 100 == 0:
                        logger.info(f"Processed {idx+1}/{len(df)} augmented images...")
                else:
                    logger.warning(f"Image not found: {source_img}")
            
            logger.info(f"✅ Processed augmented dataset")
            self.stats['datasets_processed'] += 1
        else:
            # Process without metadata (scan directory)
            logger.info("No metadata found, scanning directory...")
            spec_dir = f"{source_dir}/spectrograms"
            if os.path.exists(spec_dir):
                png_files = [f for f in os.listdir(spec_dir) if f.endswith('.png')]
                logger.info(f"Found {len(png_files)} PNG files")
                
                for idx, filename in enumerate(png_files):
                    if 'aug' in filename:  # Only augmented images
                        source_img = f"{spec_dir}/{filename}"
                        dest_img = f"{self.output_dir}/spectrograms/augmented/{filename}"
                        shutil.copy2(source_img, dest_img)
                        
                        # Create basic metadata record
                        metadata_record = {
                            'spectrogram_file': filename,
                            'dataset_source': 'augmented_no_metadata',
                            'image_type': 'augmented',
                            'unified_path': f"spectrograms/augmented/{filename}"
                        }
                        self.consolidated_metadata.append(metadata_record)
                        
                        self.stats['total_images_copied'] += 1
                        self.stats['total_augmented_images'] += 1
                        
                        if idx % 100 == 0:
                            logger.info(f"Processed {idx+1}/{len(png_files)} files...")
    
    def process_other_datasets(self):
        """Process other datasets if they exist"""
        other_datasets = [
            "dataset_spectrogram_ssh",
            "dataset_spectrogram_ssh_backup", 
            "dataset_spectrogram_ssh_v22"
        ]
        
        for dataset_name in other_datasets:
            if os.path.exists(dataset_name):
                logger.info(f"Found additional dataset: {dataset_name}")
                
                # Check if it has unique data not already processed
                spec_dir = f"{dataset_name}/spectrograms"
                if os.path.exists(spec_dir):
                    png_files = [f for f in os.listdir(spec_dir) if f.endswith('.png')]
                    
                    # Check for unique files
                    existing_files = set()
                    for record in self.consolidated_metadata:
                        existing_files.add(record['spectrogram_file'])
                    
                    unique_files = [f for f in png_files if f not in existing_files]
                    
                    if unique_files:
                        logger.info(f"Found {len(unique_files)} unique files in {dataset_name}")
                        
                        for filename in unique_files:
                            source_img = f"{spec_dir}/{filename}"
                            dest_img = f"{self.output_dir}/spectrograms/original/{filename}"
                            shutil.copy2(source_img, dest_img)
                            
                            # Create basic metadata record
                            metadata_record = {
                                'spectrogram_file': filename,
                                'dataset_source': dataset_name,
                                'image_type': 'original',
                                'unified_path': f"spectrograms/original/{filename}"
                            }
                            self.consolidated_metadata.append(metadata_record)
                            
                            self.stats['total_images_copied'] += 1
                            self.stats['total_original_images'] += 1
                    else:
                        logger.info(f"No unique files found in {dataset_name}")
    
    def save_consolidated_metadata(self):
        """Save consolidated metadata"""
        if not self.consolidated_metadata:
            logger.warning("No metadata to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.consolidated_metadata)
        
        # Save to CSV
        metadata_path = f"{self.output_dir}/metadata/unified_metadata.csv"
        df.to_csv(metadata_path, index=False)
        logger.info(f"Saved consolidated metadata: {metadata_path}")
        
        # Generate summary
        self._generate_summary(df)
        
        return df
    
    def _generate_summary(self, df):
        """Generate unified dataset summary"""
        summary_path = f"{self.output_dir}/metadata/unified_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("UNIFIED GEOMAGNETIC DATASET SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Consolidation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images: {len(df)}\n")
            f.write(f"Original Images: {self.stats['total_original_images']}\n")
            f.write(f"Augmented Images: {self.stats['total_augmented_images']}\n")
            f.write(f"Datasets Processed: {self.stats['datasets_processed']}\n")
            f.write(f"Duplicates Skipped: {self.stats['duplicates_skipped']}\n\n")
            
            # Dataset source distribution
            if 'dataset_source' in df.columns:
                f.write("Dataset Source Distribution:\n")
                f.write("-" * 40 + "\n")
                source_dist = df['dataset_source'].value_counts()
                for source, count in source_dist.items():
                    f.write(f"  {source}: {count} ({count/len(df)*100:.1f}%)\n")
                f.write("\n")
            
            # Image type distribution
            if 'image_type' in df.columns:
                f.write("Image Type Distribution:\n")
                f.write("-" * 40 + "\n")
                type_dist = df['image_type'].value_counts()
                for img_type, count in type_dist.items():
                    f.write(f"  {img_type}: {count} ({count/len(df)*100:.1f}%)\n")
                f.write("\n")
            
            # Azimuth distribution (if available)
            if 'azimuth_class' in df.columns:
                f.write("Azimuth Distribution:\n")
                f.write("-" * 40 + "\n")
                az_dist = df['azimuth_class'].value_counts()
                for az_class, count in az_dist.items():
                    f.write(f"  {az_class}: {count} ({count/len(df)*100:.1f}%)\n")
                f.write("\n")
            
            # Magnitude distribution (if available)
            if 'magnitude_class' in df.columns:
                f.write("Magnitude Distribution:\n")
                f.write("-" * 40 + "\n")
                mag_dist = df['magnitude_class'].value_counts()
                for mag_class, count in mag_dist.items():
                    f.write(f"  {mag_class}: {count} ({count/len(df)*100:.1f}%)\n")
                f.write("\n")
            
            # Station distribution (if available)
            if 'station' in df.columns:
                f.write("Station Distribution:\n")
                f.write("-" * 40 + "\n")
                station_dist = df['station'].value_counts()
                for station, count in station_dist.items():
                    f.write(f"  {station}: {count} ({count/len(df)*100:.1f}%)\n")
        
        logger.info(f"Generated unified summary: {summary_path}")
    
    def consolidate_all(self):
        """Main consolidation process"""
        logger.info("="*70)
        logger.info("STARTING DATASET CONSOLIDATION")
        logger.info("="*70)
        
        # Process each dataset
        self.process_v21_dataset()
        self.process_augmented_dataset()
        self.process_other_datasets()
        
        # Save consolidated metadata
        df = self.save_consolidated_metadata()
        
        # Print final statistics
        logger.info("="*70)
        logger.info("CONSOLIDATION COMPLETED")
        logger.info("="*70)
        logger.info(f"Total Images Copied: {self.stats['total_images_copied']}")
        logger.info(f"Original Images: {self.stats['total_original_images']}")
        logger.info(f"Augmented Images: {self.stats['total_augmented_images']}")
        logger.info(f"Datasets Processed: {self.stats['datasets_processed']}")
        logger.info(f"Duplicates Skipped: {self.stats['duplicates_skipped']}")
        logger.info(f"Output Directory: {self.output_dir}")
        
        return df


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Consolidate multiple geomagnetic datasets into unified structure'
    )
    parser.add_argument('--output-dir', default='dataset_unified',
                       help='Output directory for unified dataset')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually copying files')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be copied")
        return
    
    # Create consolidator and run
    consolidator = DatasetConsolidator(output_dir=args.output_dir)
    df = consolidator.consolidate_all()
    
    print(f"\n✅ Dataset consolidation completed!")
    print(f"   Total images: {len(df) if df is not None else 0}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Ready for development!")


if __name__ == '__main__':
    main()