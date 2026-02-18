#!/usr/bin/env python3
"""
SMOTE Data Augmentation for Minority Classes
Generates synthetic samples to balance extreme class imbalance

Target: Generate synthetic samples for minority azimuth classes
- NE class: 4 → 100 samples
- E class: 48 → 100 samples  
- SE class: 92 → 100 samples

Author: Earthquake Prediction Research Team
Date: February 1, 2026
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpectrogramSMOTE:
    """
    SMOTE augmentation for spectrogram images
    Generates synthetic samples in feature space
    """
    
    def __init__(self, dataset_dir: str = 'dataset_unified', output_dir: str = 'dataset_smote'):
        """
        Initialize SMOTE augmentation
        
        Args:
            dataset_dir: Path to original dataset
            output_dir: Path to save augmented dataset
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'spectrograms').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        
        # Load metadata
        metadata_path = self.dataset_dir / 'metadata' / 'unified_metadata.csv'
        self.metadata = pd.read_csv(metadata_path)
        
        # Clean data
        self._clean_metadata()
        
        logger.info(f"📊 Loaded {len(self.metadata)} samples")
        
    def _clean_metadata(self):
        """Clean metadata - remove NaN values"""
        original_size = len(self.metadata)
        
        # Remove NaN values
        self.metadata = self.metadata.dropna(subset=['magnitude_class', 'azimuth_class'])
        self.metadata['magnitude_class'] = self.metadata['magnitude_class'].astype(str)
        self.metadata['azimuth_class'] = self.metadata['azimuth_class'].astype(str)
        
        # Remove 'nan' strings
        self.metadata = self.metadata[
            (self.metadata['magnitude_class'] != 'nan') & 
            (self.metadata['azimuth_class'] != 'nan')
        ]
        
        cleaned_size = len(self.metadata)
        logger.info(f"🧹 Cleaned: {original_size} → {cleaned_size} samples")
        
    def analyze_class_distribution(self):
        """Analyze and display class distribution"""
        logger.info("\n📊 CLASS DISTRIBUTION ANALYSIS")
        logger.info("=" * 60)
        
        # Magnitude distribution
        mag_dist = self.metadata['magnitude_class'].value_counts().sort_index()
        logger.info("\n🔢 Magnitude Classes:")
        for cls, count in mag_dist.items():
            logger.info(f"  {cls}: {count} samples")
        
        # Azimuth distribution
        az_dist = self.metadata['azimuth_class'].value_counts().sort_index()
        logger.info("\n🧭 Azimuth Classes:")
        for cls, count in az_dist.items():
            logger.info(f"  {cls}: {count} samples")
        
        # Identify minority classes (< 100 samples)
        minority_classes = az_dist[az_dist < 100].index.tolist()
        logger.info(f"\n⚠️  Minority classes (< 100 samples): {minority_classes}")
        
        return minority_classes
    
    def load_images_as_features(self, sample_indices):
        """
        Load images and convert to feature vectors
        
        Args:
            sample_indices: List of sample indices to load
            
        Returns:
            features: Flattened image features (n_samples, n_features)
            labels: Corresponding labels
        """
        features = []
        mag_labels = []
        az_labels = []
        
        logger.info(f"📥 Loading {len(sample_indices)} images...")
        
        for idx in tqdm(sample_indices, desc="Loading images"):
            sample = self.metadata.iloc[idx]
            
            # Load image
            image_path = self.dataset_dir / sample['unified_path']
            
            if not image_path.exists():
                logger.warning(f"⚠️  Image not found: {image_path}")
                continue
            
            try:
                # Load and resize to consistent size
                img = Image.open(image_path).convert('RGB')
                img = img.resize((64, 64), Image.LANCZOS)  # Smaller size for SMOTE
                
                # Convert to numpy and flatten
                img_array = np.array(img).flatten()
                
                features.append(img_array)
                mag_labels.append(sample['magnitude_class'])
                az_labels.append(sample['azimuth_class'])
                
            except Exception as e:
                logger.error(f"❌ Error loading {image_path}: {e}")
                continue
        
        features = np.array(features)
        
        logger.info(f"✅ Loaded {len(features)} images")
        logger.info(f"📐 Feature shape: {features.shape}")
        
        return features, mag_labels, az_labels
    
    def apply_smote(self, target_samples_per_class=100):
        """
        Apply SMOTE to generate synthetic samples
        
        Args:
            target_samples_per_class: Target number of samples per minority class
        """
        logger.info("\n🔄 APPLYING SMOTE AUGMENTATION")
        logger.info("=" * 60)
        
        # Analyze distribution
        minority_classes = self.analyze_class_distribution()
        
        # Get all sample indices
        all_indices = list(range(len(self.metadata)))
        
        # Load images as features
        features, mag_labels, az_labels = self.load_images_as_features(all_indices)
        
        # Encode labels
        az_encoder = LabelEncoder()
        az_encoded = az_encoder.fit_transform(az_labels)
        
        # Calculate sampling strategy
        az_dist = pd.Series(az_labels).value_counts()
        sampling_strategy = {}
        
        for cls in minority_classes:
            if cls in az_dist.index:
                current_count = az_dist[cls]
                if current_count < target_samples_per_class:
                    sampling_strategy[cls] = target_samples_per_class
        
        logger.info(f"\n🎯 Sampling strategy: {sampling_strategy}")
        
        # Apply SMOTE
        logger.info("\n⚙️  Applying SMOTE...")
        
        # Convert sampling strategy to encoded labels
        encoded_strategy = {}
        for cls, count in sampling_strategy.items():
            encoded_label = az_encoder.transform([cls])[0]
            encoded_strategy[encoded_label] = count
        
        smote = SMOTE(
            sampling_strategy=encoded_strategy,
            k_neighbors=min(3, min([az_dist[cls] for cls in minority_classes]) - 1),
            random_state=42
        )
        
        try:
            features_resampled, az_encoded_resampled = smote.fit_resample(features, az_encoded)
            
            logger.info(f"✅ SMOTE completed!")
            logger.info(f"📊 Original samples: {len(features)}")
            logger.info(f"📊 Augmented samples: {len(features_resampled)}")
            logger.info(f"📊 New synthetic samples: {len(features_resampled) - len(features)}")
            
            # Decode labels
            az_labels_resampled = az_encoder.inverse_transform(az_encoded_resampled)
            
            # Save synthetic samples
            self._save_synthetic_samples(
                features_resampled[len(features):],  # Only new samples
                az_labels_resampled[len(features):],
                mag_labels  # Use original magnitude distribution
            )
            
            # Update metadata
            self._update_metadata(az_labels_resampled, mag_labels)
            
        except Exception as e:
            logger.error(f"❌ SMOTE failed: {e}")
            logger.error("This might be due to insufficient samples for k_neighbors")
            logger.error("Try collecting more data or reducing target_samples_per_class")
    
    def _save_synthetic_samples(self, synthetic_features, synthetic_az_labels, original_mag_labels):
        """Save synthetic samples as images"""
        logger.info("\n💾 Saving synthetic samples...")
        
        # Determine magnitude labels for synthetic samples (random from original distribution)
        mag_dist = pd.Series(original_mag_labels).value_counts(normalize=True)
        synthetic_mag_labels = np.random.choice(
            mag_dist.index.tolist(),
            size=len(synthetic_features),
            p=mag_dist.values
        )
        
        synthetic_metadata = []
        
        for idx, (features, az_label, mag_label) in enumerate(
            tqdm(
                zip(synthetic_features, synthetic_az_labels, synthetic_mag_labels),
                total=len(synthetic_features),
                desc="Saving synthetic images"
            )
        ):
            # Reshape features back to image
            img_array = features.reshape(64, 64, 3).astype(np.uint8)
            img = Image.fromarray(img_array)
            
            # Resize to original size
            img = img.resize((224, 224), Image.LANCZOS)
            
            # Generate filename
            filename = f"synthetic_{az_label}_{mag_label}_{idx:04d}.png"
            filepath = self.output_dir / 'spectrograms' / filename
            
            # Save image
            img.save(filepath)
            
            # Add to metadata
            synthetic_metadata.append({
                'unified_path': f'spectrograms/{filename}',
                'magnitude_class': mag_label,
                'azimuth_class': az_label,
                'is_synthetic': True,
                'synthetic_method': 'SMOTE'
            })
        
        # Save synthetic metadata
        synthetic_df = pd.DataFrame(synthetic_metadata)
        synthetic_df.to_csv(
            self.output_dir / 'metadata' / 'synthetic_metadata.csv',
            index=False
        )
        
        logger.info(f"✅ Saved {len(synthetic_metadata)} synthetic samples")
    
    def _update_metadata(self, all_az_labels, all_mag_labels):
        """Update and save combined metadata"""
        logger.info("\n📝 Updating metadata...")
        
        # Copy original images to output directory
        logger.info("📋 Copying original images...")
        
        # Get unique subdirectories
        subdirs = set()
        for path in self.metadata['unified_path']:
            parts = Path(path).parts
            if len(parts) > 1:
                subdirs.add(parts[1])  # e.g., 'original', 'augmented', 'normal'
        
        logger.info(f"Found subdirectories: {subdirs}")
        
        # Create subdirectories
        for subdir in subdirs:
            (self.output_dir / 'spectrograms' / subdir).mkdir(parents=True, exist_ok=True)
        
        # Copy images
        copied_count = 0
        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Copying images"):
            src_path = self.dataset_dir / row['unified_path']
            dst_path = self.output_dir / row['unified_path']
            
            # Copy if not exists
            if src_path.exists() and not dst_path.exists():
                import shutil
                shutil.copy2(src_path, dst_path)
                copied_count += 1
        
        logger.info(f"✅ Copied {copied_count} original images")
        
        # Copy original metadata
        combined_metadata = self.metadata.copy()
        combined_metadata['is_synthetic'] = False
        combined_metadata['synthetic_method'] = None
        
        # Load synthetic metadata
        synthetic_df = pd.read_csv(
            self.output_dir / 'metadata' / 'synthetic_metadata.csv'
        )
        
        # Combine
        combined_metadata = pd.concat([combined_metadata, synthetic_df], ignore_index=True)
        
        # Save combined metadata
        combined_metadata.to_csv(
            self.output_dir / 'metadata' / 'combined_metadata.csv',
            index=False
        )
        
        logger.info(f"✅ Combined metadata saved: {len(combined_metadata)} total samples")
        
        # Display new distribution
        logger.info("\n📊 NEW CLASS DISTRIBUTION:")
        logger.info("=" * 60)
        
        new_az_dist = combined_metadata['azimuth_class'].value_counts().sort_index()
        logger.info("\n🧭 Azimuth Classes (after SMOTE):")
        for cls, count in new_az_dist.items():
            original_count = self.metadata['azimuth_class'].value_counts().get(cls, 0)
            synthetic_count = count - original_count
            logger.info(f"  {cls}: {count} samples (original: {original_count}, synthetic: {synthetic_count})")


def main():
    """Main function"""
    logger.info("🚀 SMOTE DATA AUGMENTATION")
    logger.info("=" * 60)
    
    # Initialize SMOTE augmentation
    smote_aug = SpectrogramSMOTE(
        dataset_dir='dataset_unified',
        output_dir='dataset_smote'
    )
    
    # Apply SMOTE with target of 100 samples per minority class
    smote_aug.apply_smote(target_samples_per_class=100)
    
    logger.info("\n✅ SMOTE AUGMENTATION COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"📁 Output directory: dataset_smote/")
    logger.info(f"📊 Synthetic samples: dataset_smote/spectrograms/")
    logger.info(f"📝 Metadata: dataset_smote/metadata/combined_metadata.csv")
    logger.info("\n🎯 Next steps:")
    logger.info("1. Review synthetic samples in dataset_smote/spectrograms/")
    logger.info("2. Train model with combined dataset")
    logger.info("3. Compare performance with baseline")


if __name__ == '__main__':
    main()
