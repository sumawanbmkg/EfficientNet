#!/usr/bin/env python3
"""
Helper script untuk menggunakan dataset unified
Menyediakan utility functions untuk loading, analysis, dan training preparation
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class UnifiedDatasetHelper:
    """Helper class untuk dataset unified"""
    
    def __init__(self, dataset_dir='dataset_unified'):
        self.dataset_dir = dataset_dir
        self.metadata_file = f"{dataset_dir}/metadata/unified_metadata.csv"
        self.metadata = None
        
        if os.path.exists(self.metadata_file):
            self.metadata = pd.read_csv(self.metadata_file)
            print(f"✅ Loaded metadata: {len(self.metadata)} records")
        else:
            print(f"❌ Metadata file not found: {self.metadata_file}")
    
    def get_dataset_stats(self):
        """Get comprehensive dataset statistics"""
        if self.metadata is None:
            return None
        
        stats = {
            'total_images': len(self.metadata),
            'original_images': len(self.metadata[self.metadata['image_type'] == 'original']),
            'augmented_images': len(self.metadata[self.metadata['image_type'] == 'augmented']),
        }
        
        # Distribution stats
        if 'azimuth_class' in self.metadata.columns:
            stats['azimuth_distribution'] = self.metadata['azimuth_class'].value_counts().to_dict()
        
        if 'magnitude_class' in self.metadata.columns:
            stats['magnitude_distribution'] = self.metadata['magnitude_class'].value_counts().to_dict()
        
        if 'station' in self.metadata.columns:
            stats['station_distribution'] = self.metadata['station'].value_counts().to_dict()
        
        return stats
    
    def calculate_class_weights(self, target_column='azimuth_class'):
        """Calculate class weights for imbalanced dataset"""
        if self.metadata is None or target_column not in self.metadata.columns:
            return None
        
        class_counts = self.metadata[target_column].value_counts()
        total_samples = len(self.metadata)
        num_classes = len(class_counts)
        
        # Calculate inverse frequency weights
        weights = {}
        for class_name, count in class_counts.items():
            weights[class_name] = total_samples / (num_classes * count)
        
        return weights
    
    def get_balanced_subset(self, target_column='azimuth_class', samples_per_class=None):
        """Get balanced subset of dataset"""
        if self.metadata is None or target_column not in self.metadata.columns:
            return None
        
        class_counts = self.metadata[target_column].value_counts()
        
        if samples_per_class is None:
            # Use minimum class count
            samples_per_class = class_counts.min()
        
        balanced_indices = []
        for class_name in class_counts.index:
            class_indices = self.metadata[self.metadata[target_column] == class_name].index
            selected_indices = np.random.choice(
                class_indices, 
                size=min(samples_per_class, len(class_indices)), 
                replace=False
            )
            balanced_indices.extend(selected_indices)
        
        return self.metadata.loc[balanced_indices]
    
    def create_train_val_test_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                                   stratify_column='azimuth_class', random_state=42):
        """Create train/validation/test split (simple random if stratification fails)"""
        if self.metadata is None:
            return None, None, None
        
        from sklearn.model_selection import train_test_split
        
        # Filter out rows with NaN in stratify column
        if stratify_column in self.metadata.columns:
            clean_data = self.metadata.dropna(subset=[stratify_column])
        else:
            clean_data = self.metadata
        
        try:
            # Try stratified split first
            train_data, temp_data = train_test_split(
                clean_data,
                test_size=(val_ratio + test_ratio),
                stratify=clean_data[stratify_column] if stratify_column in clean_data.columns else None,
                random_state=random_state
            )
            
            # Second split: val vs test
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - val_ratio_adjusted),
                stratify=temp_data[stratify_column] if stratify_column in temp_data.columns else None,
                random_state=random_state
            )
            
            print(f"✅ Stratified split successful")
            
        except ValueError as e:
            print(f"⚠️  Stratified split failed: {e}")
            print(f"   Using random split instead...")
            
            # Fall back to random split
            train_data, temp_data = train_test_split(
                clean_data,
                test_size=(val_ratio + test_ratio),
                random_state=random_state
            )
            
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - val_ratio_adjusted),
                random_state=random_state
            )
        
        print(f"Dataset split:")
        print(f"  Train: {len(train_data)} samples ({len(train_data)/len(clean_data)*100:.1f}%)")
        print(f"  Val: {len(val_data)} samples ({len(val_data)/len(clean_data)*100:.1f}%)")
        print(f"  Test: {len(test_data)} samples ({len(test_data)/len(clean_data)*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def plot_distributions(self, save_path=None):
        """Plot dataset distributions"""
        if self.metadata is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Azimuth distribution
        if 'azimuth_class' in self.metadata.columns:
            az_counts = self.metadata['azimuth_class'].value_counts()
            axes[0, 0].bar(az_counts.index, az_counts.values)
            axes[0, 0].set_title('Azimuth Distribution')
            axes[0, 0].set_xlabel('Azimuth Class')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Magnitude distribution
        if 'magnitude_class' in self.metadata.columns:
            mag_counts = self.metadata['magnitude_class'].value_counts()
            axes[0, 1].bar(mag_counts.index, mag_counts.values)
            axes[0, 1].set_title('Magnitude Distribution')
            axes[0, 1].set_xlabel('Magnitude Class')
            axes[0, 1].set_ylabel('Count')
        
        # Station distribution (top 10)
        if 'station' in self.metadata.columns:
            station_counts = self.metadata['station'].value_counts().head(10)
            axes[1, 0].bar(station_counts.index, station_counts.values)
            axes[1, 0].set_title('Top 10 Stations')
            axes[1, 0].set_xlabel('Station')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Image type distribution
        if 'image_type' in self.metadata.columns:
            type_counts = self.metadata['image_type'].value_counts()
            axes[1, 1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Image Type Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def export_for_training(self, output_dir='training_data', image_type='both'):
        """Export dataset in format ready for training"""
        if self.metadata is None:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter by image type
        if image_type == 'original':
            filtered_data = self.metadata[self.metadata['image_type'] == 'original']
        elif image_type == 'augmented':
            filtered_data = self.metadata[self.metadata['image_type'] == 'augmented']
        else:  # both
            filtered_data = self.metadata
        
        # Remove rows with NaN in critical columns
        filtered_data = filtered_data.dropna(subset=['azimuth_class', 'magnitude_class'])
        
        # Create training metadata
        training_metadata = filtered_data[['spectrogram_file', 'azimuth_class', 'magnitude_class', 
                                         'station', 'unified_path']].copy()
        
        # Save training metadata
        training_metadata.to_csv(f"{output_dir}/training_metadata.csv", index=False)
        
        # Create class mapping
        azimuth_classes = sorted([cls for cls in training_metadata['azimuth_class'].unique() if pd.notna(cls)])
        magnitude_classes = sorted([cls for cls in training_metadata['magnitude_class'].unique() if pd.notna(cls)])
        
        class_mapping = {
            'azimuth_to_idx': {cls: idx for idx, cls in enumerate(azimuth_classes)},
            'magnitude_to_idx': {cls: idx for idx, cls in enumerate(magnitude_classes)},
            'idx_to_azimuth': {idx: cls for idx, cls in enumerate(azimuth_classes)},
            'idx_to_magnitude': {idx: cls for idx, cls in enumerate(magnitude_classes)}
        }
        
        # Save class mapping
        import json
        with open(f"{output_dir}/class_mapping.json", 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        print(f"Training data exported to: {output_dir}")
        print(f"  Samples: {len(training_metadata)}")
        print(f"  Azimuth classes: {len(azimuth_classes)}")
        print(f"  Magnitude classes: {len(magnitude_classes)}")
        
        return training_metadata, class_mapping
    
    def print_summary(self):
        """Print comprehensive dataset summary"""
        if self.metadata is None:
            print("❌ No metadata available")
            return
        
        print("="*70)
        print("DATASET UNIFIED SUMMARY")
        print("="*70)
        
        stats = self.get_dataset_stats()
        
        print(f"📊 OVERVIEW:")
        print(f"   Total Images: {stats['total_images']}")
        print(f"   Original Images: {stats['original_images']} ({stats['original_images']/stats['total_images']*100:.1f}%)")
        print(f"   Augmented Images: {stats['augmented_images']} ({stats['augmented_images']/stats['total_images']*100:.1f}%)")
        
        print(f"\n🎯 AZIMUTH DISTRIBUTION:")
        if 'azimuth_distribution' in stats:
            for az_class, count in stats['azimuth_distribution'].items():
                print(f"   {az_class}: {count} ({count/stats['total_images']*100:.1f}%)")
        
        print(f"\n📏 MAGNITUDE DISTRIBUTION:")
        if 'magnitude_distribution' in stats:
            for mag_class, count in stats['magnitude_distribution'].items():
                print(f"   {mag_class}: {count} ({count/stats['total_images']*100:.1f}%)")
        
        print(f"\n🏢 TOP 10 STATIONS:")
        if 'station_distribution' in stats:
            for i, (station, count) in enumerate(list(stats['station_distribution'].items())[:10]):
                print(f"   {station}: {count} ({count/stats['total_images']*100:.1f}%)")
        
        # Class imbalance analysis
        print(f"\n⚠️  CLASS IMBALANCE ANALYSIS:")
        if 'azimuth_distribution' in stats:
            az_counts = list(stats['azimuth_distribution'].values())
            az_ratio = max(az_counts) / min(az_counts)
            print(f"   Azimuth imbalance ratio: {az_ratio:.1f}:1")
        
        if 'magnitude_distribution' in stats:
            mag_counts = list(stats['magnitude_distribution'].values())
            mag_ratio = max(mag_counts) / min(mag_counts)
            print(f"   Magnitude imbalance ratio: {mag_ratio:.1f}:1")


def main():
    """Main function for testing"""
    helper = UnifiedDatasetHelper()
    
    # Print summary
    helper.print_summary()
    
    # Calculate class weights
    print(f"\n🔢 CLASS WEIGHTS (Azimuth):")
    az_weights = helper.calculate_class_weights('azimuth_class')
    if az_weights:
        for class_name, weight in az_weights.items():
            print(f"   {class_name}: {weight:.2f}")
    
    print(f"\n🔢 CLASS WEIGHTS (Magnitude):")
    mag_weights = helper.calculate_class_weights('magnitude_class')
    if mag_weights:
        for class_name, weight in mag_weights.items():
            print(f"   {class_name}: {weight:.2f}")
    
    # Create train/val/test split
    print(f"\n📊 CREATING TRAIN/VAL/TEST SPLIT:")
    train_data, val_data, test_data = helper.create_train_val_test_split()
    
    # Export for training
    print(f"\n📤 EXPORTING FOR TRAINING:")
    training_metadata, class_mapping = helper.export_for_training()


if __name__ == '__main__':
    main()