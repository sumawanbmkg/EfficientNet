"""
Dataset loader for earthquake spectrogram data
Supports multi-task learning (magnitude + azimuth)
With Physics-Preserving Augmentation for ULF spectrograms
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysicsPreservingAugmentation:
    """
    Physics-preserving augmentation for ULF geomagnetic spectrograms
    Preserves frequency axis (vertical) and time relationships
    """
    
    def __init__(self, 
                 time_mask_param=30,      # Max time masking width
                 freq_mask_param=15,      # Max frequency masking width
                 noise_std=0.05,          # Gaussian noise std
                 brightness_range=0.2,    # Brightness variation
                 contrast_range=0.2,      # Contrast variation
                 time_shift_max=10,       # Max time shift in pixels
                 gain_range=0.15,         # Energy scaling range
                 p_time_mask=0.5,         # Probability of time masking
                 p_freq_mask=0.5,         # Probability of freq masking
                 p_noise=0.3,             # Probability of adding noise
                 p_time_shift=0.3,        # Probability of time shift
                 p_gain=0.3):             # Probability of gain change
        
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.time_shift_max = time_shift_max
        self.gain_range = gain_range
        
        self.p_time_mask = p_time_mask
        self.p_freq_mask = p_freq_mask
        self.p_noise = p_noise
        self.p_time_shift = p_time_shift
        self.p_gain = p_gain
    
    def time_masking(self, spec):
        """
        Mask random time segments (vertical bands)
        Simulates missing data or sensor dropout
        """
        if torch.rand(1).item() > self.p_time_mask:
            return spec
        
        _, _, width = spec.shape
        mask_width = torch.randint(1, self.time_mask_param, (1,)).item()
        mask_start = torch.randint(0, max(1, width - mask_width), (1,)).item()
        
        spec_masked = spec.clone()
        spec_masked[:, :, mask_start:mask_start + mask_width] = 0
        
        return spec_masked
    
    def frequency_masking(self, spec):
        """
        Mask random frequency bands (horizontal bands)
        Simulates frequency-specific interference
        """
        if torch.rand(1).item() > self.p_freq_mask:
            return spec
        
        _, height, _ = spec.shape
        mask_height = torch.randint(1, self.freq_mask_param, (1,)).item()
        mask_start = torch.randint(0, max(1, height - mask_height), (1,)).item()
        
        spec_masked = spec.clone()
        spec_masked[:, mask_start:mask_start + mask_height, :] = 0
        
        return spec_masked
    
    def add_gaussian_noise(self, spec):
        """
        Add Gaussian noise to simulate instrument noise
        """
        if torch.rand(1).item() > self.p_noise:
            return spec
        
        noise = torch.randn_like(spec) * self.noise_std
        return spec + noise
    
    def time_shift(self, spec):
        """
        Shift spectrogram in time (horizontal shift)
        Simulates different event timing
        """
        if torch.rand(1).item() > self.p_time_shift:
            return spec
        
        shift = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item()
        
        if shift == 0:
            return spec
        
        # Roll along width dimension
        return torch.roll(spec, shifts=shift, dims=2)
    
    def gain_scaling(self, spec):
        """
        Scale intensity (energy) to simulate magnitude variations
        """
        if torch.rand(1).item() > self.p_gain:
            return spec
        
        gain = 1.0 + torch.rand(1).item() * 2 * self.gain_range - self.gain_range
        return spec * gain
    
    def __call__(self, spec):
        """
        Apply augmentation pipeline
        Order: noise -> gain -> time_shift -> masking
        """
        # Add noise first (most subtle)
        spec = self.add_gaussian_noise(spec)
        
        # Gain scaling (energy variation)
        spec = self.gain_scaling(spec)
        
        # Time shift (preserves frequency structure)
        spec = self.time_shift(spec)
        
        # Masking (most aggressive, applied last)
        spec = self.time_masking(spec)
        spec = self.frequency_masking(spec)
        
        # Clamp to valid range
        spec = torch.clamp(spec, 0, 1)
        
        return spec


class EarthquakeSpectrogramDataset(Dataset):
    """
    Dataset for earthquake spectrogram images
    Returns: image, magnitude_label, azimuth_label
    """
    
    def __init__(self, metadata_df, image_dir, transform=None):
        """
        Initialize dataset
        
        Args:
            metadata_df: DataFrame with columns: spectrogram_file, magnitude_class, azimuth_class
            image_dir: Directory containing spectrogram images
            transform: Image transformations
        """
        self.metadata = metadata_df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        
        # Class mappings
        self.magnitude_classes = ['Small', 'Moderate', 'Medium', 'Large', 'Major']
        self.azimuth_classes = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        # Create label encoders
        self.magnitude_to_idx = {cls: idx for idx, cls in enumerate(self.magnitude_classes)}
        self.azimuth_to_idx = {cls: idx for idx, cls in enumerate(self.azimuth_classes)}
        
        self.idx_to_magnitude = {idx: cls for cls, idx in self.magnitude_to_idx.items()}
        self.idx_to_azimuth = {idx: cls for cls, idx in self.azimuth_to_idx.items()}
        
        logger.info(f"Dataset initialized with {len(self.metadata)} samples")
        logger.info(f"Magnitude classes: {self.magnitude_classes}")
        logger.info(f"Azimuth classes: {self.azimuth_classes}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """Get single sample"""
        row = self.metadata.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, row['spectrogram_file'])
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        magnitude_label = self.magnitude_to_idx[row['magnitude_class']]
        azimuth_label = self.azimuth_to_idx[row['azimuth_class']]
        
        return image, magnitude_label, azimuth_label
    
    def get_class_weights(self):
        """Compute class weights for imbalanced data"""
        # Magnitude class weights
        mag_counts = self.metadata['magnitude_class'].value_counts()
        mag_weights = 1.0 / mag_counts
        mag_weights = mag_weights / mag_weights.sum() * len(self.magnitude_classes)
        
        # Azimuth class weights
        az_counts = self.metadata['azimuth_class'].value_counts()
        az_weights = 1.0 / az_counts
        az_weights = az_weights / az_weights.sum() * len(self.azimuth_classes)
        
        return mag_weights.to_dict(), az_weights.to_dict()


def get_transforms(augment=True, augmentation_level='moderate'):
    """
    Get image transformations with physics-preserving augmentation
    
    Args:
        augment: Apply data augmentation
        augmentation_level: 'light', 'moderate', 'aggressive'
        
    Returns:
        transform: Composed transforms
    """
    # Base transforms (always applied)
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    
    if augment:
        # Physics-preserving augmentation parameters
        if augmentation_level == 'light':
            aug_params = {
                'time_mask_param': 20,
                'freq_mask_param': 10,
                'noise_std': 0.03,
                'time_shift_max': 5,
                'gain_range': 0.1,
                'p_time_mask': 0.3,
                'p_freq_mask': 0.3,
                'p_noise': 0.2,
                'p_time_shift': 0.2,
                'p_gain': 0.2
            }
        elif augmentation_level == 'moderate':
            aug_params = {
                'time_mask_param': 30,
                'freq_mask_param': 15,
                'noise_std': 0.05,
                'time_shift_max': 10,
                'gain_range': 0.15,
                'p_time_mask': 0.5,
                'p_freq_mask': 0.5,
                'p_noise': 0.3,
                'p_time_shift': 0.3,
                'p_gain': 0.3
            }
        else:  # aggressive
            aug_params = {
                'time_mask_param': 40,
                'freq_mask_param': 20,
                'noise_std': 0.08,
                'time_shift_max': 15,
                'gain_range': 0.2,
                'p_time_mask': 0.7,
                'p_freq_mask': 0.7,
                'p_noise': 0.5,
                'p_time_shift': 0.5,
                'p_gain': 0.5
            }
        
        # Create augmentation pipeline
        physics_aug = PhysicsPreservingAugmentation(**aug_params)
        
        # Mild color jitter (for visual variations only)
        color_jitter = transforms.ColorJitter(
            brightness=0.15, 
            contrast=0.15,
            saturation=0.1
        )
        
        transform = transforms.Compose([
            *base_transforms,
            physics_aug,  # Apply physics-preserving augmentation
            transforms.Lambda(lambda x: color_jitter(transforms.ToPILImage()(x))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # No augmentation (validation/test)
        transform = transforms.Compose([
            *base_transforms,
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(dataset_dir, batch_size=32, val_split=0.2, test_split=0.1,
                       num_workers=4, seed=42, augmentation_level='moderate'):
    """
    Create train/val/test dataloaders with physics-preserving augmentation
    
    Args:
        dataset_dir: Directory containing dataset (with metadata/dataset_metadata.csv)
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        num_workers: Number of workers for data loading
        seed: Random seed
        augmentation_level: 'light', 'moderate', 'aggressive'
        
    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    # Load metadata
    metadata_path = os.path.join(dataset_dir, 'metadata', 'dataset_metadata.csv')
    metadata_df = pd.read_csv(metadata_path)
    
    logger.info(f"Loaded metadata: {len(metadata_df)} samples")
    logger.info(f"Augmentation level: {augmentation_level}")
    
    # Image directory
    image_dir = os.path.join(dataset_dir, 'spectrograms')
    
    # Split data
    train_val_df, test_df = train_test_split(
        metadata_df, test_size=test_split, random_state=seed,
        stratify=metadata_df['magnitude_class']
    )
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_split/(1-test_split), random_state=seed,
        stratify=train_val_df['magnitude_class']
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets with physics-preserving augmentation
    train_dataset = EarthquakeSpectrogramDataset(
        train_df, image_dir, 
        transform=get_transforms(augment=True, augmentation_level=augmentation_level)
    )
    
    val_dataset = EarthquakeSpectrogramDataset(
        val_df, image_dir, 
        transform=get_transforms(augment=False)
    )
    
    test_dataset = EarthquakeSpectrogramDataset(
        test_df, image_dir, 
        transform=get_transforms(augment=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Dataset info
    dataset_info = {
        'num_samples': len(metadata_df),
        'num_train': len(train_df),
        'num_val': len(val_df),
        'num_test': len(test_df),
        'magnitude_classes': train_dataset.magnitude_classes,
        'azimuth_classes': train_dataset.azimuth_classes,
        'num_magnitude_classes': len(train_dataset.magnitude_classes),
        'num_azimuth_classes': len(train_dataset.azimuth_classes),
        'magnitude_distribution': metadata_df['magnitude_class'].value_counts().to_dict(),
        'azimuth_distribution': metadata_df['azimuth_class'].value_counts().to_dict(),
        'augmentation_level': augmentation_level
    }
    
    return train_loader, val_loader, test_loader, dataset_info


if __name__ == '__main__':
    # Test dataset loader
    print("Testing Earthquake Dataset Loader...")
    
    # Use test_cnn_output for testing
    dataset_dir = 'test_cnn_output'
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        print("Please run test_cnn_format.py first to generate test data")
    else:
        try:
            train_loader, val_loader, test_loader, info = create_dataloaders(
                dataset_dir, batch_size=2, num_workers=0
            )
            
            print(f"\nDataset Info:")
            print(f"  Total samples: {info['num_samples']}")
            print(f"  Train: {info['num_train']}")
            print(f"  Val: {info['num_val']}")
            print(f"  Test: {info['num_test']}")
            print(f"  Magnitude classes: {info['magnitude_classes']}")
            print(f"  Azimuth classes: {info['azimuth_classes']}")
            
            # Test batch
            images, mag_labels, az_labels = next(iter(train_loader))
            print(f"\nBatch shapes:")
            print(f"  Images: {images.shape}")
            print(f"  Magnitude labels: {mag_labels.shape}")
            print(f"  Azimuth labels: {az_labels.shape}")
            
            print("\n[OK] Dataset loader test passed!")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
