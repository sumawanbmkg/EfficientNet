#!/usr/bin/env python3
"""
Enhanced Dataset Module for EarthquakeCNN V3.0
Includes advanced augmentation techniques and progressive resizing support

Features:
- RandAugment for automatic augmentation
- MixUp and CutMix for better generalization
- Physics-preserving augmentation
- Progressive resizing support
- Efficient data loading

Author: Earthquake Prediction Research Team
Date: 30 January 2026
Version: 3.0
"""

import os
import random
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable, List

import numpy as np
import pandas as pd
from PIL import Image
# import cv2  # Not needed for current implementation

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment

from sklearn.model_selection import train_test_split


class PhysicsPreservingAugmentation:
    """
    Physics-preserving augmentation for geomagnetic spectrograms
    Maintains frequency domain properties while adding variation
    """
    
    def __init__(self, 
                 time_mask_prob: float = 0.3,
                 freq_mask_prob: float = 0.3,
                 noise_prob: float = 0.2,
                 time_shift_prob: float = 0.2,
                 gain_prob: float = 0.2):
        """
        Initialize physics-preserving augmentation
        
        Args:
            time_mask_prob: Probability of time masking
            freq_mask_prob: Probability of frequency masking
            noise_prob: Probability of adding noise
            time_shift_prob: Probability of time shifting
            gain_prob: Probability of gain scaling
        """
        self.time_mask_prob = time_mask_prob
        self.freq_mask_prob = freq_mask_prob
        self.noise_prob = noise_prob
        self.time_shift_prob = time_shift_prob
        self.gain_prob = gain_prob
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply physics-preserving augmentation
        
        Args:
            image: Input tensor (C, H, W)
            
        Returns:
            Augmented tensor
        """
        # Convert to numpy for processing
        if isinstance(image, torch.Tensor):
            img_np = image.permute(1, 2, 0).numpy()
        else:
            img_np = np.array(image)
        
        # Time masking (vertical bands)
        if random.random() < self.time_mask_prob:
            img_np = self._time_mask(img_np)
        
        # Frequency masking (horizontal bands)
        if random.random() < self.freq_mask_prob:
            img_np = self._freq_mask(img_np)
        
        # Gaussian noise
        if random.random() < self.noise_prob:
            img_np = self._add_noise(img_np)
        
        # Time shift
        if random.random() < self.time_shift_prob:
            img_np = self._time_shift(img_np)
        
        # Gain scaling
        if random.random() < self.gain_prob:
            img_np = self._gain_scale(img_np)
        
        # Convert back to tensor
        if len(img_np.shape) == 3:
            return torch.from_numpy(img_np).permute(2, 0, 1).float()
        else:
            return torch.from_numpy(img_np).unsqueeze(0).float()
    
    def _time_mask(self, img: np.ndarray, max_width: int = 20) -> np.ndarray:
        """Apply time masking (vertical bands)"""
        h, w = img.shape[:2]
        mask_width = random.randint(1, min(max_width, w // 4))
        mask_start = random.randint(0, w - mask_width)
        
        img_masked = img.copy()
        img_masked[:, mask_start:mask_start + mask_width] = 0
        return img_masked
    
    def _freq_mask(self, img: np.ndarray, max_height: int = 15) -> np.ndarray:
        """Apply frequency masking (horizontal bands)"""
        h, w = img.shape[:2]
        mask_height = random.randint(1, min(max_height, h // 4))
        mask_start = random.randint(0, h - mask_height)
        
        img_masked = img.copy()
        img_masked[mask_start:mask_start + mask_height, :] = 0
        return img_masked
    
    def _add_noise(self, img: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_factor, img.shape)
        return np.clip(img + noise, 0, 1)
    
    def _time_shift(self, img: np.ndarray, max_shift: int = 10) -> np.ndarray:
        """Apply time shift (horizontal translation)"""
        h, w = img.shape[:2]
        shift = random.randint(-max_shift, max_shift)
        
        if shift == 0:
            return img
        
        img_shifted = np.zeros_like(img)
        if shift > 0:
            img_shifted[:, shift:] = img[:, :-shift]
        else:
            img_shifted[:, :shift] = img[:, -shift:]
        
        return img_shifted
    
    def _gain_scale(self, img: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply gain scaling"""
        scale = random.uniform(*scale_range)
        return np.clip(img * scale, 0, 1)


class MixUp:
    """
    MixUp augmentation for better generalization
    Paper: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize MixUp
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
        
    def __call__(self, batch: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to batch
        
        Args:
            batch: Input batch (N, C, H, W)
            targets: Target labels (N,)
            
        Returns:
            mixed_batch: Mixed batch
            targets_a: Original targets
            targets_b: Shuffled targets
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch.size(0)
        index = torch.randperm(batch_size)
        
        mixed_batch = lam * batch + (1 - lam) * batch[index, :]
        targets_a, targets_b = targets, targets[index]
        
        return mixed_batch, targets_a, targets_b, lam


class CutMix:
    """
    CutMix augmentation for better localization
    Paper: "CutMix: Regularization Strategy to Train Strong Classifiers" (Yun et al., 2019)
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
        
    def __call__(self, batch: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to batch
        
        Args:
            batch: Input batch (N, C, H, W)
            targets: Target labels (N,)
            
        Returns:
            mixed_batch: Mixed batch
            targets_a: Original targets
            targets_b: Shuffled targets
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch.size(0)
        index = torch.randperm(batch_size)
        
        _, _, H, W = batch.shape
        
        # Generate random bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_batch = batch.clone()
        mixed_batch[:, :, bby1:bby2, bbx1:bbx2] = batch[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to actual area ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        targets_a, targets_b = targets, targets[index]
        
        return mixed_batch, targets_a, targets_b, lam


class EarthquakeDatasetV3(Dataset):
    """
    Enhanced dataset for earthquake prediction with advanced augmentation
    """
    
    def __init__(self,
                 dataset_dir: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 image_size: int = 224,
                 use_physics_aug: bool = True):
        """
        Initialize dataset
        
        Args:
            dataset_dir: Path to dataset directory
            split: Dataset split ('train', 'val', 'test')
            transform: Image transforms
            image_size: Target image size
            use_physics_aug: Use physics-preserving augmentation
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.use_physics_aug = use_physics_aug
        
        # Load metadata from final_metadata folder
        metadata_file = f"{split}_exp3.csv"
        metadata_path = self.dataset_dir / 'final_metadata' / metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        self.metadata = pd.read_csv(metadata_path)
        
        # No need for splits - already split by CSV file
        self.data = self.metadata.reset_index(drop=True)
        
        print(f"{split.upper()} dataset loaded: {len(self.data)} samples")
        
        # Create label mappings
        self._create_label_mappings()
        
        # Physics-preserving augmentation
        if use_physics_aug and split == 'train':
            self.physics_aug = PhysicsPreservingAugmentation()
        else:
            self.physics_aug = None
            
    def _create_azimuth_classes(self):
        """Create azimuth classes from magnitude data"""
        # For dataset experiment 3, we only have magnitude_class
        # Create dummy azimuth classes if not present
        if 'azimuth_class' not in self.data.columns:
            print("Warning: azimuth_class not found, creating default azimuth classes")
            # Create 9 azimuth classes based on row index distribution
            self.data['azimuth_class'] = self.data.index % 9
            self.data['azimuth_class'] = 'AZ' + self.data['azimuth_class'].astype(str)
            
    def _create_label_mappings(self):
        """Create label to index mappings"""
        # Magnitude classes (only active classes)
        magnitude_classes = sorted(self.data['magnitude_class'].unique())
        self.magnitude_to_idx = {cls: idx for idx, cls in enumerate(magnitude_classes)}
        self.idx_to_magnitude = {idx: cls for cls, idx in self.magnitude_to_idx.items()}
        
        # Create azimuth classes if not present
        self._create_azimuth_classes()
        
        # Azimuth classes
        azimuth_classes = sorted(self.data['azimuth_class'].unique())
        self.azimuth_to_idx = {cls: idx for idx, cls in enumerate(azimuth_classes)}
        self.idx_to_azimuth = {idx: cls for cls, idx in self.azimuth_to_idx.items()}
        
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get dataset item
        
        Args:
            idx: Sample index
            
        Returns:
            image: Preprocessed image tensor
            magnitude_label: Magnitude class index
            azimuth_label: Azimuth class index
        """
        # Get sample info
        sample = self.data.iloc[idx]
        
        # Load image using consolidation_path from dataset_experiment_3
        image_path = self.dataset_dir / sample['consolidation_path']
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert('RGB')
        
        # Resize if needed
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            image = transforms.ToTensor()(image)
        
        # Apply physics-preserving augmentation
        if self.physics_aug and self.split == 'train':
            if random.random() < 0.3:  # 30% chance
                image = self.physics_aug(image)
        
        # Get labels
        magnitude_label = self.magnitude_to_idx[sample['magnitude_class']]
        azimuth_label = self.azimuth_to_idx[sample['azimuth_class']]
        
        return image, magnitude_label, azimuth_label
    
    def get_targets(self, idx: int) -> Tuple[int, int]:
        """Get targets for sample (for weighted sampling)"""
        sample = self.data.iloc[idx]
        magnitude_label = self.magnitude_to_idx[sample['magnitude_class']]
        azimuth_label = self.azimuth_to_idx[sample['azimuth_class']]
        return magnitude_label, azimuth_label
    
    def get_class_counts(self) -> Dict[str, Dict[str, int]]:
        """Get class distribution"""
        magnitude_counts = self.data['magnitude_class'].value_counts().to_dict()
        azimuth_counts = self.data['azimuth_class'].value_counts().to_dict()
        
        return {
            'magnitude': magnitude_counts,
            'azimuth': azimuth_counts
        }


def create_transforms_v3(image_size: int = 224, 
                        augmentation_config: Optional[Dict] = None) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and validation transforms for V3
    
    Args:
        image_size: Target image size
        augmentation_config: Augmentation configuration
        
    Returns:
        train_transform: Training transforms
        val_transform: Validation transforms
    """
    if augmentation_config is None:
        augmentation_config = {
            'randaugment_magnitude': 9,
            'randaugment_num_ops': 2
        }
    
    # Training transforms with advanced augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
        
        # RandAugment - state-of-the-art automatic augmentation
        RandAugment(
            num_ops=augmentation_config['randaugment_num_ops'],
            magnitude=augmentation_config['randaugment_magnitude']
        ),
        
        # Additional transforms
        transforms.RandomHorizontalFlip(p=0.1),  # Low probability for spectrograms
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        
        # Normalization
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    MixUp loss function
    
    Args:
        criterion: Loss function
        pred: Predictions
        y_a: Original targets
        y_b: Mixed targets
        lam: Mixing coefficient
        
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_dataloaders_v3(dataset_dir: str,
                         batch_size: int = 32,
                         image_size: int = 224,
                         num_workers: int = 4,
                         augmentation_config: Optional[Dict] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for training, validation, and testing
    
    Args:
        dataset_dir: Path to dataset directory
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of worker processes
        augmentation_config: Augmentation configuration
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
    """
    # Create transforms
    train_transform, val_transform = create_transforms_v3(image_size, augmentation_config)
    
    # Create datasets
    train_dataset = EarthquakeDatasetV3(
        dataset_dir, split='train', transform=train_transform, image_size=image_size
    )
    val_dataset = EarthquakeDatasetV3(
        dataset_dir, split='val', transform=val_transform, image_size=image_size
    )
    test_dataset = EarthquakeDatasetV3(
        dataset_dir, split='test', transform=val_transform, image_size=image_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset
    print("🧪 Testing EarthquakeDatasetV3...")
    
    # Create dataset
    dataset = EarthquakeDatasetV3(
        'dataset_unified',
        split='train',
        image_size=224
    )
    
    print(f"📊 Dataset size: {len(dataset)}")
    print(f"📈 Class distribution: {dataset.get_class_counts()}")
    
    # Test sample
    image, mag_label, az_label = dataset[0]
    print(f"🖼️ Sample shape: {image.shape}")
    print(f"🎯 Labels: magnitude={mag_label}, azimuth={az_label}")
    
    # Test dataloader
    train_loader, val_loader, test_loader = create_dataloaders_v3(
        'dataset_unified', batch_size=8
    )
    
    print(f"📦 Dataloaders created:")
    print(f"Train: {len(train_loader)} batches")
    print(f"Val: {len(val_loader)} batches")
    print(f"Test: {len(test_loader)} batches")
    
    # Test batch
    for images, mag_labels, az_labels in train_loader:
        print(f"🔄 Batch shapes: {images.shape}, {mag_labels.shape}, {az_labels.shape}")
        break
    
    print("✅ Dataset test completed!")