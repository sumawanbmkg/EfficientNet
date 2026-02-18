"""
Visualize Physics-Preserving Augmentation
Shows original vs augmented spectrograms
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from earthquake_dataset import PhysicsPreservingAugmentation

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor for visualization"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def visualize_augmentation(image_path, num_augmentations=8, augmentation_level='moderate'):
    """
    Visualize original image and multiple augmented versions
    WITHOUT axis and text (CNN format)
    
    Args:
        image_path: Path to spectrogram image
        num_augmentations: Number of augmented versions to show
        augmentation_level: 'light', 'moderate', 'aggressive'
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Base transform (no augmentation)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Get augmentation parameters
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
    
    # Create augmentation
    augmentation = PhysicsPreservingAugmentation(**aug_params)
    
    # Original image
    original_tensor = base_transform(image)
    
    # Create figure WITHOUT axis and text (CNN format)
    rows = (num_augmentations + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4*rows))
    axes = axes.flatten()
    
    # Remove all spacing
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, hspace=0.05, wspace=0.05)
    
    # Plot original (NO AXIS, NO TEXT)
    axes[0].imshow(original_tensor.permute(1, 2, 0))
    axes[0].set_title('Original', fontsize=10, pad=5)
    axes[0].axis('off')
    
    # Plot augmented versions (NO AXIS, NO TEXT)
    for i in range(1, num_augmentations + 1):
        augmented = augmentation(original_tensor.clone())
        axes[i].imshow(augmented.permute(1, 2, 0))
        axes[i].set_title(f'Aug #{i}', fontsize=10, pad=5)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_augmentations + 1, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Physics-Preserving Augmentation ({augmentation_level.upper()} level)', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    # Save
    output_dir = 'augmentation_examples'
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(image_path).replace('.png', f'_aug_{augmentation_level}.png')
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def compare_augmentation_levels(image_path):
    """
    Compare light, moderate, and aggressive augmentation
    WITHOUT axis and text (CNN format)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    original_tensor = base_transform(image)
    
    # Create augmentations for each level
    levels = ['light', 'moderate', 'aggressive']
    aug_configs = {
        'light': {
            'time_mask_param': 20, 'freq_mask_param': 10, 'noise_std': 0.03,
            'time_shift_max': 5, 'gain_range': 0.1,
            'p_time_mask': 0.3, 'p_freq_mask': 0.3, 'p_noise': 0.2,
            'p_time_shift': 0.2, 'p_gain': 0.2
        },
        'moderate': {
            'time_mask_param': 30, 'freq_mask_param': 15, 'noise_std': 0.05,
            'time_shift_max': 10, 'gain_range': 0.15,
            'p_time_mask': 0.5, 'p_freq_mask': 0.5, 'p_noise': 0.3,
            'p_time_shift': 0.3, 'p_gain': 0.3
        },
        'aggressive': {
            'time_mask_param': 40, 'freq_mask_param': 20, 'noise_std': 0.08,
            'time_shift_max': 15, 'gain_range': 0.2,
            'p_time_mask': 0.7, 'p_freq_mask': 0.7, 'p_noise': 0.5,
            'p_time_shift': 0.5, 'p_gain': 0.5
        }
    }
    
    # Create figure WITHOUT axis and text (CNN format)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    
    # Remove spacing
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, hspace=0.08, wspace=0.05)
    
    for row, level in enumerate(levels):
        # Original (NO AXIS, NO TEXT)
        axes[row, 0].imshow(original_tensor.permute(1, 2, 0))
        axes[row, 0].set_title(f'{level.upper()}\nOriginal', fontsize=9, fontweight='bold', pad=5)
        axes[row, 0].axis('off')
        
        # Create augmentation
        aug = PhysicsPreservingAugmentation(**aug_configs[level])
        
        # Show 3 augmented versions (NO AXIS, NO TEXT)
        for col in range(1, 4):
            augmented = aug(original_tensor.clone())
            axes[row, col].imshow(augmented.permute(1, 2, 0))
            axes[row, col].set_title(f'Aug #{col}', fontsize=9, pad=5)
            axes[row, col].axis('off')
    
    plt.suptitle('Comparison of Augmentation Levels (CNN Format)', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    # Save
    output_dir = 'augmentation_examples'
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(image_path).replace('.png', '_comparison.png')
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def main():
    """Main function"""
    print("="*80)
    print("PHYSICS-PRESERVING AUGMENTATION VISUALIZATION")
    print("="*80)
    
    # Find sample images from v21 dataset (correct format)
    dataset_dir = 'dataset_spectrogram_ssh_v21/spectrograms'
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset not found: {dataset_dir}")
        print("Please ensure v21 dataset exists")
        return
    
    # Get sample images
    sample_images = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('_3comp_spec.png'):
                sample_images.append(os.path.join(root, file))
                if len(sample_images) >= 3:
                    break
        if len(sample_images) >= 3:
            break
    
    if not sample_images:
        print("No spectrogram images found!")
        return
    
    print(f"\nFound {len(sample_images)} sample images")
    
    # Visualize augmentation for each sample
    for i, image_path in enumerate(sample_images[:2]):
        print(f"\n[{i+1}] Processing: {os.path.basename(image_path)}")
        
        # Show moderate augmentation
        print("  - Generating moderate augmentation examples...")
        visualize_augmentation(image_path, num_augmentations=8, augmentation_level='moderate')
    
    # Compare augmentation levels
    print(f"\n[3] Comparing augmentation levels...")
    compare_augmentation_levels(sample_images[0])
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"Output directory: augmentation_examples/")
    print("\nReview the augmented images to ensure:")
    print("  1. Frequency axis (vertical) is preserved")
    print("  2. Time relationships are maintained")
    print("  3. Pc3 band features (10-45 mHz) are still visible")
    print("  4. Augmentations look realistic (not too aggressive)")


if __name__ == '__main__':
    main()
