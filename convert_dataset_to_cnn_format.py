"""
Convert existing dataset images to CNN format:
- Crop to remove axis/text
- Resize to 224x224
- Convert to RGB
- Much faster than regenerating from SSH server!
"""

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import shutil

def convert_image_to_cnn_format(input_path, output_path, target_size=224):
    """
    Convert spectrogram image to CNN format
    - Crop to remove axis and text
    - Resize to 224x224
    - Convert to RGB
    """
    # Load image
    img = Image.open(input_path)
    
    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Crop to remove axis and text
    # Typical matplotlib figure has margins, we need to crop them
    width, height = img.size
    
    # Crop margins (approximate values, adjust if needed)
    # Left: ~80px, Right: ~50px, Top: ~50px, Bottom: ~80px
    left_crop = int(width * 0.08)    # 8% from left
    right_crop = int(width * 0.95)   # 95% from left
    top_crop = int(height * 0.06)    # 6% from top
    bottom_crop = int(height * 0.90) # 90% from top
    
    img_cropped = img.crop((left_crop, top_crop, right_crop, bottom_crop))
    
    # Resize to target size (224x224) using high-quality Lanczos resampling
    img_resized = img_cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Save
    img_resized.save(output_path, 'PNG', quality=95)
    
    return img_resized.size, img_resized.mode


def main():
    """Main conversion function"""
    print("="*80)
    print("CONVERTING DATASET TO CNN FORMAT")
    print("="*80)
    print()
    print("This will:")
    print("  1. Crop images to remove axis and text")
    print("  2. Resize to 224x224 pixels")
    print("  3. Convert to RGB format")
    print()
    print("Original dataset will be backed up to: dataset_spectrogram_ssh_backup/")
    print("="*80)
    print()
    
    # Paths
    dataset_dir = 'dataset_spectrogram_ssh'
    backup_dir = 'dataset_spectrogram_ssh_backup'
    spectrograms_dir = os.path.join(dataset_dir, 'spectrograms')
    
    # Check if dataset exists
    if not os.path.exists(spectrograms_dir):
        print(f"ERROR: Dataset not found at {spectrograms_dir}")
        return
    
    # Backup original dataset
    print("Creating backup...")
    if not os.path.exists(backup_dir):
        shutil.copytree(dataset_dir, backup_dir)
        print(f"[OK] Backup created at {backup_dir}")
    else:
        print(f"[SKIP] Backup already exists at {backup_dir}")
    print()
    
    # Load metadata
    metadata_path = os.path.join(dataset_dir, 'metadata', 'dataset_metadata.csv')
    df = pd.read_csv(metadata_path)
    
    print(f"Found {len(df)} images to convert")
    print()
    
    # Convert images
    success_count = 0
    fail_count = 0
    
    print("Converting images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            filename = row['spectrogram_file']
            
            # Main spectrogram
            main_path = os.path.join(spectrograms_dir, filename)
            
            # By azimuth
            azimuth_path = os.path.join(
                spectrograms_dir, 'by_azimuth', 
                row['azimuth_class'], filename
            )
            
            # By magnitude
            magnitude_path = os.path.join(
                spectrograms_dir, 'by_magnitude',
                row['magnitude_class'], filename
            )
            
            # Convert all three locations
            paths_to_convert = [main_path, azimuth_path, magnitude_path]
            
            for path in paths_to_convert:
                if os.path.exists(path):
                    size, mode = convert_image_to_cnn_format(path, path, target_size=224)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n[ERROR] Failed to convert {filename}: {e}")
            fail_count += 1
    
    print()
    print("="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print(f"Successfully converted: {success_count}")
    print(f"Failed: {fail_count}")
    print()
    
    # Verify a sample
    print("Verifying sample image...")
    sample_file = df.iloc[0]['spectrogram_file']
    sample_path = os.path.join(spectrograms_dir, sample_file)
    
    img = Image.open(sample_path)
    print(f"  File: {sample_file}")
    print(f"  Size: {img.size}")
    print(f"  Mode: {img.mode}")
    
    if img.size == (224, 224) and img.mode == 'RGB':
        print("  [OK] Format is correct!")
    else:
        print("  [WARNING] Format may not be correct")
    
    print()
    print("Dataset is now ready for CNN training!")
    print("All images are 224x224 RGB without axis/text")
    print()
    print("Original dataset backed up to:", backup_dir)


if __name__ == '__main__':
    # Check if tqdm is installed
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm for progress bar...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'tqdm'])
        from tqdm import tqdm
    
    main()
