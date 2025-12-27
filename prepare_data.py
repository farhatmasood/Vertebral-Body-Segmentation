"""
Example script demonstrating how to prepare data for vertebral body segmentation.

This script helps organize the dataset into the expected directory structure.
"""

import os
import shutil
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Source directory containing raw data')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for organized data')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Fraction of data for training')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of data for validation')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='Fraction of data for testing')
    
    return parser.parse_args()


def organize_dataset(source_dir, output_dir, train_split=0.7, val_split=0.15, test_split=0.15):
    """
    Organize dataset into train/val/test splits
    
    Expected source structure:
    source_dir/
    ├── images/
    │   ├── image_001.nii.gz
    │   ├── image_002.nii.gz
    │   └── ...
    └── masks/
        ├── mask_001.nii.gz
        ├── mask_002.nii.gz
        └── ...
    
    Output structure:
    output_dir/
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
    """
    
    # Validate split ratios
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)
    
    # Get list of images
    image_dir = os.path.join(source_dir, 'images')
    mask_dir = os.path.join(source_dir, 'masks')
    
    if not os.path.exists(image_dir):
        raise ValueError(f"Images directory not found: {image_dir}")
    if not os.path.exists(mask_dir):
        raise ValueError(f"Masks directory not found: {mask_dir}")
    
    images = sorted([f for f in os.listdir(image_dir) 
                    if f.endswith(('.nii', '.nii.gz', '.png', '.jpg', '.jpeg'))])
    
    print(f"Found {len(images)} images")
    
    # Calculate split indices
    n_train = int(len(images) * train_split)
    n_val = int(len(images) * val_split)
    n_test = len(images) - n_train - n_val
    
    print(f"Split: {n_train} train, {n_val} val, {n_test} test")
    
    # Shuffle images (with fixed seed for reproducibility)
    import random
    random.seed(42)
    random.shuffle(images)
    
    # Split images
    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]
    
    # Copy files to respective directories
    for split_name, split_images in [('train', train_images), 
                                      ('val', val_images), 
                                      ('test', test_images)]:
        print(f"\nProcessing {split_name} split...")
        
        for img_name in split_images:
            # Copy image
            src_img = os.path.join(image_dir, img_name)
            dst_img = os.path.join(output_dir, split_name, 'images', img_name)
            shutil.copy2(src_img, dst_img)
            
            # Copy corresponding mask
            mask_name = img_name.replace('image', 'mask')
            src_mask = os.path.join(mask_dir, mask_name)
            dst_mask = os.path.join(output_dir, split_name, 'masks', mask_name)
            
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
            else:
                print(f"Warning: Mask not found for {img_name}")
    
    print(f"\nDataset organized successfully in {output_dir}")


def create_dummy_dataset(output_dir, n_samples=10):
    """
    Create a dummy dataset for testing purposes
    
    This is useful if you don't have the actual dataset yet.
    """
    import numpy as np
    from PIL import Image
    
    print("Creating dummy dataset for testing...")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)
    
    # Create dummy images and masks
    splits = {
        'train': int(n_samples * 0.7),
        'val': int(n_samples * 0.15),
        'test': n_samples - int(n_samples * 0.7) - int(n_samples * 0.15)
    }
    
    sample_id = 1
    for split, count in splits.items():
        for i in range(count):
            # Create dummy grayscale image (256x256)
            img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            img_pil = Image.fromarray(img)
            img_path = os.path.join(output_dir, split, 'images', f'image_{sample_id:03d}.png')
            img_pil.save(img_path)
            
            # Create dummy binary mask
            mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
            mask_pil = Image.fromarray(mask)
            mask_path = os.path.join(output_dir, split, 'masks', f'mask_{sample_id:03d}.png')
            mask_pil.save(mask_path)
            
            sample_id += 1
    
    print(f"Created {n_samples} dummy samples in {output_dir}")
    print("Note: This is synthetic data for testing only!")


def main():
    args = parse_args()
    
    if os.path.exists(args.source_dir):
        # Organize existing dataset
        organize_dataset(
            args.source_dir,
            args.output_dir,
            args.train_split,
            args.val_split,
            args.test_split
        )
    else:
        print(f"Source directory {args.source_dir} not found.")
        response = input("Create dummy dataset for testing? (y/n): ")
        if response.lower() == 'y':
            create_dummy_dataset(args.output_dir, n_samples=30)
        else:
            print("Please provide a valid source directory.")


if __name__ == "__main__":
    main()
