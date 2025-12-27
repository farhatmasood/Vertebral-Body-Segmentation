"""
Dataset and Data Loading Utilities for Vertebral Body Segmentation

This module provides dataset classes and data loading utilities.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from PIL import Image
import cv2
from torchvision import transforms
import SimpleITK as sitk


class VertebralDataset(Dataset):
    """
    Dataset class for vertebral body segmentation
    
    Args:
        image_dir (str): Directory containing images
        mask_dir (str): Directory containing masks
        transform (callable, optional): Optional transform to be applied
        image_size (int): Size to resize images to
        normalize (bool): Whether to normalize images
    """
    
    def __init__(self, image_dir, mask_dir, transform=None, image_size=256, normalize=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.normalize = normalize
        
        # Get all image filenames
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith(('.nii', '.nii.gz', '.png', '.jpg', '.jpeg'))])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Determine file type and load accordingly
        if img_name.endswith(('.nii', '.nii.gz')):
            image = self._load_nifti(img_path)
        else:
            image = self._load_image(img_path)
        
        # Load mask
        mask_name = img_name.replace('image', 'mask')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            if mask_name.endswith(('.nii', '.nii.gz')):
                mask = self._load_nifti(mask_path)
            else:
                mask = self._load_image(mask_path)
        else:
            # If mask doesn't exist, create empty mask
            mask = np.zeros_like(image)
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        # Add channel dimension if needed
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        # Normalize image
        if self.normalize:
            image = self._normalize(image)
        
        # Ensure mask is binary
        mask = (mask > 0.5).float()
        
        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask
    
    def _load_nifti(self, path):
        """Load NIfTI image and extract a 2D slice"""
        nii_img = nib.load(path)
        data = nii_img.get_fdata()
        
        # If 3D, take middle slice
        if len(data.shape) == 3:
            mid_slice = data.shape[2] // 2
            data = data[:, :, mid_slice]
        
        # Resize
        data = cv2.resize(data, (self.image_size, self.image_size))
        
        return data.astype(np.float32)
    
    def _load_image(self, path):
        """Load standard image formats"""
        img = Image.open(path).convert('L')  # Convert to grayscale
        img = img.resize((self.image_size, self.image_size))
        return np.array(img, dtype=np.float32)
    
    def _normalize(self, tensor):
        """Normalize tensor to [0, 1]"""
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 0:
            tensor = (tensor - min_val) / (max_val - min_val)
        return tensor


class DataAugmentation:
    """
    Data augmentation for medical images
    
    Args:
        rotation_range (float): Range of rotation in degrees
        flip_horizontal (bool): Whether to apply horizontal flipping
        flip_vertical (bool): Whether to apply vertical flipping
        brightness_range (float): Range for brightness adjustment
        contrast_range (float): Range for contrast adjustment
    """
    
    def __init__(self, rotation_range=15, flip_horizontal=True, flip_vertical=False,
                 brightness_range=0.2, contrast_range=0.2):
        self.rotation_range = rotation_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, image, mask):
        # Random rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)
        
        # Random horizontal flip
        if self.flip_horizontal and np.random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        # Random vertical flip
        if self.flip_vertical and np.random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        
        # Random brightness
        if self.brightness_range > 0:
            brightness_factor = 1 + np.random.uniform(-self.brightness_range, self.brightness_range)
            image = image * brightness_factor
            image = torch.clamp(image, 0, 1)
        
        # Random contrast
        if self.contrast_range > 0:
            contrast_factor = 1 + np.random.uniform(-self.contrast_range, self.contrast_range)
            mean = image.mean()
            image = (image - mean) * contrast_factor + mean
            image = torch.clamp(image, 0, 1)
        
        return image, mask


def get_data_loaders(data_dir, batch_size=8, num_workers=4, image_size=256, 
                     augmentation=True, normalize=True):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        data_dir (str): Root directory containing train/val/test folders
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        image_size (int): Size to resize images to
        augmentation (bool): Whether to apply data augmentation
        normalize (bool): Whether to normalize images
    
    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' data loaders
    """
    from torch.utils.data import DataLoader
    
    data_loaders = {}
    
    # Define augmentation for training
    train_transform = DataAugmentation() if augmentation else None
    
    # Create datasets
    for split in ['train', 'val', 'test']:
        image_dir = os.path.join(data_dir, split, 'images')
        mask_dir = os.path.join(data_dir, split, 'masks')
        
        if not os.path.exists(image_dir):
            print(f"Warning: {image_dir} does not exist, skipping {split} dataset")
            continue
        
        transform = train_transform if split == 'train' else None
        
        dataset = VertebralDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            transform=transform,
            image_size=image_size,
            normalize=normalize
        )
        
        shuffle = (split == 'train')
        
        data_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return data_loaders


if __name__ == "__main__":
    # Test dataset loading
    print("Dataset module loaded successfully")
