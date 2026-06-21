"""
Prediction Script for Vertebral Body Segmentation

This script performs inference on new images using trained models.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import nibabel as nib
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from models import get_model
from utils.visualization import plot_segmentation, plot_overlay


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict Vertebral Body Segmentation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resnet50_unet'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Path to save output prediction')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_image(image_path, image_size=256):
    """Load and preprocess image"""
    if image_path.endswith(('.nii', '.nii.gz')):
        # Load NIfTI image
        nii_img = nib.load(image_path)
        data = nii_img.get_fdata()
        
        # If 3D, take middle slice
        if len(data.shape) == 3:
            mid_slice = data.shape[2] // 2
            data = data[:, :, mid_slice]
        
        # Resize
        data = cv2.resize(data, (image_size, image_size))
        image = data.astype(np.float32)
        
    else:
        # Load standard image formats
        img = Image.open(image_path).convert('L')
        img = img.resize((image_size, image_size))
        image = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1]
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val > 0:
        image = (image - min_val) / (max_val - min_val)
    
    # Convert to tensor and add batch and channel dimensions
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
    return image_tensor, image


def save_prediction(prediction, output_path, original_shape=None):
    """Save prediction to file"""
    # Remove batch and channel dimensions
    pred_np = prediction.squeeze().cpu().numpy()
    
    # Resize to original shape if provided
    if original_shape is not None:
        pred_np = cv2.resize(pred_np, (original_shape[1], original_shape[0]))
    
    # Determine output format based on file extension
    if output_path.endswith(('.nii', '.nii.gz')):
        # Save as NIfTI
        pred_nii = nib.Nifti1Image(pred_np, affine=np.eye(4))
        nib.save(pred_nii, output_path)
    else:
        # Save as image (PNG, JPG, etc.)
        pred_img = (pred_np * 255).astype(np.uint8)
        Image.fromarray(pred_img).save(output_path)
    
    print(f"Prediction saved to {output_path}")


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating {args.model} model...")
    model = get_model(
        model_name=args.model,
        n_channels=config['model']['input_channels'],
        n_classes=config['model']['output_channels'],
        base_filters=config['model']['base_filters']
    )
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Load input image
    print(f"Loading image from {args.input}")
    image_tensor, image_np = load_image(args.input, config['data']['image_size'])
    image_tensor = image_tensor.to(device)
    
    # Predict
    print("Running inference...")
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output)
        prediction_binary = (prediction > 0.5).float()
    
    # Determine output path
    if args.output is None:
        # Create output filename based on input
        input_dir = os.path.dirname(args.input)
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        if args.input.endswith('.nii.gz'):
            input_name = input_name.replace('.nii', '')
        output_path = os.path.join(input_dir, f'{input_name}_prediction.png')
    else:
        output_path = args.output
    
    # Save prediction
    save_prediction(prediction_binary, output_path)
    
    # Create visualization
    if args.visualize:
        print("Creating visualization...")
        
        # Create output directory
        vis_dir = os.path.join(os.path.dirname(output_path), 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save segmentation plot
        seg_path = os.path.join(vis_dir, f'{os.path.splitext(os.path.basename(output_path))[0]}_segmentation.png')
        plot_segmentation(
            image_tensor[0].cpu(),
            None,  # No ground truth
            prediction_binary[0].cpu(),
            title='Vertebral Body Segmentation',
            save_path=seg_path
        )
        
        # Save overlay plot
        overlay_path = os.path.join(vis_dir, f'{os.path.splitext(os.path.basename(output_path))[0]}_overlay.png')
        plot_overlay(
            image_tensor[0].cpu(),
            prediction_binary[0].cpu(),
            None,
            title='Segmentation Overlay',
            save_path=overlay_path
        )
        
        print(f"Visualizations saved to {vis_dir}")
    
    print("\nPrediction completed!")


if __name__ == "__main__":
    main()
