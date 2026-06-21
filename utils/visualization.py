"""
Visualization Utilities for Vertebral Body Segmentation

This module provides functions for visualizing segmentation results.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from matplotlib.colors import ListedColormap


def plot_segmentation(image, mask, prediction=None, title=None, save_path=None):
    """
    Plot image, ground truth mask, and prediction side by side
    
    Args:
        image (numpy.ndarray or torch.Tensor): Input image
        mask (numpy.ndarray or torch.Tensor): Ground truth mask
        prediction (numpy.ndarray or torch.Tensor, optional): Predicted mask
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
    """
    # Convert tensors to numpy
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if prediction is not None and torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    
    # Remove channel dimension if present
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image[0]
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if prediction is not None and len(prediction.shape) == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    
    # Create figure
    n_plots = 3 if prediction is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Plot image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Plot ground truth mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot prediction
    if prediction is not None:
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_overlay(image, mask, prediction=None, alpha=0.5, title=None, save_path=None):
    """
    Plot segmentation overlay on image
    
    Args:
        image (numpy.ndarray or torch.Tensor): Input image
        mask (numpy.ndarray or torch.Tensor): Ground truth mask
        prediction (numpy.ndarray or torch.Tensor, optional): Predicted mask
        alpha (float): Transparency of overlay
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
    """
    # Convert tensors to numpy
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if prediction is not None and torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    
    # Remove channel dimension if present
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image[0]
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if prediction is not None and len(prediction.shape) == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    
    # Normalize image to 0-255
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Create RGB image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create figure
    n_plots = 2 if prediction is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 7))
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot ground truth overlay
    overlay_gt = image_rgb.copy()
    overlay_gt[mask > 0.5] = [0, 255, 0]  # Green for ground truth
    blended_gt = cv2.addWeighted(image_rgb, 1 - alpha, overlay_gt, alpha, 0)
    
    axes[0].imshow(blended_gt)
    axes[0].set_title('Ground Truth Overlay')
    axes[0].axis('off')
    
    # Plot prediction overlay
    if prediction is not None:
        overlay_pred = image_rgb.copy()
        overlay_pred[prediction > 0.5] = [255, 0, 0]  # Red for prediction
        blended_pred = cv2.addWeighted(image_rgb, 1 - alpha, overlay_pred, alpha, 0)
        
        axes[1].imshow(blended_pred)
        axes[1].set_title('Prediction Overlay')
        axes[1].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss and metrics)
    
    Args:
        history (dict): Dictionary containing training history
        save_path (str, optional): Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot training and validation loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Plot Dice coefficient
    if 'train_dice' in history and 'val_dice' in history:
        axes[0, 1].plot(history['train_dice'], label='Train Dice')
        axes[0, 1].plot(history['val_dice'], label='Val Dice')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].set_title('Dice Coefficient')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot IoU
    if 'train_iou' in history and 'val_iou' in history:
        axes[1, 0].plot(history['train_iou'], label='Train IoU')
        axes[1, 0].plot(history['val_iou'], label='Val IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].set_title('Intersection over Union')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot Accuracy
    if 'train_acc' in history and 'val_acc' in history:
        axes[1, 1].plot(history['train_acc'], label='Train Accuracy')
        axes[1, 1].plot(history['val_acc'], label='Val Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Pixel Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    return fig


def visualize_batch(images, masks, predictions=None, n_samples=4, save_path=None):
    """
    Visualize a batch of images, masks, and predictions
    
    Args:
        images (torch.Tensor): Batch of images
        masks (torch.Tensor): Batch of ground truth masks
        predictions (torch.Tensor, optional): Batch of predictions
        n_samples (int): Number of samples to visualize
        save_path (str, optional): Path to save the figure
    """
    n_samples = min(n_samples, len(images))
    n_cols = 3 if predictions is not None else 2
    
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(5 * n_cols, 5 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Convert to numpy
        image = images[i].cpu().numpy()
        mask = masks[i].cpu().numpy()
        
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image[0]
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = mask[0]
        
        # Plot image
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f'Image {i+1}')
        axes[i, 0].axis('off')
        
        # Plot mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        # Plot prediction
        if predictions is not None:
            pred = predictions[i].cpu().numpy()
            if len(pred.shape) == 3 and pred.shape[0] == 1:
                pred = pred[0]
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title(f'Prediction {i+1}')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Batch visualization saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test visualization
    print("Visualization module loaded successfully")
