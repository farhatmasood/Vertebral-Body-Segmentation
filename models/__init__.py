"""
Model Module for Vertebral Body Segmentation

This module provides access to all implemented models.
"""

from .unet import UNet, get_unet
from .resnet50_unet import ResNet50UNet, get_resnet50_unet

__all__ = [
    'UNet',
    'ResNet50UNet',
    'get_unet',
    'get_resnet50_unet',
    'get_model'
]


def get_model(model_name, n_channels=1, n_classes=1, **kwargs):
    """
    Factory function to get model by name
    
    Args:
        model_name (str): Name of the model ('unet' or 'resnet50_unet')
        n_channels (int): Number of input channels
        n_classes (int): Number of output classes
        **kwargs: Additional model-specific arguments
    
    Returns:
        Model instance
    
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = model_name.lower()
    
    if model_name == 'unet':
        return get_unet(n_channels=n_channels, n_classes=n_classes, **kwargs)
    elif model_name == 'resnet50_unet':
        return get_resnet50_unet(n_channels=n_channels, n_classes=n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: unet, resnet50_unet")
