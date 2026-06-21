"""
Evaluation Metrics for Segmentation Tasks

This module provides common metrics for evaluating segmentation performance.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient (F1 score for segmentation)
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: Dice coefficient
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) / Jaccard Index
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: IoU score
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def pixel_accuracy(pred, target):
    """
    Calculate pixel-wise accuracy
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask
        target (torch.Tensor): Ground truth mask
    
    Returns:
        float: Pixel accuracy
    """
    pred = pred.flatten()
    target = target.flatten()
    
    correct = (pred == target).sum()
    total = target.numel()
    accuracy = correct.float() / total
    
    return accuracy.item()


def precision_score(pred, target, smooth=1e-6):
    """
    Calculate precision (positive predictive value)
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: Precision score
    """
    pred = pred.flatten()
    target = target.flatten()
    
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    
    return precision.item()


def recall_score(pred, target, smooth=1e-6):
    """
    Calculate recall (sensitivity, true positive rate)
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: Recall score
    """
    pred = pred.flatten()
    target = target.flatten()
    
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    recall = (true_positive + smooth) / (actual_positive + smooth)
    
    return recall.item()


def specificity_score(pred, target, smooth=1e-6):
    """
    Calculate specificity (true negative rate)
    
    Args:
        pred (torch.Tensor): Predicted segmentation mask
        target (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: Specificity score
    """
    pred = pred.flatten()
    target = target.flatten()
    
    true_negative = ((1 - pred) * (1 - target)).sum()
    actual_negative = (1 - target).sum()
    specificity = (true_negative + smooth) / (actual_negative + smooth)
    
    return specificity.item()


class DiceLoss(torch.nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.flatten()
        target = target.flatten()
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(torch.nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss
    
    Args:
        bce_weight (float): Weight for BCE loss
        dice_weight (float): Weight for Dice loss
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def evaluate_model(model, data_loader, device, threshold=0.5):
    """
    Evaluate model on a dataset
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        threshold (float): Threshold for binary prediction
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    
    metrics = {
        'dice': [],
        'iou': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'specificity': []
    }
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > threshold).float()
            
            # Calculate metrics for each sample in batch
            for pred, mask in zip(preds, masks):
                metrics['dice'].append(dice_coefficient(pred, mask))
                metrics['iou'].append(iou_score(pred, mask))
                metrics['accuracy'].append(pixel_accuracy(pred, mask))
                metrics['precision'].append(precision_score(pred, mask))
                metrics['recall'].append(recall_score(pred, mask))
                metrics['specificity'].append(specificity_score(pred, mask))
    
    # Calculate mean metrics
    mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
    std_metrics = {k + '_std': np.std(v) for k, v in metrics.items()}
    
    return {**mean_metrics, **std_metrics}


if __name__ == "__main__":
    # Test metrics
    pred = torch.rand(4, 1, 256, 256) > 0.5
    target = torch.rand(4, 1, 256, 256) > 0.5
    
    print(f"Dice: {dice_coefficient(pred[0], target[0]):.4f}")
    print(f"IoU: {iou_score(pred[0], target[0]):.4f}")
    print(f"Accuracy: {pixel_accuracy(pred[0], target[0]):.4f}")
