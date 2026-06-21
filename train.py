"""
Training Script for Vertebral Body Segmentation

This script trains UNet or ResNet50-UNet models for vertebral body segmentation.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models import get_model
from utils.dataset import get_data_loaders
from utils.metrics import CombinedLoss, DiceLoss, dice_coefficient, iou_score, pixel_accuracy
from utils.visualization import plot_training_history, visualize_batch


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Vertebral Body Segmentation Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resnet50_unet'],
                        help='Model architecture')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    dice_scores = []
    iou_scores = []
    accuracies = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > 0.5).float()
            
            for pred, mask in zip(preds_binary, masks):
                dice_scores.append(dice_coefficient(pred, mask))
                iou_scores.append(iou_score(pred, mask))
                accuracies.append(pixel_accuracy(pred, mask))
        
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'dice': np.mean(dice_scores[-len(images):]),
        })
    
    # Calculate average metrics
    avg_loss = running_loss / len(train_loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_acc = np.mean(accuracies)
    
    return avg_loss, avg_dice, avg_iou, avg_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    
    running_loss = 0.0
    dice_scores = []
    iou_scores = []
    accuracies = []
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > 0.5).float()
            
            for pred, mask in zip(preds_binary, masks):
                dice_scores.append(dice_coefficient(pred, mask))
                iou_scores.append(iou_score(pred, mask))
                accuracies.append(pixel_accuracy(pred, mask))
            
            running_loss += loss.item()
    
    avg_loss = running_loss / len(val_loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_acc = np.mean(accuracies)
    
    return avg_loss, avg_dice, avg_iou, avg_acc


def save_checkpoint(model, optimizer, epoch, best_dice, checkpoint_dir, model_name, is_best=False):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_dice': best_dice,
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")
    
    return checkpoint_path


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['train']['learning_rate'] = args.lr
    if args.device is not None:
        config['device'] = args.device
    
    config['data']['data_dir'] = args.data_dir
    config['model']['architecture'] = args.model
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    data_loaders = get_data_loaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        image_size=config['data']['image_size'],
        augmentation=config['data']['augmentation']['enabled'],
        normalize=config['data']['normalize']
    )
    
    if 'train' not in data_loaders or 'val' not in data_loaders:
        raise ValueError("Training and validation datasets are required")
    
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"Creating {args.model} model...")
    model = get_model(
        model_name=config['model']['architecture'],
        n_channels=config['model']['input_channels'],
        n_classes=config['model']['output_channels'],
        base_filters=config['model']['base_filters']
    )
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function
    if config['loss']['type'] == 'combined':
        criterion = CombinedLoss(
            bce_weight=config['loss']['bce_weight'],
            dice_weight=config['loss']['dice_weight']
        )
    elif config['loss']['type'] == 'dice':
        criterion = DiceLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Define optimizer
    if config['optimizer']['type'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['train']['learning_rate'],
            weight_decay=config['train']['weight_decay'],
            betas=config['optimizer']['betas']
        )
    elif config['optimizer']['type'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['train']['learning_rate'],
            weight_decay=config['train']['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['train']['learning_rate'],
            momentum=config['optimizer']['momentum'],
            weight_decay=config['train']['weight_decay']
        )
    
    # Define learning rate scheduler
    if config['scheduler']['type'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=config['scheduler']['patience'],
            factor=config['scheduler']['factor'],
            min_lr=config['scheduler']['min_lr'],
            verbose=True
        )
    elif config['scheduler']['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['train']['epochs']
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    
    # Load checkpoint if specified
    start_epoch = 0
    best_dice = 0.0
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        print(f"Resuming from epoch {start_epoch}, best dice: {best_dice:.4f}")
    
    # Create directories
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=config['paths']['log_dir'])
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_iou': [],
        'val_iou': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, config['train']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['train']['epochs']}")
        
        # Train
        train_loss, train_dice, train_iou, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_dice, val_iou, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        if config['scheduler']['type'] == 'reduce_on_plateau':
            scheduler.step(val_dice)
        else:
            scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('IoU/train', train_iou, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}, Acc: {val_acc:.4f}")
        
        # Save checkpoint
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
        
        if (epoch + 1) % config['train']['save_interval'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, best_dice,
                config['paths']['checkpoint_dir'],
                config['model']['architecture'],
                is_best=is_best
            )
    
    # Save final training history plot
    plot_path = os.path.join(config['paths']['results_dir'], 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    writer.close()
    print("\nTraining completed!")
    print(f"Best validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
