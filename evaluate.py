"""
Evaluation Script for Vertebral Body Segmentation

This script evaluates trained models on test data.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import json
from tqdm import tqdm

from models import get_model
from utils.dataset import get_data_loaders
from utils.metrics import evaluate_model, dice_coefficient, iou_score, pixel_accuracy
from utils.visualization import visualize_batch, plot_segmentation


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Vertebral Body Segmentation Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resnet50_unet'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to data directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--save_predictions', action='store_true', help='Save prediction visualizations')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
    
    # Create data loaders
    print("Loading data...")
    data_loaders = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=1,  # Use batch size 1 for evaluation
        num_workers=config['train']['num_workers'],
        image_size=config['data']['image_size'],
        augmentation=False,  # No augmentation for evaluation
        normalize=config['data']['normalize']
    )
    
    if args.split not in data_loaders:
        raise ValueError(f"{args.split} dataset not found")
    
    test_loader = data_loaders[args.split]
    print(f"{args.split.capitalize()} samples: {len(test_loader.dataset)}")
    
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
    print(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Evaluate model
    print("\nEvaluating model...")
    model.eval()
    
    all_dice = []
    all_iou = []
    all_acc = []
    all_predictions = []
    all_images = []
    all_masks = []
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)
            masks = masks.to(device)
            
            # Predict
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds_binary = (preds > 0.5).float()
            
            # Calculate metrics
            dice = dice_coefficient(preds_binary[0], masks[0])
            iou = iou_score(preds_binary[0], masks[0])
            acc = pixel_accuracy(preds_binary[0], masks[0])
            
            all_dice.append(dice)
            all_iou.append(iou)
            all_acc.append(acc)
            
            # Store for visualization
            if args.save_predictions and idx < 10:  # Save first 10 samples
                all_predictions.append(preds_binary[0].cpu())
                all_images.append(images[0].cpu())
                all_masks.append(masks[0].cpu())
    
    # Calculate statistics
    results = {
        'dice_mean': float(np.mean(all_dice)),
        'dice_std': float(np.std(all_dice)),
        'dice_min': float(np.min(all_dice)),
        'dice_max': float(np.max(all_dice)),
        'iou_mean': float(np.mean(all_iou)),
        'iou_std': float(np.std(all_iou)),
        'iou_min': float(np.min(all_iou)),
        'iou_max': float(np.max(all_iou)),
        'accuracy_mean': float(np.mean(all_acc)),
        'accuracy_std': float(np.std(all_acc)),
        'accuracy_min': float(np.min(all_acc)),
        'accuracy_max': float(np.max(all_acc)),
        'num_samples': len(all_dice)
    }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"\nDice Coefficient:")
    print(f"  Mean: {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
    print(f"  Min:  {results['dice_min']:.4f}")
    print(f"  Max:  {results['dice_max']:.4f}")
    print(f"\nIoU Score:")
    print(f"  Mean: {results['iou_mean']:.4f} ± {results['iou_std']:.4f}")
    print(f"  Min:  {results['iou_min']:.4f}")
    print(f"  Max:  {results['iou_max']:.4f}")
    print(f"\nPixel Accuracy:")
    print(f"  Mean: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"  Min:  {results['accuracy_min']:.4f}")
    print(f"  Max:  {results['accuracy_max']:.4f}")
    print("="*60)
    
    # Save results
    results_dir = config['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f'evaluation_results_{args.model}_{args.split}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_file}")
    
    # Save visualizations
    if args.save_predictions and len(all_predictions) > 0:
        print("\nSaving prediction visualizations...")
        vis_dir = os.path.join(results_dir, 'predictions')
        os.makedirs(vis_dir, exist_ok=True)
        
        for i in range(len(all_predictions)):
            save_path = os.path.join(vis_dir, f'prediction_{i+1}.png')
            plot_segmentation(
                all_images[i],
                all_masks[i],
                all_predictions[i],
                title=f'Sample {i+1} - Dice: {all_dice[i]:.4f}',
                save_path=save_path
            )
        
        print(f"Visualizations saved to {vis_dir}")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
