"""
Quick Start Example for Vertebral Body Segmentation

This script demonstrates basic usage of the segmentation models.
"""

import torch
import numpy as np
from models import get_model
from utils.metrics import dice_coefficient, iou_score
import matplotlib.pyplot as plt


def example_1_create_models():
    """Example 1: Create and inspect models"""
    print("="*60)
    print("Example 1: Creating Models")
    print("="*60)
    
    # Create UNet model
    unet = get_model('unet', n_channels=1, n_classes=1, base_filters=64)
    print(f"\nUNet model created")
    print(f"Parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # Create ResNet50-UNet model
    resnet_unet = get_model('resnet50_unet', n_channels=1, n_classes=1, pretrained=False)
    print(f"\nResNet50-UNet model created")
    print(f"Parameters: {sum(p.numel() for p in resnet_unet.parameters()):,}")
    
    return unet, resnet_unet


def example_2_forward_pass():
    """Example 2: Run forward pass"""
    print("\n" + "="*60)
    print("Example 2: Forward Pass")
    print("="*60)
    
    # Create model
    model = get_model('unet', n_channels=1, n_classes=1)
    model.eval()
    
    # Create dummy input
    batch_size = 4
    image_size = 256
    x = torch.randn(batch_size, 1, image_size, image_size)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    # Apply sigmoid for binary segmentation
    prediction = torch.sigmoid(output)
    binary_prediction = (prediction > 0.5).float()
    
    print(f"Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
    print(f"Binary prediction unique values: {binary_prediction.unique().tolist()}")
    
    return output, binary_prediction


def example_3_calculate_metrics():
    """Example 3: Calculate segmentation metrics"""
    print("\n" + "="*60)
    print("Example 3: Calculating Metrics")
    print("="*60)
    
    # Create dummy prediction and target
    pred = torch.rand(1, 256, 256) > 0.3  # 30% threshold for more overlap
    target = torch.rand(1, 256, 256) > 0.3
    
    # Calculate metrics
    dice = dice_coefficient(pred.float(), target.float())
    iou = iou_score(pred.float(), target.float())
    
    print(f"\nDice Coefficient: {dice:.4f}")
    print(f"IoU Score: {iou:.4f}")
    
    # Perfect prediction example
    perfect_pred = target.clone()
    perfect_dice = dice_coefficient(perfect_pred.float(), target.float())
    perfect_iou = iou_score(perfect_pred.float(), target.float())
    
    print(f"\nPerfect prediction:")
    print(f"Dice Coefficient: {perfect_dice:.4f}")
    print(f"IoU Score: {perfect_iou:.4f}")


def example_4_training_loop():
    """Example 4: Minimal training loop structure"""
    print("\n" + "="*60)
    print("Example 4: Training Loop Structure")
    print("="*60)
    
    print("""
# Pseudo-code for training loop:

model = get_model('unet', n_channels=1, n_classes=1)
criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            outputs = model(images)
            # Calculate metrics...
    """)


def example_5_inference():
    """Example 5: Inference on new image"""
    print("\n" + "="*60)
    print("Example 5: Inference Example")
    print("="*60)
    
    print("""
# Pseudo-code for inference:

# Load model
model = get_model('unet', n_channels=1, n_classes=1)
checkpoint = torch.load('checkpoints/unet_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess image
image = load_image('path/to/image.nii.gz')
image = preprocess(image)  # Resize, normalize

# Predict
with torch.no_grad():
    output = model(image)
    prediction = torch.sigmoid(output)
    binary_mask = (prediction > 0.5).float()

# Save or visualize
save_prediction(binary_mask, 'output.png')
    """)


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print(" "*20 + "VERTEBRAL BODY SEGMENTATION")
    print(" "*25 + "Quick Start Examples")
    print("="*80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    try:
        example_1_create_models()
        example_2_forward_pass()
        example_3_calculate_metrics()
        example_4_training_loop()
        example_5_inference()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        
        print("\nNext steps:")
        print("1. Prepare your dataset using: python prepare_data.py")
        print("2. Train a model using: python train.py --model unet --data_dir ./data")
        print("3. Evaluate using: python evaluate.py --model unet --checkpoint ./checkpoints/unet_best.pth")
        print("4. Run inference using: python predict.py --model unet --checkpoint ./checkpoints/unet_best.pth --input image.nii.gz")
        print("\nFor more information, see README.md and CONTRIBUTING.md")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
