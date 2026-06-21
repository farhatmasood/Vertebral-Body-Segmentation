# Quick Reference Guide

A quick reference for common commands and workflows.

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Common Issues](#common-issues)

## Installation

### Basic Installation
```bash
# Clone repository
git clone https://github.com/farhatmasood/Vertebral-Body-Segmentation.git
cd Vertebral-Body-Segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Installation
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1+
pip install torch torchvision

# Then install other requirements
pip install -r requirements.txt
```

## Data Preparation

### Download Dataset
```bash
# Download from: https://data.mendeley.com/datasets/k3b363f3vz/2
# Extract to ./raw_data
```

### Organize Dataset
```bash
# Organize into train/val/test splits
python prepare_data.py --source_dir ./raw_data --output_dir ./data

# With custom split ratios
python prepare_data.py --source_dir ./raw_data --output_dir ./data \
    --train_split 0.7 --val_split 0.15 --test_split 0.15
```

### Create Dummy Data (for testing)
```bash
# Automatically creates dummy dataset if source not found
python prepare_data.py --source_dir ./nonexistent
# Answer 'y' when prompted
```

## Training

### Train UNet
```bash
# Basic training
python train.py --model unet --data_dir ./data

# With custom parameters
python train.py --model unet --data_dir ./data \
    --epochs 100 --batch_size 8 --lr 0.0001

# Resume from checkpoint
python train.py --model unet --data_dir ./data \
    --checkpoint ./checkpoints/unet_epoch_50.pth
```

### Train ResNet50-UNet
```bash
# Basic training
python train.py --model resnet50_unet --data_dir ./data

# With custom parameters
python train.py --model resnet50_unet --data_dir ./data \
    --epochs 100 --batch_size 8 --lr 0.0001
```

### Configuration File
```bash
# Edit config.yaml to customize:
# - Model architecture
# - Training hyperparameters
# - Data augmentation
# - Loss function
# - Optimizer settings
# - Paths

# Train with config
python train.py --config ./config.yaml --model unet
```

## Evaluation

### Evaluate on Test Set
```bash
# Basic evaluation
python evaluate.py --model unet \
    --checkpoint ./checkpoints/unet_best.pth \
    --data_dir ./data

# Save predictions
python evaluate.py --model unet \
    --checkpoint ./checkpoints/unet_best.pth \
    --data_dir ./data --save_predictions

# Evaluate on validation set
python evaluate.py --model unet \
    --checkpoint ./checkpoints/unet_best.pth \
    --data_dir ./data --split val
```

### View Results
```bash
# Results are saved to:
# - ./results/evaluation_results_[model]_[split].json
# - ./results/predictions/ (if --save_predictions used)
```

## Inference

### Predict on Single Image
```bash
# Basic prediction
python predict.py --model unet \
    --checkpoint ./checkpoints/unet_best.pth \
    --input ./image.nii.gz

# With custom output path
python predict.py --model unet \
    --checkpoint ./checkpoints/unet_best.pth \
    --input ./image.nii.gz \
    --output ./prediction.png

# With visualization
python predict.py --model unet \
    --checkpoint ./checkpoints/unet_best.pth \
    --input ./image.nii.gz \
    --visualize
```

### Supported Input Formats
- NIfTI: `.nii`, `.nii.gz`
- Images: `.png`, `.jpg`, `.jpeg`

## Monitoring Training

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir ./logs

# View in browser at: http://localhost:6006
```

### View Training Progress
Training outputs:
- Loss curves
- Dice coefficient
- IoU score
- Pixel accuracy
- Learning rate

## Common Issues

### Out of Memory
```bash
# Reduce batch size in config.yaml
batch_size: 4  # or 2, or 1

# Or reduce image size
image_size: 128  # instead of 256
```

### Slow Training
```bash
# Use GPU
--device cuda

# Increase batch size (if memory allows)
--batch_size 16

# Reduce image size
# Edit config.yaml: image_size: 128
```

### Data Loading Errors
```bash
# Check data directory structure
ls -R ./data

# Should have:
# data/train/images/
# data/train/masks/
# data/val/images/
# data/val/masks/
# data/test/images/
# data/test/masks/
```

## File Locations

### Checkpoints
```
./checkpoints/
├── unet_best.pth              # Best UNet model
├── unet_epoch_N.pth           # UNet epoch checkpoints
├── resnet50_unet_best.pth     # Best ResNet50-UNet model
└── resnet50_unet_epoch_N.pth  # ResNet50-UNet checkpoints
```

### Results
```
./results/
├── RESULTS.md                           # Performance documentation
├── training_history.png                 # Training curves
├── evaluation_results_[model].json      # Evaluation metrics
└── predictions/                         # Prediction visualizations
```

### Logs
```
./logs/
└── [timestamp]/                # TensorBoard logs
```

## Quick Tips

### Best Practices
- Start with small dataset to verify pipeline
- Monitor training with TensorBoard
- Save checkpoints regularly
- Use validation set to prevent overfitting
- Try different learning rates (0.0001, 0.00001)
- Use data augmentation for better generalization

### Performance Tuning
- **UNet**: Faster, good baseline (Dice ~0.92)
- **ResNet50-UNet**: Better accuracy (Dice ~0.94)
- Combine both in ensemble for best results

### Recommended Workflow
1. Prepare data → `prepare_data.py`
2. Train UNet → `train.py --model unet`
3. Evaluate → `evaluate.py --model unet`
4. Train ResNet50-UNet → `train.py --model resnet50_unet`
5. Evaluate → `evaluate.py --model resnet50_unet`
6. Compare results
7. Use best model for inference

## Getting Help

### Documentation
- [README.md](README.md) - Overview and usage
- [INSTALL.md](INSTALL.md) - Installation guide
- [FAQ.md](FAQ.md) - Common questions
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

### Support
- GitHub Issues: Report bugs or ask questions
- Paper: [Link](https://www.sciencedirect.com/science/article/abs/pii/S1746809421008272)
- Dataset: [Link](https://data.mendeley.com/datasets/k3b363f3vz/2)

---

For detailed documentation, see [README.md](README.md).
