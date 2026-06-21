# Frequently Asked Questions (FAQ)

## General Questions

### What is this project about?

This project implements deep learning models (UNet and ResNet50-UNet) for automatic segmentation of vertebral bodies in medical CT images. It's based on research published in Biomedical Signal Processing and Control.

### What can I use this for?

- Automated vertebral body segmentation from CT scans
- Medical image analysis research
- Educational purposes for learning medical image segmentation
- Baseline for developing custom segmentation models

### Is this suitable for clinical use?

**No.** This is a research implementation and should not be used for clinical diagnosis or treatment without proper validation, regulatory approval, and clinical supervision.

## Installation & Setup

### What are the system requirements?

- Python 3.7+
- 8GB+ RAM (16GB recommended)
- CUDA-capable GPU recommended (but not required)
- 10GB+ free disk space

See [INSTALL.md](INSTALL.md) for detailed requirements.

### Do I need a GPU?

A GPU is highly recommended for training but not strictly required. You can:
- Train on GPU (much faster): 2-4 hours for 100 epochs
- Train on CPU (slower): 20-40 hours for 100 epochs
- Use Google Colab or similar cloud services for free GPU access

### How do I install without GPU support?

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Data & Dataset

### Where can I get the dataset?

Download from: [https://data.mendeley.com/datasets/k3b363f3vz/2](https://data.mendeley.com/datasets/k3b363f3vz/2)

### Can I use my own dataset?

Yes! Organize your data as:
```
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### What image formats are supported?

- NIfTI files (.nii, .nii.gz)
- Standard images (PNG, JPG, JPEG)

### How should I prepare my data?

Use the provided script:
```bash
python prepare_data.py --source_dir ./raw_data --output_dir ./data
```

## Training

### How long does training take?

- UNet on GPU: ~2-3 hours for 100 epochs
- ResNet50-UNet on GPU: ~3-4 hours for 100 epochs
- Times vary based on dataset size and GPU type

### What batch size should I use?

Recommended batch sizes:
- 8-16 for GPUs with 8GB+ VRAM
- 4-8 for GPUs with 4-6GB VRAM
- 1-2 for GPUs with <4GB VRAM
- 1 for CPU training

### How do I resume training from a checkpoint?

```bash
python train.py --checkpoint ./checkpoints/unet_epoch_50.pth --epochs 100
```

### My training loss is not decreasing. What should I do?

1. Check that data is loaded correctly
2. Verify masks are binary (0s and 1s)
3. Reduce learning rate
4. Check for data normalization
5. Try different loss function weights

### Should I use UNet or ResNet50-UNet?

- **UNet**: Faster training, fewer parameters, good baseline
- **ResNet50-UNet**: Better accuracy, uses pre-trained weights, more robust

Start with UNet, then try ResNet50-UNet for improvement.

## Evaluation & Results

### What metrics are reported?

- **Dice Coefficient**: Overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Jaccard index
- **Pixel Accuracy**: Overall accuracy
- **Precision & Recall**: For detailed analysis

### What's a good Dice score?

- >0.90: Excellent
- 0.80-0.90: Good
- 0.70-0.80: Acceptable
- <0.70: Needs improvement

### How do I evaluate my model?

```bash
python evaluate.py --model unet --checkpoint ./checkpoints/unet_best.pth --data_dir ./data
```

## Inference & Prediction

### How do I use the model on new images?

```bash
python predict.py --model unet --checkpoint ./checkpoints/unet_best.pth --input image.nii.gz --output prediction.png
```

### Can I process multiple images at once?

Currently, the predict.py script processes one image at a time. For batch processing, you can:
1. Write a simple loop script
2. Use the model directly in Python
3. Modify predict.py to accept multiple inputs

### How do I visualize predictions?

Add the `--visualize` flag:
```bash
python predict.py --model unet --checkpoint ./checkpoints/unet_best.pth --input image.nii.gz --visualize
```

## Troubleshooting

### "CUDA out of memory" error

Solutions:
1. Reduce batch size
2. Reduce image size in config.yaml
3. Use gradient checkpointing
4. Close other applications using GPU
5. Use CPU instead

### "No module named 'torch'" error

Install PyTorch:
```bash
pip install torch torchvision
```

### Poor segmentation results

Possible causes:
1. Insufficient training data
2. Unbalanced dataset
3. Inappropriate hyperparameters
4. Need more epochs
5. Data quality issues

Solutions:
1. Collect more training data
2. Apply data augmentation
3. Tune learning rate and loss weights
4. Train longer
5. Check data preprocessing

### Training is very slow

Solutions:
1. Use GPU instead of CPU
2. Increase batch size (if memory allows)
3. Reduce image size
4. Use fewer data augmentations
5. Use mixed precision training

## Advanced Usage

### Can I use 3D volumes instead of 2D slices?

The current implementation processes 2D slices. For 3D segmentation:
1. Modify the models to use 3D convolutions
2. Update the dataset loader
3. Adjust memory requirements

### How do I fine-tune on my dataset?

```bash
# Start from pre-trained checkpoint
python train.py --checkpoint ./checkpoints/unet_best.pth --data_dir ./my_data --lr 0.00001
```

Use a lower learning rate for fine-tuning.

### Can I export the model to ONNX?

Yes, you can export to ONNX for deployment:
```python
import torch
model = get_model('unet', n_channels=1, n_classes=1)
dummy_input = torch.randn(1, 1, 256, 256)
torch.onnx.export(model, dummy_input, "model.onnx")
```

### How do I use mixed precision training?

Modify the training loop to use `torch.cuda.amp`:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, masks)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Contributions are welcome!

### I found a bug. What should I do?

1. Check if it's already reported in Issues
2. Create a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details

### Can I add new features?

Yes! Please:
1. Open an issue to discuss the feature
2. Fork the repository
3. Implement the feature
4. Submit a pull request

## Citation & License

### How do I cite this work?

See [CITATION.bib](CITATION.bib) for the BibTeX entry.

### What license is this under?

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

### Can I use this for commercial purposes?

Yes, but you must comply with GPL v3.0, which requires:
- Disclosing source code
- Using the same license
- Stating changes made

## Getting Help

### Where can I get help?

1. Check this FAQ
2. Read [README.md](README.md) and [INSTALL.md](INSTALL.md)
3. Search existing GitHub Issues
4. Create a new issue with detailed information

### Is there a community forum?

Currently, use GitHub Issues for questions and discussions. Star the repository to stay updated!

---

**Still have questions?** Open an issue on GitHub with the "question" label.
