# Deep Learning based Vertebral Body Segmentation

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the implementation of deep learning models for **Vertebral Body Segmentation with Extraction of Spinal Measurements and Disorder Disease Classification** as described in our paper published in Biomedical Signal Processing and Control.

## Paper

**Title:** Deep Learning based Vertebral Body Segmentation with Extraction of Spinal Measurements and Disorder Disease Classification

**Link:** [https://www.sciencedirect.com/science/article/abs/pii/S1746809421008272](https://www.sciencedirect.com/science/article/abs/pii/S1746809421008272)

**Citation:**
```bibtex
@article{masood2022vertebral,
  title={Deep learning based Vertebral Body Segmentation with extraction of spinal measurements and disorder disease classification},
  author={Masood, Farhat and others},
  journal={Biomedical Signal Processing and Control},
  volume={71},
  pages={103197},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.bspc.2021.103197}
}
```

## Overview

This project implements state-of-the-art deep learning models for automated vertebral body segmentation from medical images:

- **UNet**: A fully convolutional network specifically designed for medical image segmentation
- **ResNet50-UNet**: A hybrid architecture combining ResNet50 encoder with UNet decoder for improved feature extraction

### Key Features

- ✅ Complete implementation of UNet and ResNet50-based segmentation models
- ✅ Data preprocessing and augmentation pipelines
- ✅ Training and evaluation scripts
- ✅ Performance metrics (Dice Score, IoU, Accuracy)
- ✅ Visualization tools for segmentation results
- ✅ Pre-trained model checkpoints
- ✅ Inference on new images

## Dataset

This implementation uses the **Vertebral Body Segmentation Dataset** available at:
[https://data.mendeley.com/datasets/k3b363f3vz/2](https://data.mendeley.com/datasets/k3b363f3vz/2)

### Dataset Structure

After downloading, organize your dataset as follows:

```
data/
├── train/
│   ├── images/
│   │   ├── image_001.nii.gz
│   │   ├── image_002.nii.gz
│   │   └── ...
│   └── masks/
│       ├── mask_001.nii.gz
│       ├── mask_002.nii.gz
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/farhatmasood/Vertebral-Body-Segmentation.git
cd Vertebral-Body-Segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the UNet model:
```bash
python train.py --model unet --data_dir ./data --epochs 100 --batch_size 8
```

Train the ResNet50-UNet model:
```bash
python train.py --model resnet50_unet --data_dir ./data --epochs 100 --batch_size 8
```

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --model unet --checkpoint ./checkpoints/unet_best.pth --data_dir ./data
```

### Inference

Run inference on new images:
```bash
python predict.py --model unet --checkpoint ./checkpoints/unet_best.pth --input ./test_image.nii.gz --output ./prediction.nii.gz
```

### Configuration

You can customize training parameters by modifying `config.yaml` or passing command-line arguments. Key parameters include:

- `--model`: Model architecture (unet, resnet50_unet)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--data_dir`: Path to dataset directory

## Results

### Model Performance

| Model | Dice Score | IoU | Accuracy |
|-------|-----------|-----|----------|
| UNet | 0.923 | 0.858 | 0.956 |
| ResNet50-UNet | 0.935 | 0.878 | 0.963 |

### Sample Segmentation Results

Example segmentation outputs can be found in the `results/` directory, showing:
- Original MR Slices
- Ground truth segmentation masks
- Model predictions
- Overlayed visualizations

## Project Structure

```
Vertebral-Body-Segmentation/
├── models/
│   ├── unet.py              # UNet architecture
│   ├── resnet50_unet.py     # ResNet50-UNet architecture
│   └── __init__.py
├── utils/
│   ├── dataset.py           # Data loading and preprocessing
│   ├── metrics.py           # Evaluation metrics
│   ├── visualization.py     # Visualization utilities
│   └── __init__.py
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── predict.py               # Inference script
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── CITATION.bib             # Citation information
└── LICENSE                  # GPL v3 License
```

## 🔬 Model Architectures

### UNet

The UNet model consists of:
- Encoder (contracting path): 4 downsampling blocks
- Decoder (expanding path): 4 upsampling blocks with skip connections
- Output: Binary segmentation mask

### ResNet50-UNet

The ResNet50-UNet model features:
- Encoder: Pre-trained ResNet50 backbone
- Decoder: UNet-style upsampling with skip connections
- Enhanced feature extraction through residual connections

## Metrics

The following metrics are used for evaluation:

- **Dice Coefficient**: Measures overlap between predicted and ground truth segmentation
- **Intersection over Union (IoU)**: Jaccard index for segmentation quality
- **Pixel Accuracy**: Overall pixel-wise classification accuracy

## Implementation Details

- **Framework**: PyTorch
- **Loss Function**: Combined Binary Cross-Entropy and Dice Loss
- **Optimizer**: Adam with learning rate scheduling
- **Data Augmentation**: Random rotations, flips, and intensity variations
- **Input Size**: 256×256 (2D slices) or 128×128×128 (3D volumes)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaborations, please contact the authors through the paper or open an issue in this repository.

## Acknowledgments

- Dataset: [Vertebral Body Segmentation Dataset](https://data.mendeley.com/datasets/k3b363f3vz/2)
- Paper: [Biomedical Signal Processing and Control](https://www.sciencedirect.com/science/article/abs/pii/S1746809421008272)

---

**Note**: This repository contains the implementation and code for research purposes. For clinical applications, please ensure proper validation and regulatory compliance.
