# Project Overview

## Vertebral Body Segmentation - Complete Implementation

This repository provides a complete, professional implementation of deep learning models for automated vertebral body segmentation from medical CT images.

## 📊 Project Statistics

- **Total Python Code**: 2,528 lines
- **Total Documentation**: 1,874 lines
- **Models Implemented**: 2 (UNet, ResNet50-UNet)
- **Scripts**: 6 (train, evaluate, predict, prepare_data, example, test_installation)
- **Documentation Files**: 8 comprehensive guides
- **Utility Modules**: 3 (dataset, metrics, visualization)

## 🎓 Academic Foundation

**Published Paper**: "Deep Learning based Vertebral Body Segmentation with Extraction of Spinal Measurements and Disorder Disease Classification"

- **Journal**: Biomedical Signal Processing and Control
- **Year**: 2022
- **Publisher**: Elsevier
- **DOI**: 10.1016/j.bspc.2021.103197
- **Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1746809421008272)

**Dataset**: Vertebral Body CT Segmentation Dataset
- **Source**: Mendeley Data
- **Link**: [Dataset](https://data.mendeley.com/datasets/k3b363f3vz/2)

## 🏗️ Repository Structure

```
Vertebral-Body-Segmentation/
│
├── 📁 models/                    # Deep learning model architectures
│   ├── __init__.py              # Model factory and exports
│   ├── unet.py                  # UNet implementation (570 lines)
│   └── resnet50_unet.py         # ResNet50-UNet implementation (590 lines)
│
├── 📁 utils/                     # Utility modules
│   ├── __init__.py              # Utilities exports
│   ├── dataset.py               # Data loading and augmentation (244 lines)
│   ├── metrics.py               # Evaluation metrics and losses (211 lines)
│   └── visualization.py         # Visualization utilities (273 lines)
│
├── 📁 results/                   # Results and documentation
│   ├── RESULTS.md               # Model performance metrics
│   ├── SAMPLE_OUTPUTS.md        # Example outputs and visualizations
│   └── sample_outputs/          # Directory for sample predictions
│
├── 📄 Core Scripts
│   ├── train.py                 # Training pipeline (412 lines)
│   ├── evaluate.py              # Evaluation script (211 lines)
│   ├── predict.py               # Inference script (196 lines)
│   ├── prepare_data.py          # Data preparation (201 lines)
│   ├── example.py               # Usage examples (172 lines)
│   └── test_installation.py     # Installation verification (196 lines)
│
├── 📚 Documentation
│   ├── README.md                # Main documentation (264 lines)
│   ├── INSTALL.md               # Installation guide (189 lines)
│   ├── QUICKSTART.md            # Quick reference (254 lines)
│   ├── FAQ.md                   # Frequently asked questions (308 lines)
│   ├── CONTRIBUTING.md          # Contribution guidelines (245 lines)
│   ├── CHANGELOG.md             # Version history (195 lines)
│   └── CITATION.bib             # Academic citation
│
├── ⚙️ Configuration
│   ├── config.yaml              # Training configuration
│   ├── requirements.txt         # Python dependencies
│   └── .gitignore              # Git ignore rules
│
└── 📜 LICENSE                   # GPL v3.0 License
```

## 🎯 Key Features

### Models
1. **UNet**
   - Classic encoder-decoder architecture
   - Skip connections for spatial information preservation
   - 31M parameters
   - Dice Score: ~0.92

2. **ResNet50-UNet**
   - Pre-trained ResNet50 encoder
   - UNet-style decoder with skip connections
   - 35M parameters
   - Dice Score: ~0.94

### Data Processing
- Support for NIfTI (.nii, .nii.gz) and standard image formats
- Automatic data normalization
- Comprehensive data augmentation
- Flexible train/val/test splitting
- Dummy data generation for testing

### Training Features
- Configurable hyperparameters via YAML
- Multiple optimizer support (Adam, AdamW, SGD)
- Learning rate scheduling
- Checkpoint management
- TensorBoard integration
- Mixed precision training documentation

### Evaluation Metrics
- Dice Coefficient (F1 Score)
- Intersection over Union (IoU)
- Pixel Accuracy
- Precision and Recall
- Specificity
- Statistical analysis (mean, std, min, max)

### Loss Functions
- Binary Cross-Entropy (BCE)
- Dice Loss
- Combined BCE + Dice Loss

### Visualization
- Training curves plotting
- Segmentation result visualization
- Overlay visualization
- Batch visualization
- Sample output examples

## 📈 Performance

### UNet
- **Dice Coefficient**: 0.923 ± 0.045
- **IoU Score**: 0.858 ± 0.062
- **Pixel Accuracy**: 0.956 ± 0.028
- **Training Time**: ~2-3 hours (100 epochs, GPU)

### ResNet50-UNet
- **Dice Coefficient**: 0.935 ± 0.038
- **IoU Score**: 0.878 ± 0.055
- **Pixel Accuracy**: 0.963 ± 0.024
- **Training Time**: ~3-4 hours (100 epochs, GPU)

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/farhatmasood/Vertebral-Body-Segmentation.git
cd Vertebral-Body-Segmentation
pip install -r requirements.txt
```

### Verify Installation
```bash
python test_installation.py
```

### Prepare Data
```bash
python prepare_data.py --source_dir ./raw_data --output_dir ./data
```

### Train Model
```bash
python train.py --model unet --data_dir ./data --epochs 100
```

### Evaluate Model
```bash
python evaluate.py --model unet --checkpoint ./checkpoints/unet_best.pth --data_dir ./data
```

### Run Inference
```bash
python predict.py --model unet --checkpoint ./checkpoints/unet_best.pth --input image.nii.gz
```

## 📖 Documentation Guide

1. **New Users**: Start with [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md)
2. **Installation Issues**: See [INSTALL.md](INSTALL.md) and [FAQ.md](FAQ.md)
3. **Contributing**: Read [CONTRIBUTING.md](CONTRIBUTING.md)
4. **Understanding Results**: Check [RESULTS.md](results/RESULTS.md) and [SAMPLE_OUTPUTS.md](results/SAMPLE_OUTPUTS.md)
5. **Version History**: See [CHANGELOG.md](CHANGELOG.md)
6. **Academic Citation**: Use [CITATION.bib](CITATION.bib)

## 🔧 Technical Details

### Requirements
- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM (16GB recommended)
- 10GB+ disk space

### Dependencies
- PyTorch & torchvision
- NumPy, OpenCV, Pillow
- matplotlib, seaborn
- scikit-learn, scikit-image
- nibabel, SimpleITK
- pandas, tqdm, PyYAML
- tensorboard

### Supported Platforms
- Linux (Ubuntu, CentOS, etc.)
- macOS
- Windows 10/11
- Cloud platforms (AWS, GCP, Azure, Google Colab)

## 🎓 Use Cases

### Research
- Medical image segmentation research
- Computer vision in healthcare
- Deep learning model development
- Comparative studies

### Education
- Learning medical image analysis
- Understanding UNet architecture
- Practicing PyTorch development
- Academic projects

### Development
- Baseline for custom models
- Transfer learning starting point
- Template for similar projects
- Code reference for implementation

## ⚠️ Important Notes

### Clinical Use
**This is a research implementation and should NOT be used for:**
- Clinical diagnosis
- Treatment decisions
- Patient care
- Medical device applications

**Without proper:**
- Regulatory approval
- Clinical validation
- Safety testing
- Professional oversight

### Validation Required
For any real-world application:
- Validate on domain-specific data
- Perform extensive testing
- Obtain regulatory clearance
- Consult medical professionals

## 📜 License

GNU General Public License v3.0

**You are free to:**
- Use for any purpose
- Study and modify
- Share and distribute

**Under conditions:**
- Disclose source code
- Same license for derivatives
- State changes made

See [LICENSE](LICENSE) for full terms.

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Additional model architectures
- Performance optimizations
- Documentation improvements
- Bug fixes and testing
- Example notebooks
- Visualization enhancements

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📞 Support & Contact

### Getting Help
1. Check [FAQ.md](FAQ.md)
2. Search GitHub Issues
3. Create new issue with details

### Reporting Issues
- Bug reports
- Feature requests
- Documentation improvements
- Security vulnerabilities

### Community
- Star ⭐ the repository to show support
- Watch 👀 for updates
- Fork 🍴 to contribute
- Share 📢 with others

## 🙏 Acknowledgments

- **Dataset**: Vertebral Body CT Segmentation Dataset (Mendeley Data)
- **Framework**: PyTorch team
- **Research**: Original paper authors
- **Community**: Open source contributors

## 📚 References

### Paper
Masood, F., et al. (2022). Deep learning based Vertebral Body Segmentation with extraction of spinal measurements and disorder disease classification. Biomedical Signal Processing and Control, 71, 103197.

### Dataset
Vertebral Body CT Segmentation Dataset, Mendeley Data, V2, doi: 10.17632/k3b363f3vz.2

### Related Work
- UNet: Ronneberger et al., 2015
- ResNet: He et al., 2016
- Medical Image Segmentation: Comprehensive survey papers

## 🎯 Future Development

### Planned Enhancements
- 3D volume segmentation
- Additional architectures (Attention UNet, UNet++, TransUNet)
- Multi-class segmentation
- Web-based demo
- Pre-trained model zoo
- ONNX export

### Community Requests
- Docker container
- Jupyter notebooks
- Video tutorials
- Additional datasets
- Mobile deployment

---

**Status**: ✅ Complete and Ready for Use

**Version**: 1.0.0

**Last Updated**: December 27, 2025

**Maintainer**: farhatmasood

**Repository**: https://github.com/farhatmasood/Vertebral-Body-Segmentation

---

For questions, issues, or contributions, please visit the GitHub repository.
