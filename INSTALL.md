# Installation Guide

This guide provides detailed instructions for installing the Vertebral Body Segmentation project.

## System Requirements

### Hardware Requirements
- **Minimum**: 8GB RAM, 2GB VRAM (GPU), 10GB disk space
- **Recommended**: 16GB RAM, 8GB VRAM (GPU), 20GB disk space
- **GPU**: CUDA-capable GPU recommended for training (NVIDIA GTX 1060 or better)

### Software Requirements
- **Python**: 3.7 or higher
- **Operating System**: Linux, macOS, or Windows
- **CUDA** (optional): 10.2 or higher for GPU acceleration

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/farhatmasood/Vertebral-Body-Segmentation.git
cd Vertebral-Body-Segmentation
```

### 2. Create a Virtual Environment

#### Using venv (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Using conda

```bash
# Create conda environment
conda create -n vertebral-seg python=3.8

# Activate environment
conda activate vertebral-seg
```

### 3. Install Dependencies

#### CPU-only Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### GPU Installation (CUDA 11.8)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### GPU Installation (CUDA 12.1)

```bash
pip install torch torchvision
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True  # or False for CPU-only
```

### 5. Test the Installation

```bash
python example.py
```

This should run without errors and display model information.

## Download Dataset

Download the vertebral body segmentation dataset:

1. Visit: [https://data.mendeley.com/datasets/k3b363f3vz/2](https://data.mendeley.com/datasets/k3b363f3vz/2)
2. Download the dataset
3. Extract to a directory (e.g., `./raw_data`)
4. Organize the dataset:

```bash
python prepare_data.py --source_dir ./raw_data --output_dir ./data
```

Or create dummy data for testing:

```bash
python prepare_data.py --source_dir ./nonexistent
# Answer 'y' when prompted to create dummy dataset
```

## Troubleshooting

### Import Errors

If you encounter import errors:

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### CUDA/GPU Issues

1. Check CUDA version:
   ```bash
   nvidia-smi
   ```

2. Install matching PyTorch version:
   - Visit: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
   - Select your configuration
   - Install using the provided command

### Memory Issues

If you run out of memory during training:
- Reduce batch size in `config.yaml`
- Use gradient checkpointing
- Use mixed precision training

### Disk Space Issues

If you run out of disk space:
- Clean pip cache: `pip cache purge`
- Remove unused packages
- Store data on external drive

## Optional Dependencies

### Development Tools

For development, install additional tools:

```bash
pip install black flake8 pytest ipython jupyter
```

### Visualization Tools

For enhanced visualization:

```bash
pip install plotly dash
```

### Monitoring Tools

For experiment tracking:

```bash
pip install wandb tensorboard
```

## Docker Installation (Alternative)

Build and run using Docker:

```bash
# Build Docker image
docker build -t vertebral-seg .

# Run container
docker run --gpus all -it -v $(pwd)/data:/app/data vertebral-seg
```

*Note: Docker setup requires a Dockerfile (not included by default)*

## Next Steps

After installation:

1. Read the [README.md](README.md) for usage instructions
2. Review [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
3. Run `python example.py` to see usage examples
4. Prepare your data with `prepare_data.py`
5. Start training with `train.py`

## Getting Help

If you encounter issues:

1. Check the [README.md](README.md)
2. Search existing [GitHub Issues](https://github.com/farhatmasood/Vertebral-Body-Segmentation/issues)
3. Create a new issue with:
   - Detailed description of the problem
   - Steps to reproduce
   - Error messages
   - System information (OS, Python version, etc.)

## Uninstallation

To remove the environment:

### Using venv

```bash
# Deactivate environment
deactivate

# Remove virtual environment
rm -rf venv
```

### Using conda

```bash
# Deactivate and remove environment
conda deactivate
conda env remove -n vertebral-seg
```

---

For more information, refer to the main [README.md](README.md).
