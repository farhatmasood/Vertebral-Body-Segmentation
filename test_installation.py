"""
Test Script to Verify Installation

This script performs basic tests to verify that the installation is working correctly.
Run this after installing dependencies to ensure everything is set up properly.
"""

import sys
import importlib


def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name or module_name} is NOT installed")
        return False


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available (device: {torch.cuda.get_device_name(0)})")
            return True
        else:
            print("✓ PyTorch is installed (CPU only)")
            return True
    except:
        print("✗ Cannot check CUDA availability")
        return False


def test_model_creation():
    """Test model creation"""
    try:
        from models import get_model
        import torch
        
        # Create UNet
        model = get_model('unet', n_channels=1, n_classes=1, base_filters=64)
        print(f"✓ UNet model created successfully ({sum(p.numel() for p in model.parameters()):,} parameters)")
        
        # Create ResNet50-UNet
        model = get_model('resnet50_unet', n_channels=1, n_classes=1, pretrained=False)
        print(f"✓ ResNet50-UNet model created successfully ({sum(p.numel() for p in model.parameters()):,} parameters)")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass"""
    try:
        from models import get_model
        import torch
        
        model = get_model('unet', n_channels=1, n_classes=1)
        model.eval()
        
        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1, 256, 256), f"Unexpected output shape: {output.shape}"
        print(f"✓ Forward pass successful (input: {x.shape}, output: {output.shape})")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_metrics():
    """Test metrics calculation"""
    try:
        from utils.metrics import dice_coefficient, iou_score, pixel_accuracy
        import torch
        
        pred = torch.ones(1, 256, 256)
        target = torch.ones(1, 256, 256)
        
        dice = dice_coefficient(pred, target)
        iou = iou_score(pred, target)
        acc = pixel_accuracy(pred, target)
        
        assert abs(dice - 1.0) < 0.01, f"Expected dice=1.0, got {dice}"
        assert abs(iou - 1.0) < 0.01, f"Expected iou=1.0, got {iou}"
        assert abs(acc - 1.0) < 0.01, f"Expected acc=1.0, got {acc}"
        
        print(f"✓ Metrics calculation successful (Dice={dice:.4f}, IoU={iou:.4f}, Acc={acc:.4f})")
        return True
    except Exception as e:
        print(f"✗ Metrics calculation failed: {e}")
        return False


def test_loss_functions():
    """Test loss functions"""
    try:
        from utils.metrics import DiceLoss, CombinedLoss
        import torch
        
        dice_loss = DiceLoss()
        combined_loss = CombinedLoss()
        
        pred = torch.randn(2, 1, 256, 256)
        target = torch.randint(0, 2, (2, 1, 256, 256)).float()
        
        loss1 = dice_loss(pred, target)
        loss2 = combined_loss(pred, target)
        
        assert loss1.numel() == 1, "Loss should be scalar"
        assert loss2.numel() == 1, "Loss should be scalar"
        
        print(f"✓ Loss functions working (DiceLoss={loss1.item():.4f}, CombinedLoss={loss2.item():.4f})")
        return True
    except Exception as e:
        print(f"✗ Loss functions failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print(" "*15 + "VERTEBRAL BODY SEGMENTATION")
    print(" "*20 + "Installation Test")
    print("="*70)
    print()
    
    print("Checking Python version...")
    py_version = sys.version_info
    print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    print()
    
    print("Checking dependencies...")
    all_good = True
    
    # Check core dependencies
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'NumPy'),
        ('cv2', 'opencv-python'),
        ('matplotlib', 'matplotlib'),
        ('sklearn', 'scikit-learn'),
        ('skimage', 'scikit-image'),
        ('PIL', 'Pillow'),
        ('tqdm', 'tqdm'),
        ('nibabel', 'nibabel'),
        ('SimpleITK', 'SimpleITK'),
        ('pandas', 'pandas'),
        ('seaborn', 'seaborn'),
        ('yaml', 'pyyaml'),
    ]
    
    for module, package in dependencies:
        if not check_import(module, package):
            all_good = False
    
    print()
    
    # Check CUDA
    print("Checking CUDA...")
    check_cuda()
    print()
    
    # Test model functionality
    print("Testing model creation...")
    if not test_model_creation():
        all_good = False
    print()
    
    print("Testing forward pass...")
    if not test_forward_pass():
        all_good = False
    print()
    
    print("Testing metrics...")
    if not test_metrics():
        all_good = False
    print()
    
    print("Testing loss functions...")
    if not test_loss_functions():
        all_good = False
    print()
    
    # Summary
    print("="*70)
    if all_good:
        print("✓ All tests passed! Installation is working correctly.")
        print()
        print("Next steps:")
        print("1. Prepare your data: python prepare_data.py --source_dir ./raw_data")
        print("2. Train a model: python train.py --model unet --data_dir ./data")
        print("3. See example.py for more usage examples")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print()
        print("To fix dependency issues:")
        print("  pip install -r requirements.txt")
        print()
        print("For help, see INSTALL.md or create an issue on GitHub")
    print("="*70)
    
    return 0 if all_good else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
