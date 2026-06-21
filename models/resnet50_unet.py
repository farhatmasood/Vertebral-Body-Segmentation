"""
ResNet50-UNet Model Implementation for Vertebral Body Segmentation

This module implements a hybrid architecture combining ResNet50 encoder with UNet decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    """Convolution block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connections"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpBlock, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResNet50UNet(nn.Module):
    """
    ResNet50-UNet Architecture for Medical Image Segmentation
    
    Uses pre-trained ResNet50 as encoder and UNet-style decoder with skip connections.
    
    Args:
        n_channels (int): Number of input channels
        n_classes (int): Number of output classes
        pretrained (bool): Use ImageNet pre-trained ResNet50 weights
        bilinear (bool): Use bilinear upsampling instead of transposed convolutions
    """
    
    def __init__(self, n_channels=1, n_classes=1, pretrained=True, bilinear=True):
        super(ResNet50UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first conv layer if input channels != 3
        if n_channels != 3:
            self.firstconv = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, 
                                       padding=3, bias=False)
            # Initialize with average of ImageNet weights if using pretrained
            if pretrained:
                weight = resnet.conv1.weight.data
                self.firstconv.weight.data = weight.mean(dim=1, keepdim=True)
        else:
            self.firstconv = resnet.conv1
        
        # Encoder (ResNet50 layers)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 256 channels
        self.encoder2 = resnet.layer2  # 512 channels
        self.encoder3 = resnet.layer3  # 1024 channels
        self.encoder4 = resnet.layer4  # 2048 channels
        
        # Decoder
        self.center = ConvBlock(2048, 2048)
        
        self.up1 = UpBlock(2048 + 1024, 512, bilinear)
        self.up2 = UpBlock(512 + 512, 256, bilinear)
        self.up3 = UpBlock(256 + 256, 128, bilinear)
        self.up4 = UpBlock(128 + 64, 64, bilinear)
        
        # Final upsampling and output
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        
        # Store feature maps for skip connections
        e0 = x
        x = self.firstmaxpool(x)
        
        e1 = self.encoder1(x)  # 256 channels
        e2 = self.encoder2(e1)  # 512 channels
        e3 = self.encoder3(e2)  # 1024 channels
        e4 = self.encoder4(e3)  # 2048 channels
        
        # Center
        center = self.center(e4)
        
        # Decoder with skip connections
        d1 = self.up1(center, e3)
        d2 = self.up2(d1, e2)
        d3 = self.up3(d2, e1)
        d4 = self.up4(d3, e0)
        
        # Final output
        out = self.final_up(d4)
        out = self.final_conv(out)
        
        return out
    
    def freeze_encoder(self):
        """Freeze encoder weights for fine-tuning"""
        for param in self.firstconv.parameters():
            param.requires_grad = False
        for param in self.firstbn.parameters():
            param.requires_grad = False
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False
        for param in self.encoder3.parameters():
            param.requires_grad = False
        for param in self.encoder4.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights"""
        for param in self.parameters():
            param.requires_grad = True


def get_resnet50_unet(n_channels=1, n_classes=1, pretrained=True):
    """
    Factory function to create ResNet50-UNet model
    
    Args:
        n_channels (int): Number of input channels
        n_classes (int): Number of output classes
        pretrained (bool): Use ImageNet pre-trained weights
    
    Returns:
        ResNet50UNet model instance
    """
    return ResNet50UNet(n_channels=n_channels, n_classes=n_classes, pretrained=pretrained)


if __name__ == "__main__":
    # Test the model
    model = ResNet50UNet(n_channels=1, n_classes=1, pretrained=False)
    x = torch.randn(1, 1, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
