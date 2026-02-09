"""
Model loading and initialization for mtrack experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetMNIST(nn.Module):
    """
    Simple ResNet model for MNIST digit classification.
    Based on ResNet architecture adapted for 28x28 grayscale images.
    """

    def __init__(self, num_classes=10, dropout=0.0):
        super(ResNetMNIST, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual blocks
        self.res_block1 = self._make_residual_block(32, 32, dropout)
        self.res_block2 = self._make_residual_block(32, 64, dropout, stride=2)
        self.res_block3 = self._make_residual_block(64, 128, dropout, stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _make_residual_block(self, in_channels, out_channels, dropout, stride=1):
        """Create a residual block."""
        layers = []

        # Main path
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        return nn.ModuleList(layers)

    def _residual_forward(self, x, block, stride=1):
        """Forward pass through a residual block."""
        identity = x

        out = x
        for i, layer in enumerate(block):
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Dropout)):
                out = layer(out)

        # Adjust identity if dimensions changed
        if stride != 1 or identity.size(1) != out.size(1):
            identity = F.avg_pool2d(identity, stride)
            # Pad channels if needed
            if identity.size(1) != out.size(1):
                padding = torch.zeros(identity.size(0),
                                      out.size(1) - identity.size(1),
                                      identity.size(2),
                                      identity.size(3),
                                      device=identity.device)
                identity = torch.cat([identity, padding], dim=1)

        out += identity
        out = F.relu(out)
        return out

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Residual blocks
        x = self._residual_forward(x, self.res_block1, stride=1)
        x = self._residual_forward(x, self.res_block2, stride=2)
        x = self._residual_forward(x, self.res_block3, stride=2)

        # Global pooling and classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.fc(x)
        return x


def load_model(model_id: str, num_classes: int = 10, dropout: float = 0.0, device: str = "cpu"):
    """
    Load model for training.

    Args:
        model_id: Model identifier
        num_classes: Number of output classes
        dropout: Dropout rate
        device: Device to load model on

    Returns:
        Model instance
    """
    # For now, we use our custom ResNet implementation
    # In the future, this could try to load from HuggingFace first
    print(f"Initializing ResNetMNIST model...")
    print(f"  Classes: {num_classes}")
    print(f"  Dropout: {dropout}")
    print(f"  Device: {device}")

    model = ResNetMNIST(num_classes=num_classes, dropout=dropout)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from {path} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss
