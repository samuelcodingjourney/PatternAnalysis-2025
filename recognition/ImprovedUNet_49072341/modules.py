import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Double convolution block with Instance Normalization and Leaky ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x


class EncoderBlock(nn.Module):
    """
    Encoder block: ConvBlock followed by MaxPooling.
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsampling followed by concatenation and ConvBlock.
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        # Handle size mismatch if necessary
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class ImprovedUNet(nn.Module):
    
    def __init__(self, in_channels=1, num_classes=4, base_features=32):
        super(ImprovedUNet, self).__init__()
        
        # Encoder path
        self.encoder1 = EncoderBlock(in_channels, base_features)
        self.encoder2 = EncoderBlock(base_features, base_features * 2)
        self.encoder3 = EncoderBlock(base_features * 2, base_features * 4)
        self.encoder4 = EncoderBlock(base_features * 4, base_features * 8)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_features * 8, base_features * 16)
        
        # Decoder path
        self.decoder4 = DecoderBlock(base_features * 16, base_features * 8)
        self.decoder3 = DecoderBlock(base_features * 8, base_features * 4)
        self.decoder2 = DecoderBlock(base_features * 4, base_features * 2)
        self.decoder1 = DecoderBlock(base_features * 2, base_features)
        
        # Final output layer
        self.out_conv = nn.Conv2d(base_features, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        
        # Output
        x = self.out_conv(x)
        return x


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Handles multi-class segmentation.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets, class_weights=None):
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class indices
            class_weights: Optional weights for each class
        """
        # Apply softmax to get probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Flatten spatial dimensions
        predictions = predictions.contiguous().view(predictions.shape[0], predictions.shape[1], -1)
        targets_one_hot = targets_one_hot.contiguous().view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        # Calculate Dice coefficient for each class
        intersection = (predictions * targets_one_hot).sum(dim=2)
        union = predictions.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        dice_scores = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Apply class weights if provided
        if class_weights is not None:
            dice_scores = dice_scores * class_weights.unsqueeze(0)
        
        # Return 1 - dice as loss (averaged over batch and classes)
        dice_loss = 1 - dice_scores.mean()
        return dice_loss


def dice_coefficient(predictions, targets, num_classes=4):
    """
    Calculate Dice coefficient for evaluation.
    Returns per-class Dice scores.
    """
    predictions = torch.argmax(predictions, dim=1)
    
    dice_scores = []
    for class_idx in range(num_classes):
        pred_mask = (predictions == class_idx).float()
        target_mask = (targets == class_idx).float()
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        
        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2. * intersection) / union
        
        dice_scores.append(dice.item())
    
    return dice_scores


if __name__ == "__main__":
    # Test the model
    model = ImprovedUNet(in_channels=1, num_classes=4, base_features=32)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 256, 128)  
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test loss
    targets = torch.randint(0, 4, (2, 256, 128))
    criterion = DiceLoss()
    loss = criterion(output, targets)
    print(f"Dice loss: {loss.item():.4f}")
