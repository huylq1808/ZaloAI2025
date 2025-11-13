"""
Custom Detection Head for Few-Shot YOLO

Replaces YOLOv5's original detection head with a similarity-aware version.

Input:  [B, C+1, H, W] - Features concatenated with similarity map
Output: [B, num_anchors*(5+num_classes), H, W] - Detection predictions
"""

import torch
import torch.nn as nn
import math


class FewShotDetectionHead(nn.Module):
    """
    Detection head for few-shot object detection.
    
    Outputs per anchor:
    - 4 values: bbox coordinates (x, y, w, h)
    - 1 value: objectness score
    - num_classes values: class probabilities
    
    For single-class few-shot: num_classes = 1
    
    Args:
        in_channels (int): Input feature channels (features + similarity)
        num_anchors (int): Number of anchor boxes per location
        num_classes (int): Number of classes (1 for few-shot)
        stride (int): Stride of this detection head
    """
    
    def __init__(self, in_channels: int, num_anchors: int = 3, 
                 num_classes: int = 1, stride: int = 32):
        super().__init__()
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_outputs = num_anchors * (5 + num_classes)  # 5 = x,y,w,h,obj
        self.stride = stride
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True)  # SiLU = Swish activation
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
        # Prediction head
        self.pred = nn.Conv2d(256, self.num_outputs, 1)
        
        # Initialize biases for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for prediction layer
        # Initialize objectness bias to predict low confidence initially
        b = self.pred.bias.view(self.num_anchors, -1)
        b.data[:, 4] += math.log(8 / (640 / self.stride) ** 2)  # obj bias
        self.pred.bias = nn.Parameter(b.view(-1), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [B, C, H, W] - Input features
            
        Returns:
            predictions: [B, num_anchors*(5+num_classes), H, W]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        predictions = self.pred(x)
        
        return predictions


class MultiScaleDetectionHead(nn.Module):
    """
    Multi-scale detection head (like YOLOv5's 3 scales).
    
    Combines predictions from different feature map resolutions.
    
    Args:
        in_channels_list (list): List of input channels for each scale
        num_anchors (int): Anchors per location
        num_classes (int): Number of classes
        strides (list): Strides for each scale
    """
    
    def __init__(self, in_channels_list: list = [513, 513, 513],
                 num_anchors: int = 3, num_classes: int = 1,
                 strides: list = [8, 16, 32]):
        super().__init__()
        
        self.num_scales = len(in_channels_list)
        
        # Create detection head for each scale
        self.heads = nn.ModuleList([
            FewShotDetectionHead(
                in_channels=in_channels_list[i],
                num_anchors=num_anchors,
                num_classes=num_classes,
                stride=strides[i]
            )
            for i in range(self.num_scales)
        ])
    
    def forward(self, features_list: list) -> list:
        """
        Forward pass on multiple scales
        
        Args:
            features_list: List of [B, C, H, W] tensors for each scale
            
        Returns:
            predictions_list: List of predictions for each scale
        """
        predictions = []
        
        for i, features in enumerate(features_list):
            pred = self.heads[i](features)
            predictions.append(pred)
        
        return predictions


# Test module
if __name__ == '__main__':
    # Test single-scale head
    head = FewShotDetectionHead(in_channels=513, num_anchors=3, num_classes=1)
    
    dummy_input = torch.randn(2, 513, 20, 20)
    output = head(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected outputs per location: {3 * (5 + 1)} = 18")
    
    # Test multi-scale head
    multi_head = MultiScaleDetectionHead(
        in_channels_list=[513, 513, 513],
        num_anchors=3,
        num_classes=1,
        strides=[8, 16, 32]
    )
    
    dummy_features = [
        torch.randn(2, 513, 80, 80),  # Scale 1: stride 8
        torch.randn(2, 513, 40, 40),  # Scale 2: stride 16
        torch.randn(2, 513, 20, 20)   # Scale 3: stride 32
    ]
    
    multi_outputs = multi_head(dummy_features)
    
    print(f"\nMulti-scale outputs:")
    for i, out in enumerate(multi_outputs):
        print(f"  Scale {i+1}: {out.shape}")
    
    print("✓ Detection head test passed")