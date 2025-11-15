"""
Few-Shot YOLOv5 Backbone Model

Uses the yolov5 Python package for easy model loading.
Install: pip install yolov5
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
import sys
from pathlib import Path


class FewShotYOLOv5(nn.Module):
    """
    Few-Shot YOLOv5 Detection Model
    
    Uses YOLOv5 as backbone with few-shot learning capabilities.
    """
    
    def __init__(self,
                 yolo_weights_path: str = 'yolov5m.pt',
                 freeze_backbone: bool = True,
                 num_refs: int = 3,
                 num_classes: int = 1,
                 device: str = 'cuda'):
        super().__init__()
        
        self.num_refs = num_refs
        self.num_classes = num_classes
        self.device = device
        
        # Load YOLOv5 backbone
        self.backbone, self.feat_dim = self._load_backbone(yolo_weights_path, freeze_backbone)
        
        # Reference encoder
        from .reference_encoder import ReferenceEncoder
        self.reference_encoder = ReferenceEncoder(
            feat_dim=self.feat_dim,
            num_refs=num_refs
        )
        
        # Similarity module
        from .similarity_module import SimilarityModule
        self.similarity_module = SimilarityModule(
            in_channels=self.feat_dim
        )
        
        # Detection head (modified for few-shot)
        self.detection_head = self._build_detection_head()
    
    def _load_backbone(self, weights_path: str, freeze: bool) -> Tuple[nn.Module, int]:
        """
        Load YOLOv5 backbone using yolov5 package
        
        Returns:
            backbone: YOLOv5 model (feature extractor)
            feat_dim: Feature dimension
        """
        print(f"Loading YOLOv5 from: {weights_path}")
        
        try:
            # Import yolov5 package
            import yolov5
            
            # Extract model name from path
            weights_file = Path(weights_path)
            model_name = weights_file.stem  # e.g., 'yolov5m'
            
            # Check if local weights exist
            if weights_file.exists():
                print(f"✓ Loading from local file: {weights_path}")
                # Load custom/local weights
                full_model = yolov5.load(str(weights_path))
            else:
                print(f"⚠️  Weights not found locally: {weights_path}")
                print(f"📥 Downloading pretrained {model_name}...")
                # Load pretrained model (auto-downloads)
                full_model = yolov5.load(f'{model_name}.pt')
                print(f"✓ Downloaded {model_name} successfully")
            
        except ImportError:
            raise ImportError(
                "yolov5 package not found. Install with: pip install yolov5"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv5 weights: {e}")
        
        # Extract backbone (feature extractor)
        try:
            # YOLOv5 model structure: model.model contains the layers
            if hasattr(full_model, 'model'):
                # Take layers 0-9 (backbone without detection head)
                backbone = full_model.model[:10]
            else:
                raise AttributeError("Unexpected YOLOv5 model structure")
            
            # Get feature dimension
            feat_dim = self._get_feature_dim(backbone)
            
            print(f"✓ Backbone extracted, feature dim: {feat_dim}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract backbone: {e}")
        
        # Freeze backbone if requested
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
            print("✓ Backbone frozen")
        else:
            print("✓ Backbone trainable")
        
        return backbone, feat_dim
    
    def _get_feature_dim(self, backbone: nn.Module) -> int:
        """
        Get output feature dimension of backbone
        
        Args:
            backbone: YOLOv5 backbone module
            
        Returns:
            feat_dim: Feature dimension (number of channels)
        """
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 640, 640)
        
        with torch.no_grad():
            features = backbone(dummy_input)
            
            # Handle different output formats
            if isinstance(features, (list, tuple)):
                features = features[-1]  # Take last feature map
            
            # Get channel dimension
            if len(features.shape) == 4:  # [B, C, H, W]
                feat_dim = features.shape[1]
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        return feat_dim
    
    def _build_detection_head(self) -> nn.Module:
        """
        Build detection head for few-shot detection
        
        Output format: [B, num_anchors * (5 + num_classes), H, W]
        Where:
        - 5: x, y, w, h, objectness
        - num_classes: class probabilities (1 for single class)
        """
        # YOLOv5 uses 3 anchors per grid cell
        num_anchors = 3
        output_channels = num_anchors * (5 + self.num_classes)
        
        head = nn.Sequential(
            # Combine query and similarity features
            nn.Conv2d(self.feat_dim * 2, self.feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
            
            # Detection output
            nn.Conv2d(self.feat_dim, output_channels, 1)
        )
        
        return head
    
    def encode_references(self, 
                         ref_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode reference images into prototype
        
        Args:
            ref_images: [B, num_refs, 3, H, W] - Reference images
            
        Returns:
            ref_prototype: [B, C, H, W] - Aggregated prototype
            attention_weights: [B, num_refs] - Attention weights per reference
        """
        B, num_refs, C, H, W = ref_images.shape
        
        # Extract features for each reference
        # Reshape: [B, num_refs, 3, H, W] -> [B*num_refs, 3, H, W]
        ref_flat = ref_images.view(B * num_refs, C, H, W)
        
        # Extract features
        ref_features = self.backbone(ref_flat)
        
        # Handle list output
        if isinstance(ref_features, (list, tuple)):
            ref_features = ref_features[-1]
        
        # Reshape back: [B*num_refs, C', H', W'] -> [B, num_refs, C', H', W']
        _, C_feat, H_feat, W_feat = ref_features.shape
        ref_features = ref_features.view(B, num_refs, C_feat, H_feat, W_feat)
        
        # Encode into prototype using attention
        ref_prototype, attention_weights = self.reference_encoder(ref_features)
        
        return ref_prototype, attention_weights
    
    def forward(self,
                query_images: torch.Tensor,
                ref_images: Optional[torch.Tensor] = None,
                ref_prototype: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query_images: [B, 3, H, W] - Query images to detect objects in
            ref_images: [B, num_refs, 3, H, W] - Reference images (optional)
            ref_prototype: [B, C, H, W] - Pre-computed prototype (optional)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
            - predictions: [B, num_anchors*(5+classes), H, W] - Detection outputs
            - similarity_map: [B, 1, H, W] - Query-reference similarity
            - attention_weights: [B, num_refs] - Reference attention (if ref_images provided)
        """
        # Encode references if not provided
        attention_weights = None
        
        if ref_prototype is None:
            if ref_images is None:
                raise ValueError("Either ref_images or ref_prototype must be provided")
            
            ref_prototype, attention_weights = self.encode_references(ref_images)
        
        # Extract query features
        query_features = self.backbone(query_images)
        
        # Handle list output
        if isinstance(query_features, (list, tuple)):
            query_features = query_features[-1]
        
        # Compute similarity between query and reference
        similarity_map, _ = self.similarity_module(ref_prototype, query_features)
        
        # Concatenate query features and similarity
        # Expand similarity to match feature dimension
        similarity_expanded = similarity_map.expand(-1, self.feat_dim, -1, -1)
        combined = torch.cat([query_features, similarity_expanded], dim=1)
        
        # Detection head
        predictions = self.detection_head(combined)
        
        # Prepare output
        output = {
            'predictions': predictions,
            'similarity_map': similarity_map
        }
        
        if attention_weights is not None:
            output['attention_weights'] = attention_weights
        
        if return_features:
            output['query_features'] = query_features
            output['ref_prototype'] = ref_prototype
        
        return output
    
    def save_checkpoint(self, 
                       path: str,
                       epoch: int,
                       optimizer_state: Optional[Dict] = None,
                       loss: Optional[float] = None,
                       **kwargs) -> None:
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict (optional)
            loss: Current loss value (optional)
            **kwargs: Additional metadata to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'num_refs': self.num_refs,
            'num_classes': self.num_classes,
            'feat_dim': self.feat_dim,
            'loss': loss
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str) -> Dict:
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Dictionary containing checkpoint metadata
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load model weights
        self.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint


def build_fewshot_model(yolo_weights: str = 'yolov5m.pt',
                       checkpoint: Optional[str] = None,
                       freeze_backbone: bool = True,
                       num_refs: int = 3,
                       num_classes: int = 1,
                       device: str = 'cuda',
                       **kwargs) -> Tuple[FewShotYOLOv5, Dict]:
    """
    Build Few-Shot YOLOv5 model
    
    Args:
        yolo_weights: YOLOv5 model name or path to weights
                     Options: 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
        checkpoint: Path to trained checkpoint (optional)
        freeze_backbone: Whether to freeze YOLOv5 backbone
        num_refs: Number of reference images
        num_classes: Number of detection classes
        device: Device to load model on ('cuda' or 'cpu')
        **kwargs: Additional arguments
        
    Returns:
        model: FewShotYOLOv5 model instance
        metadata: Dictionary with training metadata (if checkpoint loaded)
        
    Example:
        >>> model, _ = build_fewshot_model(
        ...     yolo_weights='yolov5m.pt',
        ...     freeze_backbone=True,
        ...     num_refs=3,
        ...     device='cuda'
        ... )
    """
    print("="*70)
    print("Building Few-Shot YOLOv5 Model")
    print("="*70)
    
    # Build model
    model = FewShotYOLOv5(
        yolo_weights_path=yolo_weights,
        freeze_backbone=freeze_backbone,
        num_refs=num_refs,
        num_classes=num_classes,
        device=device,
        **kwargs
    )
    
    # Load checkpoint if provided
    metadata = {}
    if checkpoint is not None:
        print(f"\nLoading checkpoint: {checkpoint}")
        metadata = model.load_checkpoint(checkpoint)
    
    # Move to device
    model = model.to(device)
    
    print("\n✓ Model built successfully")
    print(f"  - Backbone feature dim: {model.feat_dim}")
    print(f"  - Number of references: {model.num_refs}")
    print(f"  - Number of classes: {model.num_classes}")
    print(f"  - Device: {device}")
    print("="*70 + "\n")
    
    return model, metadata


# Test module
if __name__ == '__main__':
    print("Testing Few-Shot YOLOv5 Model")
    print("="*70)
    
    # Test model building
    print("\n[1/3] Building model...")
    try:
        model, _ = build_fewshot_model(
            yolo_weights='yolov5m.pt',
            freeze_backbone=True,
            num_refs=3,
            device='cpu'  # Use CPU for testing
        )
        print("✓ Model built successfully")
    except Exception as e:
        print(f"❌ Failed to build model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test forward pass
    print("\n[2/3] Testing forward pass...")
    try:
        # Create dummy inputs
        batch_size = 2
        query = torch.randn(batch_size, 3, 640, 640)
        refs = torch.randn(batch_size, 3, 3, 640, 640)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(query, refs)
        
        print(f"✓ Forward pass successful")
        print(f"  - Predictions shape: {outputs['predictions'].shape}")
        print(f"  - Similarity map shape: {outputs['similarity_map'].shape}")
        print(f"  - Attention weights shape: {outputs['attention_weights'].shape}")
        F
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test checkpoint save/load
    print("\n[3/3] Testing checkpoint operations...")
    try:
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            checkpoint_path = tmp.name
        
        # Save checkpoint
        model.save_checkpoint(
            path=checkpoint_path,
            epoch=1,
            loss=0.5
        )
        
        # Load checkpoint
        model2, metadata = build_fewshot_model(
            yolo_weights='yolov5m.pt',
            checkpoint=checkpoint_path,
            device='cpu'
        )
        
        print(f"✓ Checkpoint operations successful")
        print(f"  - Loaded epoch: {metadata['epoch']}")
        print(f"  - Loaded loss: {metadata['loss']}")
        
        # Cleanup
        os.remove(checkpoint_path)
        
    except Exception as e:
        print(f"❌ Checkpoint operations failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
