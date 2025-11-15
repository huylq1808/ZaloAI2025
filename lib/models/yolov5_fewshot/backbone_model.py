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
                full_model = yolov5.load(str(weights_path))
            else:
                print(f"⚠️  Weights not found locally: {weights_path}")
                print(f"📥 Downloading pretrained {model_name}...")
                full_model = yolov5.load(f'{model_name}.pt')
                print(f"✓ Downloaded {model_name} successfully")
            
        except ImportError:
            raise ImportError(
                "yolov5 package not found. Install with:\n"
                "  pip install yolov5"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv5 weights: {e}")
        
        # Extract backbone from DetectMultiBackend wrapper
        try:
            # yolov5.load() returns DetectMultiBackend
            # Access: DetectMultiBackend.model -> DetectMultiBackend or YOLOv5 model
            #         YOLOv5.model -> nn.Sequential with layers
            
            print(f"Extracting backbone from {type(full_model).__name__}...")
            
            # Navigate through wrappers
            current = full_model
            
            # Unwrap DetectMultiBackend
            if hasattr(current, 'model'):
                current = current.model
                print(f"  - Unwrapped to: {type(current).__name__}")
            
            # Get the actual YOLOv5 sequential layers
            if hasattr(current, 'model'):
                yolo_layers = current.model
                print(f"  - Found YOLOv5 layers: {type(yolo_layers).__name__}")
            else:
                yolo_layers = current
            
            # Extract first 10 layers (backbone)
            if isinstance(yolo_layers, nn.Sequential):
                backbone = yolo_layers[:10]
            elif isinstance(yolo_layers, (list, nn.ModuleList)):
                backbone = nn.Sequential(*list(yolo_layers[:10]))
            else:
                # Last resort: try to iterate and slice
                try:
                    layers_list = list(yolo_layers.children())[:10]
                    backbone = nn.Sequential(*layers_list)
                except Exception:
                    raise TypeError(
                        f"Cannot extract layers from {type(yolo_layers)}. "
                        f"Expected nn.Sequential, got {type(yolo_layers).__name__}"
                    )
            
            print(f"✓ Backbone extracted ({len(backbone)} layers)")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to extract backbone: {e}")
        
        # Get feature dimension
        try:
            feat_dim = self._get_feature_dim(backbone)
            print(f"✓ Feature dimension: {feat_dim}")
        except Exception as e:
            print(f"⚠️  Could not determine feature dim, using default 512")
            feat_dim = 512
        
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
        
        try:
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
        except Exception as e:
            print(f"Warning: Could not auto-detect feature dim: {e}")
            # Default to common YOLOv5m feature dimension
            feat_dim = 512
        
        return feat_dim
    
    def _build_detection_head(self) -> nn.Module:
        """
        Build detection head for few-shot detection
        
        Output format: [B, num_anchors * (5 + num_classes), H, W]
        Where:
        - 5: x, y, w, h, objectness
        - num_classes: class probabilities
        """
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
            ref_images: [B, num_refs, 3, H, W]
            
        Returns:
            ref_prototype: [B, C, H, W]
            attention_weights: [B, num_refs]
        """
        B, num_refs, C, H, W = ref_images.shape
        
        # Flatten batch and refs
        ref_flat = ref_images.view(B * num_refs, C, H, W)
        
        # Extract features
        ref_features = self.backbone(ref_flat)
        
        # Handle list/tuple output
        if isinstance(ref_features, (list, tuple)):
            ref_features = ref_features[-1]
        
        # Reshape back
        _, C_feat, H_feat, W_feat = ref_features.shape
        ref_features = ref_features.view(B, num_refs, C_feat, H_feat, W_feat)
        
        # Encode with attention
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
            query_images: [B, 3, H, W]
            ref_images: [B, num_refs, 3, H, W] (optional)
            ref_prototype: [B, C, H, W] (optional)
            return_features: Return intermediate features
            
        Returns:
            Dictionary with predictions, similarity_map, etc.
        """
        # Encode references
        attention_weights = None
        if ref_prototype is None:
            if ref_images is None:
                raise ValueError("Either ref_images or ref_prototype required")
            ref_prototype, attention_weights = self.encode_references(ref_images)
        
        # Extract query features
        query_features = self.backbone(query_images)
        if isinstance(query_features, (list, tuple)):
            query_features = query_features[-1]
        
        # Compute similarity
        similarity_map, _ = self.similarity_module(ref_prototype, query_features)
        
        # Combine features
        similarity_expanded = similarity_map.expand(-1, self.feat_dim, -1, -1)
        combined = torch.cat([query_features, similarity_expanded], dim=1)
        
        # Predict
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
    
    def save_checkpoint(self, path: str, epoch: int, 
                       optimizer_state: Optional[Dict] = None,
                       loss: Optional[float] = None, **kwargs) -> None:
        """Save checkpoint"""
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
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
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
        yolo_weights: Model name or path ('yolov5s', 'yolov5m', etc.)
        checkpoint: Path to trained checkpoint (optional)
        freeze_backbone: Freeze YOLOv5 backbone
        num_refs: Number of reference images
        num_classes: Number of classes
        device: Device ('cuda' or 'cpu')
        
    Returns:
        model: FewShotYOLOv5 instance
        metadata: Checkpoint metadata (if loaded)
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
            device='cpu'
        )
        print("✓ Model built successfully")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test forward pass
    print("\n[2/3] Testing forward pass...")
    try:
        query = torch.randn(2, 3, 640, 640)
        refs = torch.randn(2, 3, 3, 640, 640)
        
        with torch.no_grad():
            outputs = model(query, refs)
        
        print(f"✓ Forward pass successful")
        print(f"  - Predictions: {outputs['predictions'].shape}")
        print(f"  - Similarity: {outputs['similarity_map'].shape}")
        print(f"  - Attention: {outputs['attention_weights'].shape}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test checkpoint
    print("\n[3/3] Testing checkpoint...")
    try:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            checkpoint_path = tmp.name
        
        model.save_checkpoint(checkpoint_path, epoch=1, loss=0.5)
        
        model2, metadata = build_fewshot_model(
            yolo_weights='yolov5m.pt',
            checkpoint=checkpoint_path,
            device='cpu'
        )
        
        print(f"✓ Checkpoint operations successful")
        
        os.remove(checkpoint_path)
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
