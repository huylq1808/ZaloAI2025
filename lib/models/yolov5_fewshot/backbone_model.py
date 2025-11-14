import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
import sys
from pathlib import Path

class FewShotYOLOv5(nn.Module):
    """
    Few-Shot YOLOv5 Detection Model
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
        Load YOLOv5 backbone with automatic download
        
        Returns:
            backbone: YOLOv5 model
            feat_dim: Feature dimension
        """
        print(f"Loading YOLOv5 from: {weights_path}")
        
        # Check if weights exist, if not download from torch hub
        weights_file = Path(weights_path)
        
        if not weights_file.exists():
            print(f"⚠️  Weights not found: {weights_path}")
            print("📥 Downloading from torch.hub...")
            
            try:
                # Download using torch hub
                model_name = weights_file.stem  # e.g., 'yolov5m'
                
                # Load model from ultralytics hub
                model = torch.hub.load(
                    'ultralytics/yolov5',
                    model_name,
                    pretrained=True,
                    trust_repo=True,
                    verbose=False
                )
                
                print(f"✓ Downloaded {model_name} successfully")
                
            except Exception as e:
                print(f"❌ Failed to download from hub: {e}")
                raise RuntimeError(f"Cannot load YOLOv5 weights: {e}")
        else:
            # Load from local file
            try:
                # Try loading with torch.load first
                ckpt = torch.load(weights_path, map_location='cpu')
                
                # Extract model from checkpoint
                if isinstance(ckpt, dict) and 'model' in ckpt:
                    model = ckpt['model'].float()
                else:
                    # Might be direct model save
                    model = ckpt
                
                print(f"✓ Loaded from local file: {weights_path}")
                
            except Exception as e1:
                print(f"Error loading checkpoint: {e1}")
                
                # Fallback: load from hub
                try:
                    model_name = Path(weights_path).stem
                    print(f"📥 Downloading {model_name} from torch.hub...")
                    
                    model = torch.hub.load(
                        'ultralytics/yolov5',
                        model_name,
                        pretrained=True,
                        trust_repo=True,
                        verbose=False
                    )
                    
                    print(f"✓ Downloaded {model_name} successfully")
                    
                except Exception as e2:
                    raise RuntimeError(f"Failed to load YOLOv5: {e2}")
        
        # Extract backbone (feature extractor)
        # YOLOv5 structure: model.model[:10] is backbone
        try:
            if hasattr(model, 'model'):
                backbone = model.model[:10]  # Feature extraction layers
            else:
                # If model is already the sequential
                backbone = model[:10]
            
            # Get feature dimension from last layer
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
            ref_images: [B, num_refs, 3, H, W]
            
        Returns:
            ref_prototype: [B, C, H, W]
            attention_weights: [B, num_refs]
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
        
        # Encode into prototype
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
            query_images: [B, 3, H, W] - Query images to detect
            ref_images: [B, num_refs, 3, H, W] - Reference images (optional)
            ref_prototype: [B, C, H, W] - Pre-computed prototype (optional)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
            - predictions: [B, num_anchors*(5+classes), H, W]
            - similarity_map: [B, 1, H, W]
            - attention_weights: [B, num_refs] (if ref_images provided)
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
        
        # Compute similarity
        similarity_map, _ = self.similarity_module(ref_prototype, query_features)
        
        # Concatenate query features and similarity
        combined = torch.cat([query_features, 
                             similarity_map.expand(-1, self.feat_dim, -1, -1)], 
                            dim=1)
        
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
        """Save model checkpoint"""
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
        """Load model checkpoint"""
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
        yolo_weights: Path to YOLOv5 weights (will auto-download if not exists)
        checkpoint: Path to trained checkpoint (optional)
        freeze_backbone: Whether to freeze backbone
        num_refs: Number of reference images
        num_classes: Number of classes
        device: Device to load model on
        
    Returns:
        model: FewShotYOLOv5 model
        metadata: Dictionary with training metadata (if checkpoint loaded)
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


# Test
if __name__ == '__main__':
    # Test model building
    print("Testing model building...")
    
    model, _ = build_fewshot_model(
        yolo_weights='yolov5m.pt',
        freeze_backbone=True,
        num_refs=3,
        device='cpu'  # Use CPU for testing
    )
    
    print("\nTesting forward pass...")
    
    # Dummy inputs
    query = torch.randn(2, 3, 640, 640)
    refs = torch.randn(2, 3, 3, 640, 640)
    
    outputs = model(query, refs)
    
    print(f"Predictions shape: {outputs['predictions'].shape}")
    print(f"Similarity map shape: {outputs['similarity_map'].shape}")
    print(f"Attention weights shape: {outputs['attention_weights'].shape}")
    
    print("\n✓ All tests passed!")