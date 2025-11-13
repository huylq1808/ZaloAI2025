"""
Few-Shot YOLOv5 Main Model

Combines all components:
1. Pretrained YOLOv5 backbone (frozen or fine-tunable)
2. Reference encoder
3. Similarity module
4. Custom detection head

This is the CORE model file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
from typing import Optional, Dict, List, Tuple

# Add YOLOv5 to path (adjust if needed)
# sys.path.append('yolov5')

from .reference_encoder import ReferenceEncoder
from .similarity_module import SimilarityModule
from .detection_head import FewShotDetectionHead, MultiScaleDetectionHead


class FewShotYOLOv5(nn.Module):
    """
    Few-Shot YOLOv5 Model
    
    Architecture:
        Reference Images (3x) ──┐
                                ├──→ Shared Backbone ──→ Reference Encoder ──→ Prototype
        Query Image ────────────┘                              ↓
                                                         Similarity Module
                                                               ↓
                                                    Query Features + Similarity
                                                               ↓
                                                       Detection Head
                                                               ↓
                                                          Predictions
    
    Args:
        yolo_weights_path (str): Path to pretrained YOLOv5 .pt file
        freeze_backbone (bool): Whether to freeze backbone weights
        num_refs (int): Number of reference images
        num_classes (int): Number of classes (1 for few-shot)
        device (str): Device to run on
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
        self.device = torch.device(device)
        
        print("="*70)
        print("Building Few-Shot YOLOv5 Model")
        print("="*70)
        
        # ===== 1. LOAD PRETRAINED YOLO & EXTRACT BACKBONE =====
        self.backbone, self.feat_dim = self._load_backbone(yolo_weights_path, freeze_backbone)
        
        # ===== 2. REFERENCE ENCODING =====
        self.reference_encoder = ReferenceEncoder(
            feat_dim=self.feat_dim,
            num_refs=num_refs,
            dropout=0.1
        )
        print(f"✓ Reference Encoder: {num_refs} refs -> prototype")
        
        # ===== 3. SIMILARITY MODULE =====
        self.similarity_module = SimilarityModule(
            in_channels=self.feat_dim,
            embed_dim=256,
            init_temperature=10.0
        )
        print(f"✓ Similarity Module: cosine similarity with learnable temperature")
        
        # ===== 4. DETECTION HEAD =====
        # Input: features (feat_dim) + similarity (1) = feat_dim + 1
        self.detection_head = FewShotDetectionHead(
            in_channels=self.feat_dim + 1,
            num_anchors=3,
            num_classes=num_classes,
            stride=32  # Assuming single-scale for simplicity
        )
        print(f"✓ Detection Head: {num_classes} class(es), 3 anchors")
        
        # Move to device
        self.to(self.device)
        
        print("="*70)
        print(f"✓ Model built successfully")
        print(f"  - Backbone: {'Frozen' if freeze_backbone else 'Trainable'}")
        print(f"  - Feature dim: {self.feat_dim}")
        print(f"  - Device: {self.device}")
        print("="*70 + "\n")
    
    def _load_backbone(self, weights_path: str, freeze: bool) -> Tuple[nn.Module, int]:
        """
        Load YOLOv5 and extract backbone
        
        Returns:
            backbone: Sequential module (layers 0-9)
            feat_dim: Feature dimension
        """
        print(f"Loading YOLOv5 from: {weights_path}")
        
        # Load checkpoint
        try:
            ckpt = torch.load(weights_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in ckpt:
                # Training checkpoint
                yolo_model = ckpt['model'].float()
            elif 'ema' in ckpt:
                # EMA checkpoint
                yolo_model = ckpt['ema'].float()
            else:
                raise KeyError("Checkpoint format not recognized")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Attempting to load with attempt_load...")
            
            try:
                # Try alternative loading method
                sys.path.append('yolov5')
                from models.experimental import attempt_load
                yolo_model = attempt_load(weights_path, map_location='cpu')
            except Exception as e2:
                raise RuntimeError(f"Failed to load YOLOv5 weights: {e2}")
        
        print(f"✓ Loaded pretrained YOLOv5")
        
        # Extract backbone (typically layers 0-9)
        # YOLOv5 structure:
        #   0-9:   Backbone (CSPDarknet53)
        #   10-23: Neck (PANet)
        #   24:    Detection head
        
        backbone_layers = list(yolo_model.model[:10])
        backbone = nn.Sequential(*backbone_layers)
        
        print(f"✓ Extracted backbone (layers 0-9)")
        
        # Freeze/unfreeze
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
            print(f"✓ Backbone frozen (transfer learning)")
        else:
            print(f"✓ Backbone trainable (fine-tuning)")
        
        # Determine feature dimension
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            dummy_feat = backbone(dummy_input)
            
            # Handle list/tuple output
            if isinstance(dummy_feat, (list, tuple)):
                dummy_feat = dummy_feat[-1]
            
            feat_dim = dummy_feat.shape[1]
        
        print(f"✓ Feature dimension: {feat_dim}")
        
        return backbone, feat_dim
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features using backbone
        
        Args:
            images: [B, 3, H, W]
            
        Returns:
            features: [B, C, H', W']
        """
        features = self.backbone(images)
        
        # Handle list output
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        return features
    
    def encode_references(self, ref_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode reference images into prototype
        
        Args:
            ref_images: [B, num_refs, 3, H, W]
            
        Returns:
            ref_prototype: [B, C, H', W']
            attention_weights: [B, num_refs]
        """
        B, N, C, H, W = ref_images.shape
        
        # Flatten: [B, N, 3, H, W] -> [B*N, 3, H, W]
        ref_flat = ref_images.view(B * N, C, H, W)
        
        # Extract features
        ref_features = self.extract_features(ref_flat)
        
        # Reshape: [B*N, C', H', W'] -> [B, N, C', H', W']
        _, C_feat, H_feat, W_feat = ref_features.shape
        ref_features = ref_features.view(B, N, C_feat, H_feat, W_feat)
        
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
            query_images: [B, 3, H, W] - Query images to detect in
            ref_images: [B, num_refs, 3, H, W] - Reference images (optional)
            ref_prototype: [B, C, H', W'] - Pre-computed prototype (optional)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
                - 'predictions': [B, num_anchors*(5+num_classes), H, W]
                - 'similarity_map': [B, 1, H', W']
                - 'ref_prototype': [B, C, H', W']
                - 'attention_weights': [B, num_refs] (if ref_images provided)
        """
        # Extract query features
        query_features = self.extract_features(query_images)
        
        # Encode references if needed
        attention_weights = None
        
        if ref_prototype is None:
            if ref_images is None:
                raise ValueError("Must provide either ref_images or ref_prototype")
            
            ref_prototype, attention_weights = self.encode_references(ref_images)
        
        # Compute similarity
        similarity_map = self.similarity_module(ref_prototype, query_features)
        
        # Concatenate features with similarity
        enhanced_features = torch.cat([query_features, similarity_map], dim=1)
        
        # Detection
        predictions = self.detection_head(enhanced_features)
        
        # Prepare output
        output = {
            'predictions': predictions,
            'similarity_map': similarity_map,
            'ref_prototype': ref_prototype
        }
        
        if attention_weights is not None:
            output['attention_weights'] = attention_weights
        
        if return_features:
            output['query_features'] = query_features
        
        return output
    
    def save_checkpoint(self, path: str, epoch: int = 0, 
                       optimizer_state: Optional[dict] = None,
                       loss: float = 0.0,
                       **kwargs):
        """
        Save model checkpoint
        
        Args:
            path: Save path
            epoch: Current epoch
            optimizer_state: Optimizer state dict
            loss: Current loss
            **kwargs: Additional metadata
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'feat_dim': self.feat_dim,
            'num_refs': self.num_refs,
            'num_classes': self.num_classes,
            'loss': loss,
            **kwargs
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")
    
    @classmethod
    def load_from_checkpoint(cls, 
                            checkpoint_path: str,
                            yolo_weights_path: str = 'yolov5m.pt',
                            device: str = 'cuda',
                            strict: bool = True):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint .pt file
            yolo_weights_path: Path to original YOLOv5 weights (for backbone)
            device: Device to load on
            strict: Strict state dict loading
            
        Returns:
            model: Loaded model instance
            metadata: Checkpoint metadata (epoch, loss, etc.)
        """
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # Rebuild model
        model = cls(
            yolo_weights_path=yolo_weights_path,
            freeze_backbone=False,  # Allow training by default
            num_refs=ckpt.get('num_refs', 3),
            num_classes=ckpt.get('num_classes', 1),
            device=device
        )
        
        # Load state dict
        model.load_state_dict(ckpt['model_state_dict'], strict=strict)
        
        # Extract metadata
        metadata = {
            'epoch': ckpt.get('epoch', 0),
            'loss': ckpt.get('loss', 0.0)
        }
        
        print(f"✓ Loaded checkpoint from epoch {metadata['epoch']}")
        print(f"  Training loss: {metadata['loss']:.4f}")
        
        return model, metadata


def build_fewshot_model(yolo_weights: str = 'yolov5m.pt',
                       checkpoint: Optional[str] = None,
                       freeze_backbone: bool = True,
                       device: str = 'cuda',
                       **kwargs) -> FewShotYOLOv5:
    """
    Builder function for Few-Shot YOLOv5
    
    Args:
        yolo_weights: Path to pretrained YOLOv5
        checkpoint: Path to trained checkpoint (optional)
        freeze_backbone: Freeze backbone
        device: Device
        **kwargs: Additional arguments
        
    Returns:
        model: FewShotYOLOv5 instance
    """
    if checkpoint is not None:
        # Load from checkpoint
        model, _ = FewShotYOLOv5.load_from_checkpoint(
            checkpoint_path=checkpoint,
            yolo_weights_path=yolo_weights,
            device=device
        )
    else:
        # Initialize new model
        model = FewShotYOLOv5(
            yolo_weights_path=yolo_weights,
            freeze_backbone=freeze_backbone,
            device=device,
            **kwargs
        )
    
    return model


# Test model
if __name__ == '__main__':
    print("Testing Few-Shot YOLOv5 Model...\n")
    
    # Build model
    model = build_fewshot_model(
        yolo_weights='yolov5m.pt',
        freeze_backbone=True,
        device='cpu'  # Use CPU for testing
    )
    
    # Dummy inputs
    ref_images = torch.randn(2, 3, 3, 640, 640)  # [B=2, N=3, C=3, H=640, W=640]
    query_images = torch.randn(2, 3, 640, 640)   # [B=2, C=3, H=640, W=640]
    
    print("Running forward pass...")
    
    # Forward
    with torch.no_grad():
        outputs = model(
            query_images=query_images,
            ref_images=ref_images,
            return_features=True
        )
    
    print("\nOutputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test checkpoint saving
    print("\nTesting checkpoint save/load...")
    
    model.save_checkpoint('test_checkpoint.pt', epoch=1, loss=0.5)
    
    loaded_model, metadata = FewShotYOLOv5.load_from_checkpoint(
        checkpoint_path='test_checkpoint.pt',
        yolo_weights_path='yolov5m.pt',
        device='cpu'
    )
    
    print(f"Metadata: {metadata}")
    
    print("\n✓ All tests passed!")