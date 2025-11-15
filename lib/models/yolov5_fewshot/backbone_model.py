"""
Few-Shot YOLOv5 Backbone Model

Uses the yolov5 Python package with feature extraction via hooks.
This approach preserves YOLOv5's internal architecture while extracting
intermediate features for few-shot learning.

Install: pip install yolov5
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List
from pathlib import Path


class FewShotYOLOv5(nn.Module):
    """
    Few-Shot YOLOv5 Detection Model
    
    Architecture:
    1. YOLOv5 backbone (frozen/trainable) - feature extraction via hooks
    2. Reference Encoder - aggregates multiple reference images
    3. Similarity Module - computes query-reference similarity
    4. Detection Head - predicts bounding boxes
    
    Args:
        yolo_weights_path: Path to YOLOv5 weights or model name
        freeze_backbone: Whether to freeze YOLOv5 parameters
        num_refs: Number of reference images (default: 3)
        num_classes: Number of object classes (default: 1)
        device: Device to run on ('cuda' or 'cpu')
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
        
        # Load YOLOv5 as feature extractor (keeps full model structure)
        self.yolo_model, self.feat_dim = self._load_backbone(
            yolo_weights_path, 
            freeze_backbone
        )
        
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
        
        # Detection head
        self.detection_head = self._build_detection_head()
    
    def _load_backbone(self, weights_path: str, freeze: bool) -> Tuple[nn.Module, int]:
        """
        Load YOLOv5 model and configure for feature extraction
        
        Key change: Instead of slicing the model (which breaks skip connections),
        we keep the full model and use forward hooks to extract intermediate features.
        
        Args:
            weights_path: Path to weights or model name
            freeze: Whether to freeze model parameters
            
        Returns:
            yolo_model: Full YOLOv5 model
            feat_dim: Feature dimension at extraction layer
        """
        print(f"Loading YOLOv5 from: {weights_path}")
        
        try:
            import yolov5
            
            weights_file = Path(weights_path)
            model_name = weights_file.stem
            
            # Load model
            if weights_file.exists():
                print(f"✓ Loading from local file: {weights_path}")
                yolo_wrapper = yolov5.load(str(weights_path))
            else:
                print(f"⚠️  Weights not found locally: {weights_path}")
                print(f"📥 Downloading pretrained {model_name}...")
                yolo_wrapper = yolov5.load(f'{model_name}.pt')
                print(f"✓ Downloaded {model_name} successfully")
            
        except ImportError:
            raise ImportError(
                "yolov5 package not found.\n"
                "Install with: pip install yolov5"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv5: {e}")
        
        # Extract the underlying model from DetectMultiBackend wrapper
        try:
            print(f"Extracting model from {type(yolo_wrapper).__name__}...")
            
            # Navigate through wrapper layers
            yolo_model = yolo_wrapper
            
            if hasattr(yolo_model, 'model'):
                yolo_model = yolo_model.model
                print(f"  → Unwrapped to: {type(yolo_model).__name__}")
            
            if hasattr(yolo_model, 'model'):
                # This is the actual nn.Sequential with YOLOv5 layers
                print(f"  → Found YOLOv5 layers: {type(yolo_model.model).__name__}")
            else:
                print(f"  → Using model as-is: {type(yolo_model).__name__}")
            
            print(f"✓ YOLOv5 model extracted successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract model structure: {e}")
        
        # Determine feature dimension based on model size
        feat_dim = self._get_feature_dim_from_model(yolo_model)
        print(f"✓ Feature dimension: {feat_dim}")
        
        # Freeze model if requested
        if freeze:
            for param in yolo_model.parameters():
                param.requires_grad = False
            print("✓ YOLOv5 backbone frozen")
        else:
            print("✓ YOLOv5 backbone trainable")
        
        return yolo_model, feat_dim
    
    def _get_feature_dim_from_model(self, model) -> int:
        """
        Determine feature dimension based on YOLOv5 model size
        
        YOLOv5 models have different channel dimensions:
        - yolov5n/s: 256 channels at P3
        - yolov5m:   512 channels at P3
        - yolov5l/x: 512-640 channels at P3
        
        Args:
            model: YOLOv5 model
            
        Returns:
            feat_dim: Number of feature channels
        """
        # Map model size to feature dimension
        model_type_to_feat_dim = {
            'yolov5n': 256,
            'yolov5s': 256,
            'yolov5m': 512,
            'yolov5l': 512,
            'yolov5x': 640,
        }
        
        try:
            # Count parameters to identify model size
            n_params = sum(p.numel() for p in model.parameters())
            
            if n_params < 2e6:      # < 2M params → nano/small
                feat_dim = 256
            elif n_params < 25e6:   # < 25M params → medium
                feat_dim = 512
            else:                   # >= 25M params → large/xlarge
                feat_dim = 640
            
            print(f"  - Model parameters: {n_params/1e6:.1f}M")
            
        except Exception:
            # Default to medium model size
            feat_dim = 512
            print(f"  - Using default feature dim: {feat_dim}")
        
        return feat_dim
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate features from YOLOv5 using forward hooks
        
        This approach:
        1. Preserves YOLOv5's internal structure (no slicing)
        2. Captures features at layer 9 (last backbone layer before detection head)
        3. Handles skip connections and concat layers correctly
        
        Args:
            images: Input images [B, 3, H, W]
            
        Returns:
            features: Feature maps [B, C, H', W'] from layer 9
        """
        features = None
        
        def hook_fn(module, input, output):
            """Hook function to capture layer output"""
            nonlocal features
            # Handle different output types
            if isinstance(output, (list, tuple)):
                features = output[-1]
            else:
                features = output
        
        # Register hook at layer 9 (last backbone layer)
        # YOLOv5 structure: layers 0-9 are backbone, 10+ are detection heads
        try:
            if hasattr(self.yolo_model, 'model'):
                # model.model is nn.Sequential with layers
                target_layer = self.yolo_model.model[9]
            else:
                # Fallback: try to get layer 9 directly
                layers = list(self.yolo_model.children())
                target_layer = layers[9] if len(layers) > 9 else layers[-1]
            
        except Exception as e:
            raise RuntimeError(f"Failed to access layer 9: {e}")
        
        # Register hook
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            # Forward pass (hook will capture features)
            with torch.set_grad_enabled(self.training):
                _ = self.yolo_model(images)
        finally:
            # Always remove hook
            handle.remove()
        
        if features is None:
            raise RuntimeError("Failed to extract features (hook returned None)")
        
        return features
    
    def _build_detection_head(self) -> nn.Module:
        """
        Build detection head for few-shot detection
        
        Input: Concatenated features [query_features + similarity_map]
        Output: YOLOv5-format predictions [B, num_anchors * (5 + num_classes), H, W]
        
        Where:
        - num_anchors: 3 (YOLOv5 standard)
        - 5: bbox parameters (x, y, w, h, objectness)
        - num_classes: class probabilities
        
        Returns:
            Detection head module
        """
        num_anchors = 3
        output_channels = num_anchors * (5 + self.num_classes)
        
        head = nn.Sequential(
            # Combine query features and similarity map
            # Input: [B, feat_dim * 2, H, W]
            nn.Conv2d(self.feat_dim * 2, self.feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
            
            # Detection predictions
            # Output: [B, num_anchors * (5 + num_classes), H, W]
            nn.Conv2d(self.feat_dim, output_channels, 1)
        )
        
        return head
    
    def encode_references(self, 
                         ref_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode multiple reference images into single prototype
        
        Process:
        1. Extract features from each reference image
        2. Aggregate using attention mechanism
        3. Return prototype and attention weights
        
        Args:
            ref_images: [B, num_refs, 3, H, W] - Reference images
            
        Returns:
            ref_prototype: [B, C, H', W'] - Aggregated prototype
            attention_weights: [B, num_refs] - Attention weights per reference
        """
        B, num_refs, C, H, W = ref_images.shape
        
        # Flatten batch and refs: [B, num_refs, 3, H, W] → [B*num_refs, 3, H, W]
        ref_flat = ref_images.view(B * num_refs, C, H, W)
        
        # Extract features using hook (preserves YOLOv5 structure)
        ref_features = self._extract_features(ref_flat)
        
        # Reshape back: [B*num_refs, C', H', W'] → [B, num_refs, C', H', W']
        _, C_feat, H_feat, W_feat = ref_features.shape
        ref_features = ref_features.view(B, num_refs, C_feat, H_feat, W_feat)
        
        # Aggregate using attention
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
            ref_prototype: [B, C, H', W'] - Pre-computed prototype (optional)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
            - predictions: [B, num_anchors*(5+classes), H', W'] - Detection outputs
            - similarity_map: [B, 1, H', W'] - Query-reference similarity
            - attention_weights: [B, num_refs] - Reference attention (if ref_images provided)
            - query_features: [B, C, H', W'] - Query features (if return_features=True)
            - ref_prototype: [B, C, H', W'] - Reference prototype (if return_features=True)
        """
        # Encode references if not provided
        attention_weights = None
        if ref_prototype is None:
            if ref_images is None:
                raise ValueError(
                    "Either ref_images or ref_prototype must be provided. "
                    "ref_images: [B, num_refs, 3, H, W], "
                    "ref_prototype: [B, C, H', W']"
                )
            ref_prototype, attention_weights = self.encode_references(ref_images)
        
        # Extract query features using hook
        query_features = self._extract_features(query_images)
        
        # Compute similarity between query and reference
        similarity_map, _ = self.similarity_module(ref_prototype, query_features)
        
        # Combine features
        # Expand similarity to match feature dimension
        similarity_expanded = similarity_map.expand(-1, self.feat_dim, -1, -1)
        combined = torch.cat([query_features, similarity_expanded], dim=1)
        
        # Predict detections
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
        >>> 
        >>> # Forward pass
        >>> query = torch.randn(2, 3, 640, 640).cuda()
        >>> refs = torch.randn(2, 3, 3, 640, 640).cuda()
        >>> outputs = model(query, refs)
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
    print("\n[1/4] Building model...")
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
        import sys
        sys.exit(1)
    
    # Test feature extraction
    print("\n[2/4] Testing feature extraction...")
    try:
        dummy_input = torch.randn(1, 3, 640, 640)
        features = model._extract_features(dummy_input)
        print(f"✓ Feature extraction successful")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Features shape: {features.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
    
    # Test forward pass
    print("\n[3/4] Testing forward pass...")
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
        import sys
        sys.exit(1)
    
    # Test checkpoint operations
    print("\n[4/4] Testing checkpoint save/load...")
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
        print(f"  - Loaded epoch: {metadata['epoch']}")
        
        os.remove(checkpoint_path)
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
