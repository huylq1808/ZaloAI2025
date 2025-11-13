"""
Reference Encoder Module

Aggregates multiple reference images into a single prototype using
attention-weighted pooling.

Input:  [B, num_refs, C, H, W] - Multiple reference features
Output: [B, C, H, W] - Single prototype feature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceEncoder(nn.Module):
    """
    Encodes multiple reference images into a single prototype.
    
    Strategy:
    1. Extract global features from each reference
    2. Compute attention weights across references
    3. Weighted aggregation
    4. Refinement convolution
    
    Args:
        feat_dim (int): Feature dimension from backbone
        num_refs (int): Number of reference images (default: 3)
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, feat_dim: int = 512, num_refs: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.num_refs = num_refs
        
        # Global average pooling to get feature vectors
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Attention network: learns importance of each reference
        self.attention_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 4, 1)
        )
        
        # Refinement network: enhance aggregated features
        self.refine_conv = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, ref_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            ref_features: [B, num_refs, C, H, W] - Reference image features
            
        Returns:
            prototype: [B, C, H, W] - Aggregated prototype
            attention_weights: [B, num_refs] - Attention scores for analysis
        """
        B, N, C, H, W = ref_features.shape
        
        assert N == self.num_refs, f"Expected {self.num_refs} references, got {N}"
        
        # Step 1: Extract global features for attention
        # Reshape: [B, N, C, H, W] -> [B*N, C, H, W]
        ref_flat = ref_features.view(B * N, C, H, W)
        
        # Global pooling: [B*N, C, H, W] -> [B*N, C, 1, 1] -> [B*N, C]
        global_features = self.gap(ref_flat).squeeze(-1).squeeze(-1)
        
        # Reshape back: [B*N, C] -> [B, N, C]
        global_features = global_features.view(B, N, C)
        
        # Step 2: Compute attention weights
        # [B, N, C] -> [B, N, 1] -> [B, N]
        attn_scores = self.attention_net(global_features).squeeze(-1)
        
        # Softmax over references dimension
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, N]
        
        # Step 3: Weighted aggregation
        # Expand attention weights: [B, N] -> [B, N, 1, 1, 1]
        attn_expanded = attn_weights.view(B, N, 1, 1, 1)
        
        # Weighted sum: [B, N, C, H, W] * [B, N, 1, 1, 1] -> [B, C, H, W]
        prototype = (ref_features * attn_expanded).sum(dim=1)
        
        # Step 4: Refinement with residual connection
        refined = self.refine_conv(prototype)
        prototype = prototype + refined  # Residual
        
        return prototype, attn_weights
    
    def get_attention_visualization(self, attention_weights: torch.Tensor) -> dict:
        """
        Get attention weights for visualization
        
        Args:
            attention_weights: [B, num_refs]
            
        Returns:
            Dictionary with attention analysis
        """
        B, N = attention_weights.shape
        
        # Find most important reference
        max_indices = attention_weights.argmax(dim=1)
        
        return {
            'weights': attention_weights.cpu().numpy(),
            'most_important_ref': max_indices.cpu().numpy(),
            'mean_weights': attention_weights.mean(dim=0).cpu().numpy()
        }


# Test module
if __name__ == '__main__':
    # Test reference encoder
    encoder = ReferenceEncoder(feat_dim=512, num_refs=3)
    
    # Dummy input: batch=2, refs=3, channels=512, height=20, width=20
    dummy_refs = torch.randn(2, 3, 512, 20, 20)
    
    prototype, attn_weights = encoder(dummy_refs)
    
    print(f"Input shape: {dummy_refs.shape}")
    print(f"Prototype shape: {prototype.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights:\n{attn_weights}")
    
    # Check if attention sums to 1
    assert torch.allclose(attn_weights.sum(dim=1), torch.ones(2)), "Attention should sum to 1"
    print("✓ Reference encoder test passed")