"""
Similarity Module

Computes spatial similarity between reference prototype and query features.
Uses cosine similarity with learnable temperature scaling.

Input:  ref_prototype [B, C, H, W], query_features [B, C, H', W']
Output: similarity_map [B, 1, H', W']
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityModule(nn.Module):
    """
    Computes similarity map between reference and query.
    
    Method:
    1. Project features to common embedding space
    2. Normalize (L2 normalization)
    3. Compute cosine similarity at each spatial location
    4. Scale with learnable temperature
    5. Apply sigmoid for [0, 1] range
    
    Args:
        in_channels (int): Input feature channels
        embed_dim (int): Embedding dimension (default: 256)
        init_temperature (float): Initial temperature value
    """
    
    def __init__(self, in_channels: int = 512, embed_dim: int = 256, 
                 init_temperature: float = 10.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Projection networks: map to common embedding space
        self.ref_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        self.query_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Learnable temperature parameter
        # Higher temperature -> sharper similarity
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        
        # Optional: Additional refinement
        self.refine = nn.Conv2d(1, 1, 3, padding=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, ref_prototype: torch.Tensor, 
                query_features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity map
        
        Args:
            ref_prototype: [B, C, H, W] - Reference prototype
            query_features: [B, C, H', W'] - Query features
            
        Returns:
            similarity_map: [B, 1, H', W'] - Similarity scores in [0, 1]
        """
        # Step 1: Project to embedding space
        ref_embed = self.ref_proj(ref_prototype)      # [B, embed_dim, H, W]
        query_embed = self.query_proj(query_features)  # [B, embed_dim, H', W']
        
        # Step 2: Resize reference to match query spatial dimensions
        if ref_embed.shape[-2:] != query_embed.shape[-2:]:
            ref_embed = F.interpolate(
                ref_embed,
                size=query_embed.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Step 3: Normalize features (L2 normalization)
        ref_norm = F.normalize(ref_embed, p=2, dim=1)     # [B, embed_dim, H', W']
        query_norm = F.normalize(query_embed, p=2, dim=1) # [B, embed_dim, H', W']
        
        # Step 4: Compute cosine similarity at each spatial location
        # Element-wise multiplication and sum over channel dimension
        cosine_sim = (ref_norm * query_norm).sum(dim=1, keepdim=True)  # [B, 1, H', W']
        
        # Cosine similarity is in [-1, 1]
        # We expect positive similarity for matching regions
        
        # Step 5: Scale by temperature and apply sigmoid
        # Temperature scaling controls sharpness:
        #   - High temperature (>10): Sharp boundaries
        #   - Low temperature (<5): Smooth gradients
        scaled_sim = cosine_sim * self.temperature
        
        # Sigmoid: map to [0, 1]
        similarity_map = torch.sigmoid(scaled_sim)
        
        # Optional: Refinement
        similarity_map = self.refine(similarity_map)
        similarity_map = torch.sigmoid(similarity_map)  # Ensure [0, 1]
        
        return similarity_map
    
    def get_similarity_stats(self, similarity_map: torch.Tensor) -> dict:
        """
        Get statistics for monitoring
        
        Args:
            similarity_map: [B, 1, H, W]
            
        Returns:
            Dictionary with statistics
        """
        with torch.no_grad():
            return {
                'mean': similarity_map.mean().item(),
                'std': similarity_map.std().item(),
                'min': similarity_map.min().item(),
                'max': similarity_map.max().item(),
                'temperature': self.temperature.item()
            }


# Test module
if __name__ == '__main__':
    # Test similarity module
    sim_module = SimilarityModule(in_channels=512, embed_dim=256)
    
    # Dummy inputs
    ref = torch.randn(2, 512, 20, 20)    # Reference
    query = torch.randn(2, 512, 40, 40)  # Query (different size)
    
    similarity = sim_module(ref, query)
    
    print(f"Reference shape: {ref.shape}")
    print(f"Query shape: {query.shape}")
    print(f"Similarity shape: {similarity.shape}")
    
    stats = sim_module.get_similarity_stats(similarity)
    print(f"Similarity stats: {stats}")
    
    # Check output range
    assert similarity.min() >= 0 and similarity.max() <= 1, "Similarity should be in [0, 1]"
    print("✓ Similarity module test passed")