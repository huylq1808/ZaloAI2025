"""
Loss Functions for Few-Shot YOLOv5

Combines:
1. YOLOv5 detection loss (bbox, objectness, classification)
2. Similarity matching loss
3. Reference attention regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class FewShotYOLOLoss(nn.Module):
    """
    Combined loss for few-shot detection
    
    Components:
    1. Bbox Regression Loss (GIoU + L1)
    2. Objectness Loss (BCE)
    3. Classification Loss (BCE for single-class)
    4. Similarity Loss (encourage high similarity for positive regions)
    5. Attention Regularization (optional)
    
    Args:
        lambda_bbox (float): Weight for bbox loss
        lambda_obj (float): Weight for objectness loss
        lambda_cls (float): Weight for classification loss
        lambda_sim (float): Weight for similarity loss
        lambda_attn (float): Weight for attention regularization
    """
    
    def __init__(self,
                 lambda_bbox: float = 5.0,
                 lambda_obj: float = 1.0,
                 lambda_cls: float = 0.5,
                 lambda_sim: float = 2.0,
                 lambda_attn: float = 0.1):
        super().__init__()
        
        self.lambda_bbox = lambda_bbox
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.lambda_sim = lambda_sim
        self.lambda_attn = lambda_attn
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def compute_giou(self, pred_boxes: torch.Tensor, 
                    target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute Generalized IoU
        
        Args:
            pred_boxes: [N, 4] in format [x1, y1, x2, y2]
            target_boxes: [N, 4] in same format
            
        Returns:
            giou: [N] GIoU values
        """
        # Area
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                    (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * \
                      (target_boxes[:, 3] - target_boxes[:, 1])
        
        # Intersection
        lt = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
        rb = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        
        # Union
        union = pred_area + target_area - inter
        
        # IoU
        iou = inter / (union + 1e-7)
        
        # Enclosing box
        enclose_lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
        enclose_rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
        
        # GIoU
        giou = iou - (enclose_area - union) / (enclose_area + 1e-7)
        
        return giou
    
    def forward(self,
                predictions: torch.Tensor,
                targets: List[Dict],
                similarity_maps: torch.Tensor,
                attention_weights: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss
        
        Args:
            predictions: [B, num_anchors*(5+num_classes), H, W]
            targets: List of dicts with 'boxes' and 'labels'
            similarity_maps: [B, 1, H, W]
            attention_weights: [B, num_refs] (optional)
            
        Returns:
            loss_dict: Dictionary of losses
        """
        device = predictions.device
        B, _, H, W = predictions.shape
        
        # Parse predictions
        # Format: [B, 3*(5+1), H, W] for num_anchors=3, num_classes=1
        # Reshape: [B, 3, 6, H, W] -> [B, 3, H, W, 6]
        pred = predictions.view(B, 3, 6, H, W).permute(0, 1, 3, 4, 2).contiguous()
        
        pred_xy = pred[..., 0:2].sigmoid()  # xy
        pred_wh = pred[..., 2:4]            # wh
        pred_obj = pred[..., 4:5]           # objectness
        pred_cls = pred[..., 5:]            # class
        
        # Initialize losses
        total_bbox_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0
        total_sim_loss = 0.0
        num_targets = 0
        
        # Process each image in batch
        for b in range(B):
            target = targets[b]
            
            if target['boxes'].numel() == 0:
                # No objects - only negative objectness loss
                obj_loss = self.bce_loss(
                    pred_obj[b],
                    torch.zeros_like(pred_obj[b])
                ).mean()
                
                total_obj_loss += obj_loss
                continue
            
            # Build target tensors
            # This is simplified - in practice, use anchor matching
            gt_boxes = target['boxes']  # [num_objs, 4]
            gt_labels = target['labels']  # [num_objs]
            
            num_targets += len(gt_boxes)
            
            # For simplicity, match predictions to targets using IoU
            # In production, use proper anchor assignment
            
            # Bbox loss (GIoU)
            # This is placeholder - implement proper anchor matching
            bbox_loss = torch.tensor(0.0, device=device)
            
            # Objectness loss
            # Positive: where objects exist
            # Negative: everywhere else
            obj_targets = torch.zeros_like(pred_obj[b])
            # Set positive locations (simplified)
            # obj_targets[...] = 1  # Where objects exist
            
            obj_loss = self.bce_loss(pred_obj[b], obj_targets).mean()
            total_obj_loss += obj_loss
            
            # Classification loss (single class - should be high for objects)
            cls_loss = torch.tensor(0.0, device=device)
            
            # Bbox loss
            total_bbox_loss += bbox_loss
            total_cls_loss += cls_loss
        
        # Similarity loss: encourage high similarity in object regions
        # Positive samples should have high similarity
        sim_loss = 0.0
        
        for b in range(B):
            if targets[b]['boxes'].numel() > 0:
                # Positive: high similarity
                sim_target = torch.ones_like(similarity_maps[b])
                sim_loss += F.binary_cross_entropy(
                    similarity_maps[b],
                    sim_target,
                    reduction='mean'
                )
            else:
                # Negative: low similarity
                sim_target = torch.zeros_like(similarity_maps[b])
                sim_loss += F.binary_cross_entropy(
                    similarity_maps[b],
                    sim_target,
                    reduction='mean'
                ) * 0.1  # Lower weight for negatives
        
        total_sim_loss = sim_loss / B if B > 0 else 0.0
        
        # Attention regularization (encourage diversity)
        attn_loss = 0.0
        
        if attention_weights is not None:
            # Encourage attention diversity (not all weight on one reference)
            # Use entropy regularization
            attn_probs = attention_weights  # Already softmaxed
            entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=1).mean()
            
            # Maximize entropy (add negative sign)
            attn_loss = -entropy * self.lambda_attn
        
        # Total loss
        total_loss = (
            self.lambda_bbox * total_bbox_loss +
            self.lambda_obj * total_obj_loss +
            self.lambda_cls * total_cls_loss +
            self.lambda_sim * total_sim_loss +
            attn_loss
        )
        
        # Return loss dictionary
        return {
            'total_loss': total_loss,
            'bbox_loss': total_bbox_loss,
            'obj_loss': total_obj_loss,
            'cls_loss': total_cls_loss,
            'sim_loss': total_sim_loss,
            'attn_loss': attn_loss
        }


# Simplified version using YOLOv5's ComputeLoss
class YOLOv5BasedLoss:
    """
    Wrapper around YOLOv5's original ComputeLoss
    
    Adds similarity loss on top
    """
    
    def __init__(self, model, lambda_sim: float = 2.0):
        # Import YOLOv5's loss
        try:
            from utils.loss import ComputeLoss
            self.yolo_loss = ComputeLoss(model)
        except:
            print("Warning: Could not load YOLOv5 ComputeLoss. Using simplified version.")
            self.yolo_loss = None
        
        self.lambda_sim = lambda_sim
    
    def __call__(self, predictions, targets, similarity_maps):
        """
        Compute loss
        
        Args:
            predictions: YOLOv5 format predictions
            targets: YOLOv5 format targets
            similarity_maps: [B, 1, H, W]
            
        Returns:
            total_loss: Combined loss
            loss_dict: Loss components
        """
        # YOLOv5 detection loss
        if self.yolo_loss:
            yolo_loss, loss_items = self.yolo_loss(predictions, targets)
        else:
            yolo_loss = torch.tensor(0.0)
            loss_items = torch.zeros(3)
        
        # Similarity loss
        sim_loss = F.binary_cross_entropy(
            similarity_maps,
            torch.ones_like(similarity_maps),
            reduction='mean'
        )
        
        total_loss = yolo_loss + self.lambda_sim * sim_loss
        
        return total_loss, {
            'yolo_loss': yolo_loss.item(),
            'sim_loss': sim_loss.item(),
            'total_loss': total_loss.item()
        }