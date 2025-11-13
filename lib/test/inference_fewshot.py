"""
Inference Engine for Few-Shot YOLOv5

Handles:
- Loading trained model
- Processing videos with reference images
- Post-processing predictions
- Batch inference
- Result caching
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import time

from lib.models.yolov5_fewshot.backbone_model import build_fewshot_model


class FewShotYOLOInference:
    """
    Inference engine for Few-Shot YOLOv5
    
    Features:
    - Efficient batch processing
    - Reference caching (encode once, use multiple times)
    - NMS post-processing
    - Confidence filtering
    - Multi-scale testing support
    
    Args:
        checkpoint_path (str): Path to trained model checkpoint
        yolo_weights (str): Path to original YOLOv5 weights
        device (str): Device to run inference on
        conf_threshold (float): Confidence threshold
        nms_threshold (float): NMS IoU threshold
        img_size (int): Input image size
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 yolo_weights: str = 'yolov5m.pt',
                 device: str = 'cuda',
                 conf_threshold: float = 0.25,
                 nms_threshold: float = 0.45,
                 img_size: int = 640):
        
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.img_size = img_size
        
        # Load model
        print("="*70)
        print("Loading Few-Shot YOLOv5 for Inference")
        print("="*70)
        
        self.model, self.metadata = build_fewshot_model(
            yolo_weights=yolo_weights,
            checkpoint=checkpoint_path,
            freeze_backbone=True,  # Not training
            device=device
        )
        
        self.model.eval()
        
        print(f"✓ Model loaded from epoch {self.metadata['epoch']}")
        print(f"✓ Training loss: {self.metadata['loss']:.4f}")
        print(f"✓ Confidence threshold: {conf_threshold}")
        print(f"✓ NMS threshold: {nms_threshold}")
        print("="*70 + "\n")
        
        # Reference cache
        self.ref_prototype_cache = {}
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: [H, W, 3] RGB numpy array
            
        Returns:
            tensor: [1, 3, img_size, img_size]
        """
        # Resize with padding
        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Normalize and convert to tensor
        tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor, scale, (top, left)
    
    def encode_references(self, ref_image_paths: List[str], 
                         video_id: str = None) -> torch.Tensor:
        """
        Encode reference images into prototype
        
        Args:
            ref_image_paths: List of 3 reference image paths
            video_id: Optional video ID for caching
            
        Returns:
            ref_prototype: [1, C, H, W] reference prototype
        """
        # Check cache
        cache_key = tuple(sorted(ref_image_paths)) if video_id is None else video_id
        
        if cache_key in self.ref_prototype_cache:
            return self.ref_prototype_cache[cache_key]
        
        # Load and preprocess references
        ref_tensors = []
        
        for img_path in ref_image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError(f"Cannot load reference image: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor, _, _ = self.preprocess_image(img)
            ref_tensors.append(tensor)
        
        # Stack: [3, 3, H, W] -> [1, 3, 3, H, W]
        ref_batch = torch.stack([t.squeeze(0) for t in ref_tensors]).unsqueeze(0)
        ref_batch = ref_batch.to(self.device)
        
        # Encode
        with torch.no_grad():
            ref_prototype, _ = self.model.encode_references(ref_batch)
        
        # Cache
        self.ref_prototype_cache[cache_key] = ref_prototype
        
        return ref_prototype
    
    def postprocess_predictions(self, 
                                predictions: torch.Tensor,
                                orig_shape: Tuple[int, int],
                                scale: float,
                                offset: Tuple[int, int]) -> List[Dict]:
        """
        Post-process model predictions
        
        Args:
            predictions: [1, num_anchors*(5+num_classes), H, W]
            orig_shape: Original image shape (H, W)
            scale: Resize scale factor
            offset: Padding offset (top, left)
            
        Returns:
            detections: List of detection dicts
        """
        # Parse predictions
        # This is simplified - implement proper YOLO decoding
        
        B, _, H, W = predictions.shape
        
        # Reshape: [1, 3*6, H, W] -> [1, 3, 6, H, W] -> [1, 3, H, W, 6]
        pred = predictions.view(B, 3, 6, H, W).permute(0, 1, 3, 4, 2).contiguous()
        
        # Extract components
        pred_xy = pred[..., 0:2].sigmoid()  # [1, 3, H, W, 2]
        pred_wh = pred[..., 2:4]            # [1, 3, H, W, 2]
        pred_obj = pred[..., 4].sigmoid()   # [1, 3, H, W]
        pred_cls = pred[..., 5].sigmoid()   # [1, 3, H, W]
        
        # Generate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=predictions.device),
            torch.arange(W, device=predictions.device),
            indexing='ij'
        )
        
        grid = torch.stack([grid_x, grid_y], dim=-1).float()  # [H, W, 2]
        
        # Decode boxes
        stride = self.img_size / H
        
        # Center points
        pred_xy = (pred_xy + grid.unsqueeze(0).unsqueeze(0)) * stride  # [1, 3, H, W, 2]
        
        # Width/height
        pred_wh = torch.exp(pred_wh) * stride  # Simplified, should use anchors
        
        # Convert to [x1, y1, x2, y2]
        pred_x1y1 = pred_xy - pred_wh / 2
        pred_x2y2 = pred_xy + pred_wh / 2
        
        # Confidence = objectness * class_prob
        confidence = pred_obj * pred_cls  # [1, 3, H, W]
        
        # Flatten and filter by confidence
        num_anchors = 3
        all_boxes = []
        all_scores = []
        
        for anchor_idx in range(num_anchors):
            boxes = torch.cat([
                pred_x1y1[0, anchor_idx].reshape(-1, 2),
                pred_x2y2[0, anchor_idx].reshape(-1, 2)
            ], dim=1)  # [H*W, 4]
            
            scores = confidence[0, anchor_idx].reshape(-1)  # [H*W]
            
            # Filter by threshold
            mask = scores > self.conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
        
        if len(all_boxes) == 0:
            return []
        
        # Concatenate all anchors
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        
        # NMS
        keep_indices = self.nms(all_boxes, all_scores, self.nms_threshold)
        
        # Get final detections
        final_boxes = all_boxes[keep_indices].cpu().numpy()
        final_scores = all_scores[keep_indices].cpu().numpy()
        
        # Transform back to original image coordinates
        detections = []
        
        top_offset, left_offset = offset
        
        for box, score in zip(final_boxes, final_scores):
            x1, y1, x2, y2 = box
            
            # Remove padding
            x1 = (x1 - left_offset) / scale
            y1 = (y1 - top_offset) / scale
            x2 = (x2 - left_offset) / scale
            y2 = (y2 - top_offset) / scale
            
            # Clip to image bounds
            x1 = max(0, min(x1, orig_shape[1]))
            y1 = max(0, min(y1, orig_shape[0]))
            x2 = max(0, min(x2, orig_shape[1]))
            y2 = max(0, min(y2, orig_shape[0]))
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(score)
            })
        
        return detections
    
    @staticmethod
    def nms(boxes: torch.Tensor, scores: torch.Tensor, 
            iou_threshold: float) -> torch.Tensor:
        """
        Non-Maximum Suppression
        
        Args:
            boxes: [N, 4] in [x1, y1, x2, y2] format
            scores: [N] confidence scores
            iou_threshold: IoU threshold
            
        Returns:
            keep: Indices of kept boxes
        """
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by score
        _, order = scores.sort(descending=True)
        
        keep = []
        
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0].item()
            keep.append(i)
            
            # Compute IoU with remaining boxes
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IoU < threshold
            mask = iou <= iou_threshold
            order = order[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long)
    
    @torch.no_grad()
    def detect_image(self, 
                     image: np.ndarray,
                     ref_prototype: torch.Tensor) -> List[Dict]:
        """
        Detect objects in a single image
        
        Args:
            image: [H, W, 3] RGB numpy array
            ref_prototype: [1, C, H, W] pre-computed reference prototype
            
        Returns:
            detections: List of detection dicts
        """
        orig_shape = image.shape[:2]
        
        # Preprocess
        img_tensor, scale, offset = self.preprocess_image(image)
        img_tensor = img_tensor.to(self.device)
        
        # Forward pass
        outputs = self.model(
            query_images=img_tensor,
            ref_prototype=ref_prototype,
            return_features=False
        )
        
        # Post-process
        detections = self.postprocess_predictions(
            predictions=outputs['predictions'],
            orig_shape=orig_shape,
            scale=scale,
            offset=offset
        )
        
        return detections
    
    def detect_video(self,
                     video_path: str,
                     ref_image_paths: List[str],
                     video_id: str = None,
                     save_video: bool = False,
                     output_video_path: str = None) -> List[Dict]:
        """
        Detect objects in video
        
        Args:
            video_path: Path to video file
            ref_image_paths: List of 3 reference image paths
            video_id: Optional video ID for caching
            save_video: Whether to save visualization
            output_video_path: Path to save output video
            
        Returns:
            frame_detections: List of detections per frame
        """
        # Encode references once
        ref_prototype = self.encode_references(ref_image_paths, video_id)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        writer = None
        if save_video and output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {total_frames} frames @ {fps} FPS")
        
        frame_detections = []
        frame_id = 0
        
        # Progress bar
        pbar = tqdm(total=total_frames, desc="Detecting")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect
            detections = self.detect_image(frame_rgb, ref_prototype)
            
            # Store detections
            for det in detections:
                frame_detections.append({
                    'frame': frame_id,
                    'x1': det['bbox'][0],
                    'y1': det['bbox'][1],
                    'x2': det['bbox'][2],
                    'y2': det['bbox'][3],
                    'confidence': det['confidence']
                })
            
            # Visualize
            if writer:
                for det in detections:
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    conf = det['confidence']
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"{conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2
                    )
                
                writer.write(frame)
            
            frame_id += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        if writer:
            writer.release()
            print(f"✓ Saved visualization to {output_video_path}")
        
        return frame_detections


# Batch inference for multiple videos
class BatchInference:
    """
    Batch inference handler for processing multiple videos efficiently
    """
    
    def __init__(self, inference_engine: FewShotYOLOInference):
        self.engine = inference_engine
    
    def process_test_set(self,
                        test_dir: str,
                        output_json: str,
                        save_visualizations: bool = False,
                        vis_dir: str = None) -> Dict:
        """
        Process entire test set
        
        Args:
            test_dir: Directory containing test videos
            output_json: Path to save results JSON
            save_visualizations: Whether to save visualizations
            vis_dir: Directory to save visualizations
            
        Returns:
            results: Dictionary with all results
        """
        test_path = Path(test_dir)
        
        # Find all video folders
        video_folders = sorted([f for f in test_path.iterdir() if f.is_dir()])
        
        print(f"\n{'='*70}")
        print(f"Processing {len(video_folders)} test videos")
        print(f"{'='*70}\n")
        
        all_results = []
        
        # Create visualization directory
        if save_visualizations and vis_dir:
            Path(vis_dir).mkdir(parents=True, exist_ok=True)
        
        # Process each video
        for video_folder in tqdm(video_folders, desc="Videos"):
            video_id = video_folder.name
            video_path = video_folder / 'drone_video.mp4'
            ref_dir = video_folder / 'object_images'
            
            if not video_path.exists():
                print(f"Warning: Video not found: {video_path}")
                continue
            
            # Reference images
            ref_paths = [
                str(ref_dir / f'img_{i}.jpg')
                for i in range(1, 4)
            ]
            
            # Check all exist
            if not all(Path(p).exists() for p in ref_paths):
                print(f"Warning: Missing references for {video_id}")
                continue
            
            # Output video path
            output_video = None
            if save_visualizations:
                output_video = str(Path(vis_dir) / f"{video_id}.mp4")
            
            # Detect
            start_time = time.time()
            
            detections = self.engine.detect_video(
                video_path=str(video_path),
                ref_image_paths=ref_paths,
                video_id=video_id,
                save_video=save_visualizations,
                output_video_path=output_video
            )
            
            elapsed = time.time() - start_time
            
            # Format result
            result = {
                'video_id': video_id,
                'detections': [{'bboxes': detections}] if detections else [],
                'num_detections': len(detections),
                'processing_time': elapsed
            }
            
            all_results.append(result)
            
            print(f"  {video_id}: {len(detections)} detections in {elapsed:.2f}s")
        
        # Save results
        with open(output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Saved results to {output_json}")
        print(f"✓ Processed {len(all_results)} videos")
        
        return {
            'results': all_results,
            'total_videos': len(all_results),
            'output_json': output_json
        }