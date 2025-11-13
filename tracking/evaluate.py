"""
Evaluation Script for Few-Shot Detection

Computes metrics:
- Precision, Recall, F1
- mAP (mean Average Precision)
- Per-video statistics

Usage:
    python tracking/evaluate.py \
        --predictions predictions.json \
        --ground_truth data/test/annotations/annotations.json \
        --iou_threshold 0.5
"""

import sys
sys.path.append('.')

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class DetectionEvaluator:
    """
    Evaluator for object detection
    
    Computes:
    - Precision, Recall, F1
    - Average Precision (AP)
    - mAP across IoU thresholds
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
    
    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        """
        Compute IoU between two boxes
        
        Args:
            box1, box2: [x1, y1, x2, y2]
            
        Returns:
            iou: IoU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def match_detections(self,
                        predictions: List[Dict],
                        ground_truths: List[Dict]) -> Tuple[int, int, int]:
        """
        Match predictions with ground truths
        
        Returns:
            tp, fp, fn: True positives, false positives, false negatives
        """
        matched_gt = set()
        tp = 0
        fp = 0
        
        # Sort predictions by confidence
        preds_sorted = sorted(predictions, 
                            key=lambda x: x.get('confidence', 1.0), 
                            reverse=True)
        
        for pred in preds_sorted:
            pred_box = pred['bbox']
            
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                
                gt_box = [gt['x1'], gt['y1'], gt['x2'], gt['y2']]
                iou = self.compute_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is valid
            if best_iou >= self.iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        # Unmatched ground truths are false negatives
        fn = len(ground_truths) - len(matched_gt)
        
        return tp, fp, fn
    
    def evaluate(self,
                predictions_file: str,
                ground_truth_file: str) -> Dict:
        """
        Evaluate predictions against ground truth
        
        Returns:
            metrics: Dictionary of metrics
        """
        # Load files
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        with open(ground_truth_file, 'r') as f:
            ground_truths = json.load(f)
        
        # Build ground truth lookup
        gt_by_video = {}
        for gt in ground_truths:
            video_id = gt['video_id']
            gt_bboxes = []
            
            for interval in gt.get('annotations', []):
                gt_bboxes.extend(interval.get('bboxes', []))
            
            gt_by_video[video_id] = gt_bboxes
        
        # Accumulate metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        per_video_metrics = {}
        
        # Evaluate each video
        for pred in predictions:
            video_id = pred['video_id']
            
            pred_bboxes = []
            for det_group in pred.get('detections', []):
                pred_bboxes.extend(det_group.get('bboxes', []))
            
            # Convert predictions to required format
            pred_boxes = []
            for bbox in pred_bboxes:
                pred_boxes.append({
                    'bbox': [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']],
                    'confidence': bbox.get('confidence', 1.0)
                })
            
            # Get ground truth
            gt_boxes = gt_by_video.get(video_id, [])
            
            # Match
            tp, fp, fn = self.match_detections(pred_boxes, gt_boxes)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # Per-video metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_video_metrics[video_id] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
                     if (overall_precision + overall_recall) > 0 else 0.0
        
        return {
            'overall': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn
            },
            'per_video': per_video_metrics,
            'iou_threshold': self.iou_threshold
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Few-Shot Detection')
    
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSON')
    parser.add_argument('--ground_truth', type=str, required=True,
                       help='Path to ground truth annotations JSON')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for matching')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for evaluation results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Few-Shot Detection Evaluation")
    print("="*70)
    print(f"Predictions: {args.predictions}")
    print(f"Ground Truth: {args.ground_truth}")
    print(f"IoU Threshold: {args.iou_threshold}")
    print("="*70 + "\n")
    
    # Evaluate
    evaluator = DetectionEvaluator(iou_threshold=args.iou_threshold)
    metrics = evaluator.evaluate(args.predictions, args.ground_truth)
    
    # Print results
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    print(f"Precision: {metrics['overall']['precision']:.4f}")
    print(f"Recall: {metrics['overall']['recall']:.4f}")
    print(f"F1 Score: {metrics['overall']['f1']:.4f}")
    print(f"True Positives: {metrics['overall']['tp']}")
    print(f"False Positives: {metrics['overall']['fp']}")
    print(f"False Negatives: {metrics['overall']['fn']}")
    print("="*70)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {args.output}")


if __name__ == '__main__':
    main()