"""
Visualization Tools for Few-Shot Detection

Features:
- Draw bounding boxes on images/videos
- Visualize attention weights
- Visualize similarity maps
- Create comparison videos (GT vs Predictions)
- Generate analysis plots
"""

import sys
sys.path.append('.')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch


class DetectionVisualizer:
    """
    Visualizer for detection results
    
    Features:
    - Draw predictions and ground truth
    - Color-coded confidence scores
    - Side-by-side comparisons
    - Attention weight visualization
    """
    
    def __init__(self, 
                 font_scale: float = 0.5,
                 thickness: int = 2,
                 colormap: str = 'viridis'):
        self.font_scale = font_scale
        self.thickness = thickness
        self.colormap = plt.get_cmap(colormap)
    
    def draw_boxes(self,
                   image: np.ndarray,
                   boxes: List[Dict],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   label_prefix: str = "",
                   show_confidence: bool = True) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: [H, W, 3] BGR image
            boxes: List of box dicts with 'bbox' and optional 'confidence'
            color: Box color in BGR
            label_prefix: Prefix for labels
            show_confidence: Whether to show confidence scores
            
        Returns:
            annotated_image: Image with boxes drawn
        """
        img = image.copy()
        
        for box in boxes:
            bbox = box['bbox']
            conf = box.get('confidence', 1.0)
            
            # Convert to int
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color based on confidence if available
            if show_confidence and 'confidence' in box:
                # Map confidence to color
                color_intensity = int(conf * 255)
                box_color = (0, color_intensity, 255 - color_intensity)
            else:
                box_color = color
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, self.thickness)
            
            # Draw label
            if show_confidence:
                label = f"{label_prefix}{conf:.2f}"
            else:
                label = label_prefix
            
            if label:
                # Background for text
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 
                    self.font_scale, self.thickness
                )
                
                cv2.rectangle(
                    img, 
                    (x1, y1 - text_height - 4),
                    (x1 + text_width, y1),
                    box_color, -1
                )
                
                # Text
                cv2.putText(
                    img, label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    (255, 255, 255),
                    self.thickness
                )
        
        return img
    
    def create_comparison_frame(self,
                               image: np.ndarray,
                               predictions: List[Dict],
                               ground_truths: List[Dict]) -> np.ndarray:
        """
        Create side-by-side comparison
        
        Args:
            image: Original image
            predictions: Predicted boxes
            ground_truths: Ground truth boxes
            
        Returns:
            comparison: Side-by-side image
        """
        # Draw predictions (green)
        pred_img = self.draw_boxes(
            image, predictions,
            color=(0, 255, 0),
            label_prefix="Pred: ",
            show_confidence=True
        )
        
        # Draw ground truths (blue)
        gt_img = self.draw_boxes(
            image, ground_truths,
            color=(255, 0, 0),
            label_prefix="GT",
            show_confidence=False
        )
        
        # Concatenate horizontally
        comparison = np.hstack([pred_img, gt_img])
        
        # Add labels
        h, w = comparison.shape[:2]
        cv2.putText(
            comparison, "Predictions",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2
        )
        
        cv2.putText(
            comparison, "Ground Truth",
            (w // 2 + 20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 0, 0), 2
        )
        
        return comparison
    
    def visualize_similarity_map(self,
                                image: np.ndarray,
                                similarity_map: torch.Tensor,
                                alpha: float = 0.6) -> np.ndarray:
        """
        Overlay similarity map on image
        
        Args:
            image: [H, W, 3] BGR image
            similarity_map: [1, H, W] or [H, W] similarity scores
            alpha: Overlay transparency
            
        Returns:
            overlay: Image with similarity heatmap
        """
        # Convert similarity to numpy
        if isinstance(similarity_map, torch.Tensor):
            sim = similarity_map.cpu().numpy()
            if sim.ndim == 3:
                sim = sim[0]  # Remove batch/channel dimension
        else:
            sim = similarity_map
        
        # Resize to match image
        h, w = image.shape[:2]
        sim_resized = cv2.resize(sim, (w, h))
        
        # Normalize to [0, 255]
        sim_norm = (sim_resized * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(sim_norm, cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    
    def visualize_attention_weights(self,
                                   ref_images: List[np.ndarray],
                                   attention_weights: np.ndarray,
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize attention weights on reference images
        
        Args:
            ref_images: List of 3 reference images
            attention_weights: [3] attention scores
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (img, weight) in enumerate(zip(ref_images, attention_weights)):
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Plot
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"Reference {i+1}\nWeight: {weight:.3f}", 
                            fontsize=12, fontweight='bold')
            axes[i].axis('off')
            
            # Add border based on weight
            border_width = int(weight * 10)
            for spine in axes[i].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(border_width)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved attention visualization to {save_path}")
        
        plt.show()
    
    def create_comparison_video(self,
                               video_path: str,
                               predictions: List[Dict],
                               ground_truths: List[Dict],
                               output_path: str) -> None:
        """
        Create comparison video with predictions and GT
        
        Args:
            video_path: Input video path
            predictions: Frame-level predictions
            ground_truths: Frame-level ground truths
            output_path: Output video path
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create writer (double width for side-by-side)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        # Group predictions and GT by frame
        pred_by_frame = {}
        gt_by_frame = {}
        
        for pred in predictions:
            frame_id = pred['frame']
            if frame_id not in pred_by_frame:
                pred_by_frame[frame_id] = []
            pred_by_frame[frame_id].append({
                'bbox': [pred['x1'], pred['y1'], pred['x2'], pred['y2']],
                'confidence': pred.get('confidence', 1.0)
            })
        
        for gt in ground_truths:
            frame_id = gt['frame']
            if frame_id not in gt_by_frame:
                gt_by_frame[frame_id] = []
            gt_by_frame[frame_id].append({
                'bbox': [gt['x1'], gt['y1'], gt['x2'], gt['y2']]
            })
        
        # Process frames
        frame_id = 0
        pbar = tqdm(total=total_frames, desc="Creating comparison video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get predictions and GT for this frame
            frame_preds = pred_by_frame.get(frame_id, [])
            frame_gts = gt_by_frame.get(frame_id, [])
            
            # Create comparison
            comparison = self.create_comparison_frame(frame, frame_preds, frame_gts)
            
            # Write
            writer.write(comparison)
            
            frame_id += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        writer.release()
        
        print(f"✓ Saved comparison video to {output_path}")


class AnalysisPlotter:
    """
    Create analysis plots for evaluation results
    """
    
    @staticmethod
    def plot_confidence_distribution(predictions_file: str,
                                     save_path: str = None) -> None:
        """
        Plot confidence score distribution
        
        Args:
            predictions_file: Path to predictions JSON
            save_path: Path to save plot
        """
        # Load predictions
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
        
        # Extract confidences
        confidences = []
        for pred in predictions:
            for det_group in pred.get('detections', []):
                for bbox in det_group.get('bboxes', []):
                    conf = bbox.get('confidence', 1.0)
                    confidences.append(conf)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Statistics
        mean_conf = np.mean(confidences)
        median_conf = np.median(confidences)
        
        plt.axvline(mean_conf, color='red', linestyle='--', 
                   label=f'Mean: {mean_conf:.3f}')
        plt.axvline(median_conf, color='green', linestyle='--',
                   label=f'Median: {median_conf:.3f}')
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved confidence distribution to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_per_video_metrics(eval_results_file: str,
                               save_path: str = None) -> None:
        """
        Plot per-video metrics (Precision, Recall, F1)
        
        Args:
            eval_results_file: Path to evaluation results JSON
            save_path: Path to save plot
        """
        # Load results
        with open(eval_results_file, 'r') as f:
            results = json.load(f)
        
        per_video = results['per_video']
        
        # Extract data
        video_ids = list(per_video.keys())
        precisions = [per_video[vid]['precision'] for vid in video_ids]
        recalls = [per_video[vid]['recall'] for vid in video_ids]
        f1_scores = [per_video[vid]['f1'] for vid in video_ids]
        
        # Plot
        x = np.arange(len(video_ids))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(max(12, len(video_ids) * 0.5), 6))
        
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)
        
        ax.set_xlabel('Video ID', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Video Detection Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(video_ids, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved per-video metrics to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(eval_results_file: str,
                                    predictions_file: str,
                                    save_path: str = None) -> None:
        """
        Plot Precision-Recall curve at different confidence thresholds
        
        Args:
            eval_results_file: Path to evaluation results JSON
            predictions_file: Path to predictions JSON
            save_path: Path to save plot
        """
        # This would require re-evaluating at different thresholds
        # For now, plotting a simplified version
        
        # Load results
        with open(eval_results_file, 'r') as f:
            results = json.load(f)
        
        overall = results['overall']
        
        # Single point on PR curve
        precision = overall['precision']
        recall = overall['recall']
        
        plt.figure(figsize=(8, 8))
        plt.scatter([recall], [precision], s=100, c='red', 
                   label=f'Current (P={precision:.3f}, R={recall:.3f})', zorder=5)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved PR curve to {save_path}")
        
        plt.show()
    
    @staticmethod
    def create_summary_report(eval_results_file: str,
                             output_html: str = 'evaluation_report.html') -> None:
        """
        Create HTML summary report
        
        Args:
            eval_results_file: Path to evaluation results JSON
            output_html: Path to save HTML report
        """
        # Load results
        with open(eval_results_file, 'r') as f:
            results = json.load(f)
        
        overall = results['overall']
        per_video = results['per_video']
        
        # HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Few-Shot Detection Evaluation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 30px;
                }}
                .metric {{
                    display: inline-block;
                    margin: 20px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    min-width: 200px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 36px;
                    font-weight: bold;
                    color: #4CAF50;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #777;
                    margin-top: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Few-Shot Detection Evaluation Report</h1>
                
                <h2>Overall Metrics</h2>
                <div>
                    <div class="metric">
                        <div class="metric-value">{overall['precision']:.3f}</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{overall['recall']:.3f}</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{overall['f1']:.3f}</div>
                        <div class="metric-label">F1 Score</div>
                    </div>
                </div>
                
                <div>
                    <div class="metric">
                        <div class="metric-value">{overall['tp']}</div>
                        <div class="metric-label">True Positives</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{overall['fp']}</div>
                        <div class="metric-label">False Positives</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{overall['fn']}</div>
                        <div class="metric-label">False Negatives</div>
                    </div>
                </div>
                
                <h2>Per-Video Results</h2>
                <table>
                    <tr>
                        <th>Video ID</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>TP</th>
                        <th>FP</th>
                        <th>FN</th>
                    </tr>
        """
        
        # Add per-video rows
        for video_id, metrics in per_video.items():
            html += f"""
                    <tr>
                        <td>{video_id}</td>
                        <td>{metrics['precision']:.3f}</td>
                        <td>{metrics['recall']:.3f}</td>
                        <td>{metrics['f1']:.3f}</td>
                        <td>{metrics['tp']}</td>
                        <td>{metrics['fp']}</td>
                        <td>{metrics['fn']}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Save
        with open(output_html, 'w') as f:
            f.write(html)
        
        print(f"✓ Saved HTML report to {output_html}")


# CLI for visualization
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Few-Shot Detection Results')
    
    subparsers = parser.add_subparsers(dest='command', help='Visualization command')
    
    # Comparison video
    video_parser = subparsers.add_parser('video', help='Create comparison video')
    video_parser.add_argument('--video', required=True, help='Input video path')
    video_parser.add_argument('--predictions', required=True, help='Predictions JSON')
    video_parser.add_argument('--ground_truth', required=True, help='Ground truth JSON')
    video_parser.add_argument('--output', required=True, help='Output video path')
    
    # Confidence distribution
    conf_parser = subparsers.add_parser('confidence', help='Plot confidence distribution')
    conf_parser.add_argument('--predictions', required=True, help='Predictions JSON')
    conf_parser.add_argument('--output', default='confidence_dist.png', help='Output plot')
    
    # Per-video metrics
    metrics_parser = subparsers.add_parser('metrics', help='Plot per-video metrics')
    metrics_parser.add_argument('--eval_results', required=True, help='Evaluation results JSON')
    metrics_parser.add_argument('--output', default='per_video_metrics.png', help='Output plot')
    
    # HTML report
    report_parser = subparsers.add_parser('report', help='Create HTML report')
    report_parser.add_argument('--eval_results', required=True, help='Evaluation results JSON')
    report_parser.add_argument('--output', default='evaluation_report.html', help='Output HTML')
    
    args = parser.parse_args()
    
    if args.command == 'video':
        # Load data
        with open(args.predictions, 'r') as f:
            predictions = json.load(f)
        
        with open(args.ground_truth, 'r') as f:
            ground_truth = json.load(f)
        
        # Create visualizer
        visualizer = DetectionVisualizer()
        
        # Find video in predictions
        video_id = Path(args.video).parent.name
        
        # Extract detections for this video
        video_preds = []
        for pred in predictions:
            if pred['video_id'] == video_id:
                for det_group in pred.get('detections', []):
                    video_preds.extend(det_group.get('bboxes', []))
        
        # Extract GT
        video_gts = []
        for gt in ground_truth:
            if gt['video_id'] == video_id:
                for interval in gt.get('annotations', []):
                    video_gts.extend(interval.get('bboxes', []))
        
        # Create comparison
        visualizer.create_comparison_video(
            video_path=args.video,
            predictions=video_preds,
            ground_truths=video_gts,
            output_path=args.output
        )
    
    elif args.command == 'confidence':
        plotter = AnalysisPlotter()
        plotter.plot_confidence_distribution(args.predictions, args.output)
    
    elif args.command == 'metrics':
        plotter = AnalysisPlotter()
        plotter.plot_per_video_metrics(args.eval_results, args.output)
    
    elif args.command == 'report':
        plotter = AnalysisPlotter()
        plotter.create_summary_report(args.eval_results, args.output)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()