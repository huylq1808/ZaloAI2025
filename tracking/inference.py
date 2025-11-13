"""
Inference Entry Point for Few-Shot YOLOv5

Usage:
    python tracking/inference.py \
        --checkpoint checkpoints/fewshot_yolo/best.pt \
        --test_dir data/test/samples \
        --output predictions.json \
        --save_vis \
        --vis_dir visualizations
"""

import sys
sys.path.append('.')

import argparse
from pathlib import Path

from lib.test.inference_fewshot import FewShotYOLOInference, BatchInference


def main():
    parser = argparse.ArgumentParser(description='Few-Shot YOLOv5 Inference')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--yolo_weights', type=str, default='yolov5m.pt',
                       help='Path to original YOLOv5 weights')
    
    # Data
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing test videos')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Output JSON file for predictions')
    
    # Inference settings
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.45,
                       help='NMS IoU threshold')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Input image size')
    
    # Visualization
    parser.add_argument('--save_vis', action='store_true',
                       help='Save visualization videos')
    parser.add_argument('--vis_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Few-Shot YOLOv5 Inference")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test Dir: {args.test_dir}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("="*70 + "\n")
    
    # Build inference engine
    engine = FewShotYOLOInference(
        checkpoint_path=args.checkpoint,
        yolo_weights=args.yolo_weights,
        device=args.device,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        img_size=args.img_size
    )
    
    # Batch inference
    batch_processor = BatchInference(engine)
    
    # Process test set
    results = batch_processor.process_test_set(
        test_dir=args.test_dir,
        output_json=args.output,
        save_visualizations=args.save_vis,
        vis_dir=args.vis_dir if args.save_vis else None
    )
    
    print("\n" + "="*70)
    print("Inference Complete!")
    print(f"Total videos: {results['total_videos']}")
    print(f"Results saved to: {results['output_json']}")
    print("="*70)


if __name__ == '__main__':
    main()