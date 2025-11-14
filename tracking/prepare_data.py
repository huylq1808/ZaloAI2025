"""
Prepare data for Few-Shot YOLOv5 training

Converts your raw data to the required format
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2

def prepare_training_data(raw_data_dir: str, output_dir: str, split_ratio: float = 0.8):
    """
    Prepare and split data into train/val
    
    Args:
        raw_data_dir: Directory containing raw videos and annotations
        output_dir: Output directory for processed data
        split_ratio: Train/val split ratio
    """
    raw_path = Path(raw_data_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val']:
        (output_path / split / 'samples').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'annotations').mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    ann_file = raw_path / 'annotations.json'
    with open(ann_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Split data
    n_train = int(len(annotations) * split_ratio)
    train_annotations = annotations[:n_train]
    val_annotations = annotations[n_train:]
    
    # Process train split
    print("Processing training data...")
    process_split(raw_path, output_path / 'train', train_annotations)
    
    # Process val split
    print("Processing validation data...")
    process_split(raw_path, output_path / 'val', val_annotations)
    
    print(f"\n✓ Data preparation complete!")
    print(f"  Train videos: {len(train_annotations)}")
    print(f"  Val videos: {len(val_annotations)}")

def process_split(raw_path: Path, output_path: Path, annotations: list):
    """Process single split (train or val)"""
    
    split_annotations = []
    
    for ann in tqdm(annotations):
        video_id = ann['video_id']
        
        # Create video directory
        video_dir = output_path / 'samples' / video_id
        (video_dir / 'object_images').mkdir(parents=True, exist_ok=True)
        (video_dir / 'frames').mkdir(parents=True, exist_ok=True)
        
        # Copy reference images
        raw_ref_dir = raw_path / 'samples' / video_id / 'object_images'
        for i in range(1, 4):  # 3 reference images
            src = raw_ref_dir / f'img_{i}.jpg'
            dst = video_dir / 'object_images' / f'img_{i}.jpg'
            
            if src.exists():
                shutil.copy2(src, dst)
        
        # Extract and save frames
        video_file = raw_path / 'samples' / video_id / 'drone_video.mp4'
        if video_file.exists():
            extract_frames(video_file, video_dir / 'frames')
        
        split_annotations.append(ann)
    
    # Save annotations
    ann_output = output_path / 'annotations' / 'annotations.json'
    with open(ann_output, 'w', encoding='utf-8') as f:
        json.dump(split_annotations, f, indent=2, ensure_ascii=False)

def extract_frames(video_path: Path, output_dir: Path):
    """Extract all frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        frame_path = output_dir / f'frame_{frame_id:06d}.jpg'
        cv2.imwrite(str(frame_path), frame)
        
        frame_id += 1
    
    cap.release()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data', type=str, required=True,
                       help='Path to raw data directory')
    parser.add_argument('--output', type=str, default='data',
                       help='Output directory')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Train/val split ratio')
    
    args = parser.parse_args()
    
    prepare_training_data(args.raw_data, args.output, args.split_ratio)