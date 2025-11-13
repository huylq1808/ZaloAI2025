"""
Dataset Loader for Few-Shot YOLOv5 Training

Loads:
- Query images with annotations
- 3 reference images per video
- Handles data augmentation
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional
import random


class FewShotDetectionDataset(Dataset):
    """
    Few-Shot Detection Dataset
    
    Structure:
        data_dir/
        ├── samples/
        │   ├── video_001/
        │   │   ├── object_images/
        │   │   │   ├── img_1.jpg  (reference 1)
        │   │   │   ├── img_2.jpg  (reference 2)
        │   │   │   └── img_3.jpg  (reference 3)
        │   │   └── frames/
        │   │       ├── frame_0001.jpg
        │   │       └── ...
        │   └── ...
        └── annotations/
            └── annotations.json
    
    Args:
        data_dir (str): Root data directory
        split (str): 'train' or 'val'
        img_size (int): Image size for training
        num_refs (int): Number of reference images
        augment (bool): Whether to apply augmentation
        cache_images (bool): Cache images in memory
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 img_size: int = 640,
                 num_refs: int = 3,
                 augment: bool = True,
                 cache_images: bool = False):
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.num_refs = num_refs
        self.augment = augment
        self.cache_images = cache_images
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Build sample list
        self.samples = self._build_sample_list()
        
        # Setup augmentation pipeline
        self.transform = self._get_transform()
        
        # Image cache
        self.image_cache = {} if cache_images else None
        
        print(f"✓ Loaded {len(self.samples)} samples for {split}")
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotation JSON"""
        ann_file = self.data_dir / 'annotations' / 'annotations.json'
        
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotations not found: {ann_file}")
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        return annotations
    
    def _build_sample_list(self) -> List[Dict]:
        """
        Build list of training samples
        
        Each sample contains:
        - video_id
        - frame_path
        - reference_paths
        - bboxes
        """
        samples = []
        
        for video_ann in self.annotations:
            video_id = video_ann['video_id']
            video_dir = self.data_dir / 'samples' / video_id
            
            # Reference images
            ref_dir = video_dir / 'object_images'
            ref_paths = [
                str(ref_dir / f'img_{i}.jpg')
                for i in range(1, self.num_refs + 1)
            ]
            
            # Check references exist
            if not all(Path(p).exists() for p in ref_paths):
                print(f"Warning: Missing references for {video_id}, skipping")
                continue
            
            # Process each annotated interval
            for interval in video_ann.get('annotations', []):
                for bbox_info in interval.get('bboxes', []):
                    frame_num = bbox_info['frame']
                    
                    # Frame path
                    frame_path = video_dir / 'frames' / f'frame_{frame_num:06d}.jpg'
                    
                    if not frame_path.exists():
                        continue
                    
                    # Bbox in [x1, y1, x2, y2] format
                    bbox = [
                        bbox_info['x1'],
                        bbox_info['y1'],
                        bbox_info['x2'],
                        bbox_info['y2']
                    ]
                    
                    sample = {
                        'video_id': video_id,
                        'frame_path': str(frame_path),
                        'ref_paths': ref_paths,
                        'bboxes': [bbox],  # Can have multiple objects
                        'labels': [0]  # Single class: 0
                    }
                    
                    samples.append(sample)
        
        return samples
    
    def _get_transform(self) -> A.Compose:
        """Get augmentation pipeline"""
        
        if self.augment and self.split == 'train':
            # Training augmentation
            transform = A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(
                    min_height=self.img_size,
                    min_width=self.img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(114, 114, 114)
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
            ))
        else:
            # Validation: no augmentation
            transform = A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(
                    min_height=self.img_size,
                    min_width=self.img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(114, 114, 114)
                ),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
            ))
        
        return transform
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load image with caching"""
        
        if self.cache_images and img_path in self.image_cache:
            return self.image_cache[img_path].copy()
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.cache_images:
            self.image_cache[img_path] = img.copy()
        
        return img
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample
        
        Returns:
            Dictionary containing:
            - 'ref_images': [num_refs, 3, H, W]
            - 'query_image': [3, H, W]
            - 'boxes': [num_objs, 4]
            - 'labels': [num_objs]
            - 'video_id': str
        """
        sample = self.samples[idx]
        
        # Load query image
        query_img = self._load_image(sample['frame_path'])
        bboxes = sample['bboxes'].copy()
        labels = sample['labels'].copy()
        
        # Apply transform to query
        transformed = self.transform(
            image=query_img,
            bboxes=bboxes,
            labels=labels
        )
        
        query_tensor = transformed['image']  # [3, H, W]
        bboxes_tensor = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        labels_tensor = torch.tensor(transformed['labels'], dtype=torch.long)
        
        # Load reference images
        ref_tensors = []
        
        # Simple transform for references (no bbox)
        ref_transform = A.Compose([
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            ),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2()
        ])
        
        for ref_path in sample['ref_paths']:
            ref_img = self._load_image(ref_path)
            ref_transformed = ref_transform(image=ref_img)
            ref_tensors.append(ref_transformed['image'])
        
        ref_stack = torch.stack(ref_tensors)  # [num_refs, 3, H, W]
        
        return {
            'ref_images': ref_stack,
            'query_image': query_tensor,
            'boxes': bboxes_tensor,
            'labels': labels_tensor,
            'video_id': sample['video_id']
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate function for batching
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            Batched dictionary
        """
        ref_images = torch.stack([item['ref_images'] for item in batch])
        query_images = torch.stack([item['query_image'] for item in batch])
        
        # Boxes and labels are variable length - keep as list
        boxes = [item['boxes'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        video_ids = [item['video_id'] for item in batch]
        
        return {
            'ref_images': ref_images,      # [B, num_refs, 3, H, W]
            'query_images': query_images,   # [B, 3, H, W]
            'boxes': boxes,                 # List of [num_objs, 4]
            'labels': labels,               # List of [num_objs]
            'video_ids': video_ids
        }


# Test dataset
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    # Test dataset
    dataset = FewShotDetectionDataset(
        data_dir='data/train',
        split='train',
        img_size=640,
        num_refs=3,
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print("\nSample keys:", sample.keys())
    print(f"Ref images shape: {sample['ref_images'].shape}")
    print(f"Query image shape: {sample['query_image'].shape}")
    print(f"Boxes shape: {sample['boxes'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    
    # Test dataloader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=FewShotDetectionDataset.collate_fn,
        num_workers=2
    )
    
    batch = next(iter(loader))
    print("\nBatch keys:", batch.keys())
    print(f"Batched ref images: {batch['ref_images'].shape}")
    print(f"Batched query images: {batch['query_images'].shape}")
    print(f"Number of boxes per sample: {[len(b) for b in batch['boxes']]}")
    
    print("\n✓ Dataset test passed")