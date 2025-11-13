"""
Training Entry Point for Few-Shot YOLOv5

Usage:
    python tracking/train.py --config configs/train_config.yaml
"""

import sys
sys.path.append('.')

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from lib.models.yolov5_fewshot.backbone_model import build_fewshot_model
from lib.train.dataset_fewshot import FewShotDetectionDataset
from lib.train.trainer import FewShotYOLOTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Few-Shot YOLOv5')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to training config YAML')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device if specified
    if args.device:
        config['device'] = args.device
    
    print("="*70)
    print("Few-Shot YOLOv5 Training")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Device: {config['device']}")
    print("="*70 + "\n")
    
    # Build model
    print("Building model...")
    model = build_fewshot_model(
        yolo_weights=config['yolo_weights'],
        checkpoint=args.resume,
        freeze_backbone=config['freeze_backbone'],
        num_refs=config['num_refs'],
        num_classes=config['num_classes'],
        device=config['device']
    )
    
    # Build datasets
    print("\nLoading datasets...")
    train_dataset = FewShotDetectionDataset(
        data_dir=config['train_data'],
        split='train',
        img_size=config['img_size'],
        num_refs=config['num_refs'],
        augment=config['augment'],
        cache_images=config.get('cache_images', False)
    )
    
    val_dataset = FewShotDetectionDataset(
        data_dir=config['val_data'],
        split='val',
        img_size=config['img_size'],
        num_refs=config['num_refs'],
        augment=False,
        cache_images=config.get('cache_images', False)
    )
    
    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=FewShotDetectionDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=FewShotDetectionDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # Build trainer
    print("\nInitializing trainer...")
    trainer = FewShotYOLOTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config['device']
    )
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()