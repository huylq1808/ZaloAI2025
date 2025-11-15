"""
Training Entry Point for Few-Shot YOLOv5

Usage:
    python tracking/train_fewshot.py --config configs/train_config.yaml
"""

import sys
sys.path.append('.')

import argparse
import yaml
import torch
from pathlib import Path
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
    parser.add_argument('--device', type=str, default=None,
                       help='Device to train on (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}...")
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device if specified via CLI
    if args.device:
        config['device'] = args.device
    elif 'device' not in config:
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set default values for missing config keys
    config.setdefault('yolo_weights', 'yolov5m.pt')
    config.setdefault('freeze_backbone', True)
    config.setdefault('num_refs', 3)
    config.setdefault('num_classes', 1)
    config.setdefault('img_size', 640)
    config.setdefault('batch_size', 8)
    config.setdefault('num_workers', 4)
    config.setdefault('augment', True)
    config.setdefault('cache_images', False)
    
    print("="*70)
    print("Few-Shot YOLOv5 Training")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Device: {config['device']}")
    print(f"Resume: {args.resume if args.resume else 'None'}")
    print("="*70 + "\n")
    
    # Validate data paths from config
    train_data = config.get('train_data')
    val_data = config.get('val_data')
    
    if not train_data:
        raise ValueError("'train_data' path not specified in config file")
    if not val_data:
        raise ValueError("'val_data' path not specified in config file")
    
    train_data_path = Path(train_data)
    val_data_path = Path(val_data)
    
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_data_path}")
    if not val_data_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_data_path}")
    
    print(f"Train data: {train_data_path}")
    print(f"Val data: {val_data_path}\n")
    
    # Build model
    print("Building model...")
    # ✅ FIX: Unpack the tuple - build_fewshot_model returns (model, metadata)
    model, metadata = build_fewshot_model(
        yolo_weights=config['yolo_weights'],
        checkpoint=args.resume,
        freeze_backbone=config['freeze_backbone'],
        num_refs=config['num_refs'],
        num_classes=config['num_classes'],
        device=config['device']
    )
    
    # Print checkpoint info if loaded
    if metadata:
        print(f"\n✓ Loaded checkpoint metadata:")
        print(f"  - Epoch: {metadata.get('epoch', 'N/A')}")
        print(f"  - Loss: {metadata.get('loss', 'N/A')}")
    
    # Build datasets
    print("\nLoading datasets...")
    train_dataset = FewShotDetectionDataset(
        data_dir=str(train_data_path),
        split='train',
        img_size=config['img_size'],
        num_refs=config['num_refs'],
        augment=config['augment'],
        cache_images=config['cache_images']
    )
    
    val_dataset = FewShotDetectionDataset(
        data_dir=str(val_data_path),
        split='val',
        img_size=config['img_size'],
        num_refs=config['num_refs'],
        augment=False,
        cache_images=config['cache_images']
    )
    
    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=FewShotDetectionDataset.collate_fn,
        pin_memory=True if config['device'] == 'cuda' else False,
        persistent_workers=True if config['num_workers'] > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=FewShotDetectionDataset.collate_fn,
        pin_memory=True if config['device'] == 'cuda' else False,
        persistent_workers=True if config['num_workers'] > 0 else False,
        drop_last=False
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Build trainer
    print("\nInitializing trainer...")
    # ✅ FIX: Pass only the model (not the tuple)
    trainer = FewShotYOLOTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config['device']
    )
    
    # Start training
    print("\nStarting training...")
    print("="*70 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(
            epoch=trainer.epoch,
            val_loss=float('inf'),
            is_best=False
        )
        print("✓ Checkpoint saved")
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best checkpoint: {trainer.save_dir / 'best.pt'}")
    print(f"Last checkpoint: {trainer.save_dir / 'last.pt'}")
    print(f"Logs: {trainer.save_dir / 'logs'}")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)