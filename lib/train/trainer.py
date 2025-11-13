"""
Trainer Class for Few-Shot YOLOv5

Handles:
- Training loop
- Validation
- Checkpointing
- Logging
- Learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
from tqdm import tqdm
from typing import Dict, Optional
import json

from lib.models.yolov5_fewshot.backbone_model import FewShotYOLOv5
from lib.train.losses import FewShotYOLOLoss
from lib.train.dataset_fewshot import FewShotDetectionDataset


class FewShotYOLOTrainer:
    """
    Trainer for Few-Shot YOLOv5
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - TensorBoard logging
    
    Args:
        model: FewShotYOLOv5 model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict
        device: Device to train on
    """
    
    def __init__(self,
                 model: FewShotYOLOv5,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: str = 'cuda'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = FewShotYOLOLoss(
            lambda_bbox=config.get('lambda_bbox', 5.0),
            lambda_obj=config.get('lambda_obj', 1.0),
            lambda_cls=config.get('lambda_cls', 0.5),
            lambda_sim=config.get('lambda_sim', 2.0),
            lambda_attn=config.get('lambda_attn', 0.1)
        )
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Gradient accumulation
        self.accumulation_steps = config.get('accumulation_steps', 1)
        
        # Checkpointing
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.save_dir / 'logs'))
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        
        print("="*70)
        print("Trainer Initialized")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Accumulation: {self.accumulation_steps} steps")
        print(f"Save Directory: {self.save_dir}")
        print("="*70 + "\n")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        
        # Separate parameters: backbone vs. custom modules
        backbone_params = []
        custom_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                custom_params.append(param)
        
        # Different learning rates for backbone and custom modules
        param_groups = [
            {
                'params': backbone_params,
                'lr': self.config.get('lr_backbone', 1e-5),
                'name': 'backbone'
            },
            {
                'params': custom_params,
                'lr': self.config.get('lr', 1e-4),
                'name': 'custom'
            }
        ]
        
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        print(f"✓ Optimizer: AdamW")
        print(f"  - Backbone LR: {self.config.get('lr_backbone', 1e-5)}")
        print(f"  - Custom LR: {self.config.get('lr', 1e-4)}")
        print(f"  - Weight Decay: {self.config.get('weight_decay', 1e-4)}")
        
        return optimizer
    
    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler"""
        
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config.get('lr', 1e-4) * 0.01
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('lr_step', 10),
                gamma=self.config.get('lr_gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        print(f"✓ Scheduler: {scheduler_type}")
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'bbox_loss': 0.0,
            'obj_loss': 0.0,
            'cls_loss': 0.0,
            'sim_loss': 0.0,
            'attn_loss': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.config['epochs']}",
            dynamic_ncols=True
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            ref_images = batch['ref_images'].to(self.device)      # [B, 3, 3, H, W]
            query_images = batch['query_images'].to(self.device)  # [B, 3, H, W]
            
            # Prepare targets
            targets = []
            for i in range(len(batch['boxes'])):
                targets.append({
                    'boxes': batch['boxes'][i].to(self.device),
                    'labels': batch['labels'][i].to(self.device)
                })
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Model forward
                outputs = self.model(
                    query_images=query_images,
                    ref_images=ref_images,
                    return_features=False
                )
                
                # Compute loss
                loss_dict = self.criterion(
                    predictions=outputs['predictions'],
                    targets=targets,
                    similarity_maps=outputs['similarity_map'],
                    attention_weights=outputs.get('attention_weights')
                )
                
                # Scale loss for gradient accumulation
                loss = loss_dict['total_loss'] / self.accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get('grad_clip', 10.0)
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Accumulate losses
            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to TensorBoard (every N steps)
            if self.global_step % self.config.get('log_interval', 10) == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(
                        f'train/{key}',
                        value.item(),
                        self.global_step
                    )
        
        # Average losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        
        self.model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'bbox_loss': 0.0,
            'obj_loss': 0.0,
            'cls_loss': 0.0,
            'sim_loss': 0.0,
            'attn_loss': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        pbar = tqdm(
            self.val_loader,
            desc="Validation",
            dynamic_ncols=True
        )
        
        for batch in pbar:
            # Move to device
            ref_images = batch['ref_images'].to(self.device)
            query_images = batch['query_images'].to(self.device)
            
            targets = []
            for i in range(len(batch['boxes'])):
                targets.append({
                    'boxes': batch['boxes'][i].to(self.device),
                    'labels': batch['labels'][i].to(self.device)
                })
            
            # Forward pass
            outputs = self.model(
                query_images=query_images,
                ref_images=ref_images
            )
            
            # Compute loss
            loss_dict = self.criterion(
                predictions=outputs['predictions'],
                targets=targets,
                similarity_maps=outputs['similarity_map'],
                attention_weights=outputs.get('attention_weights')
            )
            
            # Accumulate
            for key in val_losses.keys():
                val_losses[key] += loss_dict[key].item()
            
            pbar.set_postfix({'val_loss': f"{loss_dict['total_loss'].item():.4f}"})
        
        # Average
        for key in val_losses.keys():
            val_losses[key] /= num_batches
        
        return val_losses
    
    def train(self):
        """Main training loop"""
        
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Learning rate scheduling
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_losses['total_loss'])
            else:
                self.scheduler.step()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Val Loss: {val_losses['total_loss']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # TensorBoard logging
            for key in train_losses.keys():
                self.writer.add_scalar(f'epoch/train_{key}', train_losses[key], epoch)
                self.writer.add_scalar(f'epoch/val_{key}', val_losses[key], epoch)
            
            self.writer.add_scalar('epoch/lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            self.save_checkpoint(
                epoch=epoch + 1,
                val_loss=val_losses['total_loss'],
                is_best=val_losses['total_loss'] < self.best_val_loss
            )
            
            # Early stopping check
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("Training Complete!")
        print(f"Total Time: {total_time / 3600:.2f} hours")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print("="*70 + "\n")
        
        self.writer.close()
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        
        # Save last checkpoint
        self.model.save_checkpoint(
            path=str(self.save_dir / 'last.pt'),
            epoch=epoch,
            optimizer_state=self.optimizer.state_dict(),
            loss=val_loss,
            scheduler_state=self.scheduler.state_dict()
        )
        
        # Save best checkpoint
        if is_best:
            self.model.save_checkpoint(
                path=str(self.save_dir / 'best.pt'),
                epoch=epoch,
                optimizer_state=self.optimizer.state_dict(),
                loss=val_loss,
                scheduler_state=self.scheduler.state_dict()
            )
            print(f"  ✓ Saved best checkpoint (val_loss: {val_loss:.4f})")
        
        # Save periodic checkpoints
        if epoch % self.config.get('save_interval', 10) == 0:
            self.model.save_checkpoint(
                path=str(self.save_dir / f'epoch_{epoch}.pt'),
                epoch=epoch,
                optimizer_state=self.optimizer.state_dict(),
                loss=val_loss
            )
            