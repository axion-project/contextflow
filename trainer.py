import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import wandb
from tqdm import tqdm
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
import os
import math
from collections import defaultdict
import warnings

# Import our components
from config import ContextFlowConfig, ModelConfig, TrainingConfig
from contextflow import ORACTransformer
from memory import MemoryStore, MemoryEntry, MemoryType, MemoryPriority
from tools import ToolOrchestrator

class TrainingPhase:
    """Training phase definitions"""
    WARMUP = "warmup"
    MAIN = "main"
    COOLDOWN = "cooldown"
    EVALUATION = "evaluation"

@dataclass
class TrainingMetrics:
    """Training metrics container"""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    lm_loss: float = 0.0
    memory_loss: float = 0.0
    prediction_loss: float = 0.0
    tool_loss: float = 0.0
    mode_loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    throughput: float = 0.0  # tokens/second
    memory_usage: float = 0.0  # GB
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class ORACDataset(Dataset):
    """Dataset for ORAC training with memory and tool integration"""
    
    def __init__(
        self,
        data_path: Union[str, Path, List[Dict[str, Any]]],
        tokenizer: Any,
        max_length: int = 1024,
        memory_store: Optional[MemoryStore] = None,
        include_memory: bool = True,
        include_predictions: bool = True,
        include_tools: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.memory_store = memory_store
        self.include_memory = include_memory
        self.include_predictions = include_predictions
        self.include_tools = include_tools
        
        # Load data
        if isinstance(data_path, (str, Path)):
            self.data = self._load_data(data_path)
        else:
            self.data = data_path
        
        # Preprocess data
        self.processed_data = self._preprocess_data()
        
    def _load_data(self, data_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load training data from file"""
        data_path = Path(data_path)
        
        if data_path.suffix == '.json':
            with open(data_path, 'r') as f:
                return json.load(f)
        elif data_path.suffix == '.jsonl':
            data = []
            with open(data_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
    
    def _preprocess_data(self) -> List[Dict[str, Any]]:
        """Preprocess raw data for training"""
        processed = []
        
        for item in tqdm(self.data, desc="Preprocessing data"):
            # Extract components
            text = item.get('text', '')
            context = item.get('context', '')
            memory_context = item.get('memory', [])
            tools_used = item.get('tools', [])
            mode = item.get('mode', 'personal')
            
            # Tokenize input
            input_text = f"{context}\n{text}" if context else text
            tokens = self.tokenizer.encode(input_text, max_length=self.max_length, truncation=True)
            
            processed_item = {
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'attention_mask': torch.ones(len(tokens), dtype=torch.long),
                'labels': torch.tensor(tokens, dtype=torch.long),  # For language modeling
                'mode': mode,
                'original_text': text,
                'context': context
            }
            
            # Add memory embeddings if available
            if self.include_memory and self.memory_store and memory_context:
                memory_embeddings = self._get_memory_embeddings(memory_context)
                processed_item['memory_embeddings'] = memory_embeddings
            
            # Add prediction targets if available
            if self.include_predictions and 'prediction_target' in item:
                prediction_tokens = self.tokenizer.encode(
                    item['prediction_target'], 
                    max_length=64, 
                    truncation=True
                )
                processed_item['prediction_target'] = torch.tensor(prediction_tokens, dtype=torch.long)
            
            # Add tool information
            if self.include_tools and tools_used:
                processed_item['tools_used'] = tools_used
            
            processed.append(processed_item)
        
        return processed
    
    def _get_memory_embeddings(self, memory_context: List[str]) -> torch.Tensor:
        """Retrieve relevant memory embeddings"""
        memory_embeddings = []
        
        for query in memory_context:
            memories = self.memory_store.retrieve_memories(query, k=3)
            for memory in memories:
                if memory.embedding is not None:
                    memory_embeddings.append(torch.tensor(memory.embedding))
        
        if memory_embeddings:
            # Stack and pad to consistent size
            max_memories = 10
            embeddings = torch.stack(memory_embeddings[:max_memories])
            
            # Pad if necessary
            if len(embeddings) < max_memories:
                padding = torch.zeros(max_memories - len(embeddings), embeddings.size(-1))
                embeddings = torch.cat([embeddings, padding], dim=0)
            
            return embeddings
        else:
            # Return empty embeddings
            return torch.zeros(10, 384)  # Default embedding size
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_data[idx]

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for ORAC training data"""
    # Pad sequences to same length
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    collated = {
        'input_ids': torch.zeros(len(batch), max_len, dtype=torch.long),
        'attention_mask': torch.zeros(len(batch), max_len, dtype=torch.long),
        'labels': torch.full((len(batch), max_len), -100, dtype=torch.long),
    }
    
    # Fill in the data
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].size(0)
        collated['input_ids'][i, :seq_len] = item['input_ids']
        collated['attention_mask'][i, :seq_len] = item['attention_mask']
        collated['labels'][i, :seq_len] = item['labels']
    
    # Add memory embeddings if present
    if 'memory_embeddings' in batch[0]:
        memory_embeddings = torch.stack([item['memory_embeddings'] for item in batch])
        collated['memory_embeddings'] = memory_embeddings
    
    # Add prediction targets if present
    if 'prediction_target' in batch[0]:
        pred_targets = [item['prediction_target'] for item in batch if 'prediction_target' in item]
        if pred_targets:
            max_pred_len = max(t.size(0) for t in pred_targets)
            padded_predictions = torch.full((len(pred_targets), max_pred_len), -100, dtype=torch.long)
            
            for i, target in enumerate(pred_targets):
                padded_predictions[i, :target.size(0)] = target
            
            collated['prediction_targets'] = padded_predictions
    
    # Add modes
    if 'mode' in batch[0]:
        collated['modes'] = [item['mode'] for item in batch]
    
    return collated

class ORACTrainer:
    """Comprehensive trainer for ORAC model"""
    
    def __init__(
        self,
        model: ORACTransformer,
        config: ContextFlowConfig,
        train_dataset: ORACDataset,
        val_dataset: Optional[ORACDataset] = None,
        memory_store: Optional[MemoryStore] = None,
        tool_orchestrator: Optional[ToolOrchestrator] = None
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.memory_store = memory_store
        self.tool_orchestrator = tool_orchestrator
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_phase = TrainingPhase.WARMUP
        
        # Setup training components
        self._setup_device()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_dataloaders()
        self._setup_mixed_precision()
        self._setup_distributed()
        self._setup_logging()
        
        # Metrics tracking
        self.train_metrics: List[TrainingMetrics] = []
        self.val_metrics: List[TrainingMetrics] = []
        
    def _setup_device(self):
        """Setup training device"""
        if self.config.system.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.system.device)
        
        self.model = self.model.to(self.device)
        
        # Model compilation for PyTorch 2.0+
        if self.config.system.compile_mode and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode=self.config.system.compile_mode)
    
    def _setup_optimizer(self):
        """Setup optimizer with weight decay and parameter grouping"""
        # Separate parameters that should and shouldn't have weight decay
        no_decay = {'bias', 'LayerNorm.weight', 'layer_norm.weight'}
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.training.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            betas=(self.config.training.beta1, self.config.training.beta2),
            eps=self.config.training.epsilon
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        total_steps = self.config.training.max_steps
        warmup_steps = self.config.training.warmup_steps
        
        if self.config.training.lr_decay_type == "cosine":
            # Warmup + Cosine decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.training.learning_rate * self.config.training.min_lr_ratio
            )
            
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        else:
            # Linear decay
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.training.min_lr_ratio,
                total_iters=total_steps
            )
    
    def _setup_dataloaders(self):
        """Setup training and validation dataloaders"""
        # Training dataloader
        sampler = None
        if self.config.training.use_ddp and dist.is_initialized():
            sampler = DistributedSampler(self.train_dataset)
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        # Validation dataloader
        if self.val_dataset is not None:
            val_sampler = None
            if self.config.training.use_ddp and dist.is_initialized():
                val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
            
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True
            )
        else:
            self.val_dataloader = None
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        self.use_amp = self.config.training.use_amp and self.device.type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
    
    def _setup_distributed(self):
        """Setup distributed training"""
        if self.config.training.use_ddp and torch.cuda.device_count() > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            
            self.model = DDP(
                self.model,
                find_unused_parameters=self.config.training.find_unused_parameters
            )
            self.is_ddp = True
        else:
            self.is_ddp = False
    
    def _setup_logging(self):
        """Setup logging and monitoring"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize wandb if configured
        if hasattr(wandb, 'init') and os.getenv('WANDB_PROJECT'):
            wandb.init(
                project=os.getenv('WANDB_PROJECT', 'contextflow'),
                config=self.config.to_dict(),
                name=f"orac_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-objective loss"""
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            memory_embeddings=batch.get('memory_embeddings'),
            attention_mask=batch.get('attention_mask'),
            use_tools=self.config.model.enable_tools,
            generate_predictions=self.config.model.enable_prediction
        )
        
        losses = {}
        
        # Language modeling loss
        lm_logits = outputs['logits']
        lm_loss = F.cross_entropy(
            lm_logits.view(-1, lm_logits.size(-1)),
            batch['labels'].view(-1),
            ignore_index=-100
        )
        losses['lm_loss'] = lm_loss
        
        # Memory loss (if memory embeddings provided)
        if 'memory_embeddings' in batch and 'hidden_states' in outputs:
            # Simple memory alignment loss
            memory_loss = F.mse_loss(
                outputs['hidden_states'].mean(dim=1),
                batch['memory_embeddings'].mean(dim=1)
            )
            losses['memory_loss'] = memory_loss
        else:
            losses['memory_loss'] = torch.tensor(0.0, device=self.device)
        
        # Prediction loss
        if 'prediction_targets' in batch and 'future_states' in outputs:
            pred_loss = F.mse_loss(
                outputs['future_states'][:, :batch['prediction_targets'].size(1)],
                batch['prediction_targets'].float()
            )
            losses['prediction_loss'] = pred_loss
        else:
            losses['prediction_loss'] = torch.tensor(0.0, device=self.device)
        
        # Tool usage loss (encourage appropriate tool selection)
        if 'tool_probabilities' in outputs:
            # Entropy regularization to encourage selective tool use
            tool_probs = outputs['tool_probabilities']
            tool_entropy = -torch.sum(tool_probs * torch.log(tool_probs + 1e-8), dim=-1)
            tool_loss = -tool_entropy.mean()  # Negative because we want some entropy
            losses['tool_loss'] = tool_loss
        else:
            losses['tool_loss'] = torch.tensor(0.0, device=self.device)
        
        # Mode classification loss (if mode targets available)
        if 'modes' in batch and 'detected_modes' in outputs:
            # Simple mode consistency loss (simplified)
            mode_loss = torch.tensor(0.0, device=self.device)
            losses['mode_loss'] = mode_loss
        else:
            losses['mode_loss'] = torch.tensor(0.0, device=self.device)
        
        # Combine losses
        total_loss = (
            self.config.training.lm_loss_weight * losses['lm_loss'] +
            self.config.training.memory_loss_weight * losses['memory_loss'] +
            self.config.training.prediction_loss_weight * losses['prediction_loss'] +
            self.config.training.tool_loss_weight * losses['tool_loss'] +
            self.config.training.mode_loss_weight * losses['mode_loss']
        )
        
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Single training step"""
        self.model.train()
        start_time = time.time()
        
        # Compute loss
        if self.use_amp:
            with autocast():
                loss, loss_dict = self.compute_loss(batch)
        else:
            loss, loss_dict = self.compute_loss(batch)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.training.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights if we've accumulated enough gradients
        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
            if self.use_amp:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
        else:
            grad_norm = 0.0
        
        # Calculate metrics
        step_time = time.time() - start_time
        tokens_per_second = (batch['input_ids'].numel() / step_time) if step_time > 0 else 0
        memory_used = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        metrics = TrainingMetrics(
            step=self.global_step,
            epoch=self.current_epoch,
            loss=loss_dict['total_loss'],
            lm_loss=loss_dict['lm_loss'],
            memory_loss=loss_dict['memory_loss'],
            prediction_loss=loss_dict['prediction_loss'],
            tool_loss=loss_dict['tool_loss'],
            mode_loss=loss_dict['mode_loss'],
            learning_rate=self.scheduler.get_last_lr()[0],
            grad_norm=grad_norm,
            throughput=tokens_per_second,
            memory_usage=memory_used
        )
        
        return metrics
    
    def validate(self) -> TrainingMetrics:
        """Run validation"""
        if self.val_dataloader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                if self.use_amp:
                    with autocast():
                        loss, loss_dict = self.compute_loss(batch)
                else:
                    loss, loss_dict = self.compute_loss(batch)
                
                batch_size = batch['input_ids'].size(0)
                total_loss += loss.item() * batch_size
                total_lm_loss += loss_dict['lm_loss'] * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        avg_lm_loss = total_lm_loss / total_samples
        
        metrics = TrainingMetrics(
            step=self.global_step,
            epoch=self.current_epoch,
            loss=avg_loss,
            lm_loss=avg_lm_loss,
        )
        
        return metrics
    
    def save_checkpoint(self, checkpoint_dir: Union[str, Path], is_best: bool = False):
        """Save training checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model state dict (handle DDP wrapper)
        model_state_dict = self.model.module.state_dict() if self.is_ddp else self.model.state_dict()
        
        checkpoint = {
            'step': self.global_step,
            'epoch': self.current_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'train_metrics': [m.to_dict() for m in self.train_metrics[-100:]],  # Keep last 100
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save as latest
        latest_path = checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save as best if applicable
        if is_best:
            best_path = checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints(checkpoint_dir)
        
        self.logger.info(f"Saved checkpoint at step {self.global_step}")
    
    def _cleanup_checkpoints(self, checkpoint_dir: Path):
        """Remove old checkpoints to save space"""
        checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Keep only the last N checkpoints
        keep_count = self.config.training.keep_checkpoints
        if len(checkpoints) > keep_count:
            for checkpoint in checkpoints[:-keep_count]:
                checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load training checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.is_ddp:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint['step']
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Loaded checkpoint from step {self.global_step}")
    
    def train(self, num_epochs: int, checkpoint_dir: str = "./checkpoints"):
        """Main training loop"""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Set sampler epoch for distributed training
            if self.is_ddp and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # Training loop
            epoch_metrics = []
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Training step
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                self.train_metrics.append(metrics)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics.loss:.4f}",
                    'lr': f"{metrics.learning_rate:.2e}",
                    'mem': f"{metrics.memory_usage:.1f}GB"
                })
                
                # Log metrics
                if self.use_wandb and self.global_step % 10 == 0:
                    wandb.log(metrics.to_dict(), step=self.global_step)
                
                # Validation
                if self.global_step % self.config.training.eval_steps == 0:
                    val_metrics = self.validate()
                    if val_metrics:
                        self.val_metrics.append(val_metrics)
                        
                        # Check if this is the best model
                        is_best = val_metrics.loss < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_metrics.loss
                        
                        if self.use_wandb:
                            wandb.log({f"val_{k}": v for k, v in val_metrics.to_dict().items()}, 
                                     step=self.global_step)
                        
                        self.logger.info(f"Validation loss: {val_metrics.loss:.4f} (best: {self.best_val_loss:.4f})")
                
                # Save checkpoint
                if self.global_step % self.config.training.save_steps == 0:
                    is_best = len(self.val_metrics) > 0 and self.val_metrics[-1].loss == self.best_val_loss
                    self.save_checkpoint(checkpoint_dir, is_best=is_best)
                
                self.global_step += 1
                
                # Check if we've reached max steps
                if self.global_step >= self.config.training.max_steps:
                    break
            
            # End of epoch validation
            val_metrics = self.validate()
            if val_metrics:
                self.val_metrics.append(val_metrics)
                is_best = val_metrics.loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics.loss
                    self.save_checkpoint(checkpoint_dir, is_best=True)
                
                self.logger.info(f"End of epoch {epoch+1} - Val loss: {val_metrics.loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(checkpoint_dir)
            
            if self.global_step >= self.config.training.max_steps:
                break
        
        self.logger.info("Training completed")
        
        # Final save
        self.save_checkpoint(checkpoint_dir)
