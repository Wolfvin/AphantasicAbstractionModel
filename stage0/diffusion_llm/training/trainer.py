"""
AAM Diffusion LLM — Trainer

Training loop for the AAM Diffusion Model.

Handles:
    - Training loop with gradient accumulation
    - Learning rate scheduling with warmup
    - Mixed precision training (AMP)
    - EMA model updates
    - Checkpoint saving/loading
    - Logging to console and Weights & Biases
    - Evaluation on validation set

Analogi: Seperti latihan fisik Jin Soun — berulang-ulang,
bertahap meningkat intensitas, dengan instruktur yang
mengawasi dan memberi koreksi.
"""

from __future__ import annotations

import sys as _stage0_sys
from pathlib import Path as _stage0_Path
_stage0_dir = str(_stage0_Path(__file__).resolve().parent)
while _stage0_dir and not _stage0_Path(_stage0_dir, "layer0").is_dir() and _stage0_Path(_stage0_dir).parent != _stage0_dir:
    _stage0_dir = str(_stage0_Path(_stage0_dir).parent)
if _stage0_dir not in _stage0_sys.path:
    _stage0_sys.path.insert(0, _stage0_dir)

import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusion_llm.config.model_config import AamDiffusionConfig
from diffusion_llm.model.aam_diffusion_model import AamDiffusionModel
from diffusion_llm.training.dataset import GraphNarrativeDataset, collate_fn
from diffusion_llm.tokenizer.aam_tokenizer import AamTokenizer
from diffusion_llm.training.losses import DiffusionLoss

logger = logging.getLogger(__name__)


class AamTrainer:
    """Trainer for the AAM Diffusion Model.

    Args:
        config: AamDiffusionConfig with training settings.
        model: AamDiffusionModel instance.
        tokenizer: AamTokenizer instance.
        train_dataset: Training dataset.
        val_dataset: Optional validation dataset.
    """

    def __init__(
        self,
        config: AamDiffusionConfig,
        model: AamDiffusionModel,
        tokenizer: AamTokenizer,
        train_dataset: GraphNarrativeDataset,
        val_dataset: Optional[GraphNarrativeDataset] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        logger.info("Training on device: %s", self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_eps,
        )

        # Loss function
        self.loss_fn = DiffusionLoss(config.diffusion)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.training.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
        else:
            self.val_loader = None

        # LR scheduler
        self.scheduler = self._create_lr_scheduler()

        # AMP
        self.scaler = None
        if config.training.use_amp:
            dtype = torch.bfloat16 if config.training.amp_dtype == "bf16" else torch.float16
            self.scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

        # EMA
        self.ema_model = None
        if config.training.use_ema:
            self.ema_model = self._create_ema_model()

        # State tracking
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses: list[float] = []

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Seed
        torch.manual_seed(config.seed)

    def _create_lr_scheduler(self):
        """Create learning rate scheduler with warmup."""
        total_steps = self.config.training.max_steps
        warmup_steps = self.config.training.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            if self.config.training.lr_schedule == "cosine":
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            elif self.config.training.lr_schedule == "linear":
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 1.0 - progress
            else:
                return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _create_ema_model(self) -> AamDiffusionModel:
        """Create EMA copy of the model."""
        import copy
        ema = copy.deepcopy(self.model)
        for param in ema.parameters():
            param.requires_grad = False
        return ema

    @torch.no_grad()
    def _update_ema(self) -> None:
        """Update EMA model weights."""
        if self.ema_model is None:
            return
        decay = self.config.training.ema_decay
        for ema_param, model_param in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)

    def train(self) -> None:
        """Main training loop.

        Runs for max_steps or max_epochs, whichever comes first.
        Saves checkpoints and runs evaluation periodically.
        """
        logger.info("Starting training...")
        logger.info("  Max steps: %d", self.config.training.max_steps)
        logger.info("  Batch size: %d", self.config.training.batch_size)
        logger.info("  Gradient accumulation: %d", self.config.training.gradient_accumulation_steps)
        logger.info("  Effective batch size: %d",
                     self.config.training.batch_size * self.config.training.gradient_accumulation_steps)

        start_time = time.time()
        epoch = 0

        while self.global_step < self.config.training.max_steps:
            epoch += 1
            if epoch > self.config.training.max_epochs:
                break

            logger.info("=== Epoch %d ===", epoch)
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                loss = self._train_step(batch)
                epoch_loss += loss
                n_batches += 1

                # Logging
                if self.global_step % self.config.training.log_every_steps == 0:
                    avg_loss = epoch_loss / max(n_batches, 1)
                    lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / max(elapsed, 1)

                    logger.info(
                        "Step %d | Loss: %.4f | LR: %.2e | Speed: %.1f steps/s",
                        self.global_step, loss, lr, steps_per_sec,
                    )

                # Evaluation
                if (self.global_step % self.config.training.eval_every_steps == 0
                        and self.val_loader is not None):
                    val_loss = self.evaluate()
                    logger.info("Validation loss: %.4f", val_loss)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best.pt")

                # Checkpoint
                if self.global_step % self.config.training.save_every_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}.pt")

                # Stop condition
                if self.global_step >= self.config.training.max_steps:
                    break

            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            logger.info("Epoch %d complete. Average loss: %.4f", epoch, avg_epoch_loss)

        # Final save
        self._save_checkpoint("final.pt")
        elapsed = time.time() - start_time
        logger.info(
            "Training complete! %d steps in %.1f hours",
            self.global_step, elapsed / 3600,
        )

    def _train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Single training step.

        Args:
            batch: Batch of training data.

        Returns:
            Loss value for this step.
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Sample random timesteps
        batch_size = batch["token_ids"].shape[0]
        t = torch.randint(
            0, self.config.diffusion.n_timesteps,
            (batch_size,), device=self.device,
        )

        # Forward pass
        if self.scaler is not None:
            with torch.amp.autocast("cuda", enabled=True):
                predicted, target = self.model(
                    token_ids=batch["token_ids"],
                    timestep=t,
                    evidence_ids=batch.get("evidence_ids"),
                    evidence_confidence=batch.get("evidence_confidence"),
                    anomaly_ids=batch.get("anomaly_ids"),
                    anomaly_confidence=batch.get("anomaly_confidence"),
                    reasoning_ids=batch.get("reasoning_ids"),
                    reasoning_confidence=batch.get("reasoning_confidence"),
                    source_trust=batch.get("source_trust"),
                )
                loss = self.model.compute_loss(predicted, target, t)
                loss = loss / self.config.training.gradient_accumulation_steps
        else:
            predicted, target = self.model(
                token_ids=batch["token_ids"],
                timestep=t,
                evidence_ids=batch.get("evidence_ids"),
                evidence_confidence=batch.get("evidence_confidence"),
                anomaly_ids=batch.get("anomaly_ids"),
                anomaly_confidence=batch.get("anomaly_confidence"),
                reasoning_ids=batch.get("reasoning_ids"),
                reasoning_confidence=batch.get("reasoning_confidence"),
                source_trust=batch.get("source_trust"),
            )
            loss = self.model.compute_loss(predicted, target, t)
            loss = loss / self.config.training.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.grad_clip_norm,
            )

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # LR schedule
            self.scheduler.step()

            # Zero gradients
            self.optimizer.zero_grad()

            # EMA update
            self._update_ema()

        self.global_step += 1
        self.train_losses.append(loss.item())

        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set.

        Returns:
            Average validation loss.
        """
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            batch_size = batch["token_ids"].shape[0]
            t = torch.randint(
                0, self.config.diffusion.n_timesteps,
                (batch_size,), device=self.device,
            )

            predicted, target = self.model(
                token_ids=batch["token_ids"],
                timestep=t,
                evidence_ids=batch.get("evidence_ids"),
                evidence_confidence=batch.get("evidence_confidence"),
                anomaly_ids=batch.get("anomaly_ids"),
                anomaly_confidence=batch.get("anomaly_confidence"),
                reasoning_ids=batch.get("reasoning_ids"),
                reasoning_confidence=batch.get("reasoning_confidence"),
                source_trust=batch.get("source_trust"),
            )
            loss = self.model.compute_loss(predicted, target, t)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        self.model.train()
        return avg_loss

    def _save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        path = self.output_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict(),
        }
        if self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        keep_n = self.config.training.keep_last_n_checkpoints
        checkpoints = sorted(self.output_dir.glob("step_*.pt"))
        while len(checkpoints) > keep_n:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            logger.info("Removed old checkpoint: %s", oldest)

    def load_checkpoint(self, path: str) -> None:
        """Load from checkpoint.

        Args:
            path: Checkpoint file path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info("Loaded checkpoint from step %d", self.global_step)
