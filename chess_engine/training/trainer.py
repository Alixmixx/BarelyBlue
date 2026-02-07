"""
Training pipeline for chess neural network.

Provides ChessTrainer class for training ResNet models with validation,
checkpointing, early stopping, and TensorBoard logging.
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from chess_engine.training.config import TrainingConfig
from chess_engine.training.model import ChessNet

logger = logging.getLogger(__name__)


class ChessTrainer:
    """Training pipeline for chess neural network.

    Handles:
        - Training loop with progress bars
        - Validation with sign accuracy metric
        - Learning rate scheduling (ReduceLROnPlateau)
        - Early stopping based on validation loss
        - Checkpointing (regular + best model)
        - TensorBoard logging
    """

    def __init__(
        self,
        model: ChessNet,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        """Initialize trainer.

        Args:
            model: ChessNet model to train
            config: Training configuration
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        """
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.MSELoss()

        # Learning rate scheduler (reduce on plateau)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.lr_factor,
            patience=config.lr_patience,
        )

        # TensorBoard logging
        log_dir = config.checkpoint_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        if config.enable_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        # Early stopping state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.epoch = 0

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialized on device: {config.device}")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number (for logging)

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.num_epochs}",
            disable=not self.config.log_interval > 0,
        )

        for batch_idx, (tensors, labels) in enumerate(pbar):
            # Move to device
            tensors = tensors.to(self.config.device)
            labels = labels.to(self.config.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(tensors)
            loss = self.criterion(predictions, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate loss
            batch_loss = loss.item()
            total_loss += batch_loss

            # Update progress bar
            if self.config.log_interval > 0:
                pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

                # Log to TensorBoard
                if self.writer and (batch_idx + 1) % self.config.log_interval == 0:
                    global_step = (epoch - 1) * num_batches + batch_idx
                    self.writer.add_scalar("Loss/batch", batch_loss, global_step)

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> Tuple[float, float]:
        """Validate on validation set.

        Returns:
            Tuple of (validation_loss, sign_accuracy):
                - validation_loss: Average MSE loss on validation set
                - sign_accuracy: Fraction of predictions with correct sign
                  (i.e., correctly predicting which side is winning)
        """
        self.model.eval()
        total_loss = 0.0
        correct_signs = 0
        total_samples = 0

        with torch.no_grad():
            for tensors, labels in self.val_loader:
                # Move to device
                tensors = tensors.to(self.config.device)
                labels = labels.to(self.config.device)

                # Forward pass
                predictions = self.model(tensors)
                loss = self.criterion(predictions, labels)
                total_loss += loss.item()

                # Sign accuracy: does model predict correct winner?
                # Both > 0 (White winning) or both < 0 (Black winning) or both == 0
                pred_signs = torch.sign(predictions)
                label_signs = torch.sign(labels)
                correct_signs += (pred_signs == label_signs).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        sign_accuracy = correct_signs / total_samples if total_samples > 0 else 0.0

        return avg_loss, sign_accuracy

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss at this epoch
            is_best: If True, also save as "best_model.pt"
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Save regular checkpoint
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"✓ New best model saved! val_loss={val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.config.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logger.info(f"Loaded checkpoint from epoch {self.epoch}")

    def train(self):
        """Main training loop.

        Trains for config.num_epochs epochs with validation, learning rate
        scheduling, checkpointing, and early stopping.
        """
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        logger.info(f"Training samples: {len(self.train_loader.dataset):,}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset):,}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info("=" * 60)

        for epoch in range(1, self.config.num_epochs + 1):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, sign_accuracy = self.validate()

            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/sign", sign_accuracy, epoch)
                self.writer.add_scalar("Learning_Rate", current_lr, epoch)

            # Console logging
            logger.info(
                f"Epoch {epoch:3d}/{self.config.num_epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"sign_acc={sign_accuracy:.3f}, lr={current_lr:.6f}"
            )

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Checkpointing
            if epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)

            # Early stopping check
            if self.patience_counter >= self.config.patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement for {self.config.patience} epochs)"
                )
                break

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)
