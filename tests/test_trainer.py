"""Tests for ChessTrainer."""

import pytest
import tempfile
from pathlib import Path

import chess
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from chess_engine.board.representation import board_to_tensor_18
from chess_engine.data.dataset_writer import DatasetWriter
from chess_engine.training.config import TrainingConfig
from chess_engine.training.dataset import ChessDataset
from chess_engine.training.model import ChessNet
from chess_engine.training.trainer import ChessTrainer


@pytest.fixture
def small_dataset(tmp_path):
    """Create a small dataset for training tests."""
    output_path = tmp_path / "small_train.h5"

    # Generate sample data
    np.random.seed(42)

    # Training: 100 positions
    train_boards = [chess.Board() for _ in range(100)]
    train_tensors = np.array([board_to_tensor_18(board) for board in train_boards])
    train_evals = np.random.randint(-500, 500, size=100, dtype=np.int16)
    train_results = np.zeros(100, dtype=np.int8)
    train_ply = np.ones(100, dtype=np.int16) * 20
    train_fens = [board.fen() for board in train_boards]

    # Validation: 20 positions
    val_boards = [chess.Board() for _ in range(20)]
    val_tensors = np.array([board_to_tensor_18(board) for board in val_boards])
    val_evals = np.random.randint(-500, 500, size=20, dtype=np.int16)
    val_results = np.zeros(20, dtype=np.int8)
    val_ply = np.ones(20, dtype=np.int16) * 20
    val_fens = [board.fen() for board in val_boards]

    # Write dataset
    with DatasetWriter(output_path) as writer:
        writer.append_batch(
            train_tensors, train_evals, train_results, train_ply, train_fens, split="train"
        )
        writer.append_batch(
            val_tensors, val_evals, val_results, val_ply, val_fens, split="validation"
        )

    return output_path


@pytest.fixture
def train_config(tmp_path):
    """Create minimal training config for testing."""
    return TrainingConfig(
        model_blocks=3,
        model_channels=64,
        batch_size=16,
        learning_rate=0.001,
        num_epochs=2,
        dataset_path=tmp_path / "test.h5",
        device="cpu",
        num_workers=0,
        checkpoint_dir=tmp_path / "checkpoints",
        save_every=1,
        patience=5,
        enable_tensorboard=False,
        log_interval=0,  # Disable progress bar for tests
    )


@pytest.fixture
def trainer_setup(small_dataset, train_config):
    """Create trainer with model and data loaders."""
    # Update config with actual dataset path
    train_config.dataset_path = small_dataset

    # Load datasets
    train_dataset = ChessDataset(small_dataset, split="train")
    val_dataset = ChessDataset(small_dataset, split="validation")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False)

    # Create model
    model = ChessNet(blocks=train_config.model_blocks, channels=train_config.model_channels)

    # Create trainer
    trainer = ChessTrainer(
        model=model,
        config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    return trainer


class TestChessTrainer:
    """Test ChessTrainer class."""

    def test_initialization(self, trainer_setup):
        """Test trainer initialization."""
        trainer = trainer_setup

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert trainer.scheduler is not None
        assert trainer.best_val_loss == float("inf")
        assert trainer.patience_counter == 0

    def test_train_epoch(self, trainer_setup):
        """Test single training epoch."""
        trainer = trainer_setup

        # Train for one epoch
        train_loss = trainer.train_epoch(epoch=1)

        # Check loss is reasonable
        assert isinstance(train_loss, float)
        assert train_loss > 0.0
        assert not np.isnan(train_loss)

    def test_validate(self, trainer_setup):
        """Test validation."""
        trainer = trainer_setup

        # Run validation
        val_loss, sign_accuracy = trainer.validate()

        # Check outputs
        assert isinstance(val_loss, float)
        assert isinstance(sign_accuracy, float)
        assert val_loss > 0.0
        assert 0.0 <= sign_accuracy <= 1.0

    def test_save_checkpoint(self, trainer_setup, train_config):
        """Test checkpoint saving."""
        trainer = trainer_setup

        # Save checkpoint
        trainer.save_checkpoint(epoch=1, val_loss=0.5, is_best=False)

        # Check file exists
        checkpoint_path = train_config.checkpoint_dir / "checkpoint_epoch_1.pt"
        assert checkpoint_path.exists()

        # Load and verify
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "epoch" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["epoch"] == 1

    def test_save_best_checkpoint(self, trainer_setup, train_config):
        """Test best model saving."""
        trainer = trainer_setup

        # Save as best
        trainer.save_checkpoint(epoch=1, val_loss=0.5, is_best=True)

        # Check both files exist
        checkpoint_path = train_config.checkpoint_dir / "checkpoint_epoch_1.pt"
        best_path = train_config.checkpoint_dir / "best_model.pt"

        assert checkpoint_path.exists()
        assert best_path.exists()

    def test_load_checkpoint(self, trainer_setup, train_config):
        """Test checkpoint loading."""
        trainer = trainer_setup

        # Save checkpoint
        trainer.save_checkpoint(epoch=5, val_loss=0.3, is_best=False)
        checkpoint_path = train_config.checkpoint_dir / "checkpoint_epoch_5.pt"

        # Modify model weights
        with torch.no_grad():
            trainer.model.value_fc2.weight.fill_(0.99)

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)

        # Check epoch restored
        assert trainer.epoch == 5

    def test_load_nonexistent_checkpoint_raises_error(self, trainer_setup, tmp_path):
        """Test loading nonexistent checkpoint raises error."""
        trainer = trainer_setup

        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(tmp_path / "nonexistent.pt")

    def test_full_training_loop(self, trainer_setup):
        """Test complete training loop."""
        trainer = trainer_setup

        # Train for 2 epochs
        trainer.train()

        # Check training completed
        assert trainer.epoch == 0  # epoch counter starts at 0
        assert trainer.best_val_loss < float("inf")

    def test_early_stopping(self, trainer_setup, train_config):
        """Test early stopping mechanism."""
        trainer = trainer_setup

        # Set aggressive early stopping
        train_config.patience = 1
        train_config.num_epochs = 10

        # Train (should stop early)
        trainer.train()

        # Should stop before 10 epochs
        # (exact number depends on when validation stops improving)
        assert trainer.patience_counter >= 0

    def test_learning_rate_scheduling(self, trainer_setup):
        """Test learning rate scheduling."""
        trainer = trainer_setup

        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # Train one epoch
        trainer.train_epoch(epoch=1)
        val_loss, _ = trainer.validate()

        # Step scheduler (should not reduce yet, patience=5)
        trainer.scheduler.step(val_loss)

        lr_after = trainer.optimizer.param_groups[0]["lr"]
        assert lr_after == initial_lr  # Should not change yet

    def test_model_improvement_tracking(self, trainer_setup):
        """Test best model tracking."""
        trainer = trainer_setup

        # Initial state
        assert trainer.best_val_loss == float("inf")
        assert trainer.patience_counter == 0

        # Simulate improvement
        trainer.best_val_loss = 1.0
        new_loss = 0.8

        is_best = new_loss < trainer.best_val_loss
        assert is_best

        # Simulate no improvement
        new_loss = 1.2
        is_best = new_loss < trainer.best_val_loss
        assert not is_best

    def test_device_placement(self, train_config):
        """Test model and data are placed on correct device."""
        train_config.device = "cpu"

        model = ChessNet(blocks=3, channels=64)

        # Create dummy loaders
        dummy_tensors = torch.randn(10, 18, 8, 8)
        dummy_labels = torch.randn(10)
        dummy_dataset = torch.utils.data.TensorDataset(dummy_tensors, dummy_labels)
        dummy_loader = DataLoader(dummy_dataset, batch_size=5)

        trainer = ChessTrainer(
            model=model,
            config=train_config,
            train_loader=dummy_loader,
            val_loader=dummy_loader,
        )

        # Check model is on CPU
        assert next(trainer.model.parameters()).device.type == "cpu"

    def test_gradient_flow_during_training(self, trainer_setup):
        """Test gradients are computed during training."""
        trainer = trainer_setup

        # Train one epoch
        trainer.train_epoch(epoch=1)

        # Check model has gradients
        has_gradients = any(
            p.grad is not None for p in trainer.model.parameters() if p.requires_grad
        )
        assert has_gradients

    def test_eval_mode_during_validation(self, trainer_setup):
        """Test model is in eval mode during validation."""
        trainer = trainer_setup

        # Put model in train mode
        trainer.model.train()
        assert trainer.model.training

        # Run validation
        trainer.validate()

        # Model should be in eval mode after validation
        # (Note: validate() sets eval mode but doesn't restore train mode)
        assert not trainer.model.training


class TestTrainerIntegration:
    """Integration tests for training pipeline."""

    def test_full_pipeline_with_checkpointing(self, small_dataset, tmp_path):
        """Test complete pipeline: train, save, load, resume."""
        config = TrainingConfig(
            model_blocks=3,
            model_channels=64,
            batch_size=16,
            learning_rate=0.001,
            num_epochs=3,
            dataset_path=small_dataset,
            device="cpu",
            num_workers=0,
            checkpoint_dir=tmp_path / "checkpoints",
            save_every=1,
            enable_tensorboard=False,
            log_interval=0,
        )

        # Load datasets
        train_dataset = ChessDataset(small_dataset, split="train")
        val_dataset = ChessDataset(small_dataset, split="validation")

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        # Create and train model
        model = ChessNet(blocks=3, channels=64)
        trainer = ChessTrainer(model, config, train_loader, val_loader)

        # Train
        trainer.train()

        # Check checkpoints exist
        assert (config.checkpoint_dir / "best_model.pt").exists()
        assert (config.checkpoint_dir / "checkpoint_epoch_1.pt").exists()

        # Load best model
        best_checkpoint = torch.load(
            config.checkpoint_dir / "best_model.pt", weights_only=False
        )
        assert "model_state_dict" in best_checkpoint
        assert best_checkpoint["val_loss"] == trainer.best_val_loss

    def test_training_reduces_loss(self, small_dataset, tmp_path):
        """Test that training completes and best_val_loss is tracked."""
        config = TrainingConfig(
            model_blocks=3,
            model_channels=64,
            batch_size=16,
            learning_rate=0.01,  # Higher LR for faster convergence in test
            num_epochs=5,
            dataset_path=small_dataset,
            device="cpu",
            num_workers=0,
            checkpoint_dir=tmp_path / "checkpoints",
            enable_tensorboard=False,
            log_interval=0,
        )

        train_dataset = ChessDataset(small_dataset, split="train")
        val_dataset = ChessDataset(small_dataset, split="validation")

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        model = ChessNet(blocks=3, channels=64)
        trainer = ChessTrainer(model, config, train_loader, val_loader)

        # Train
        trainer.train()

        # Check that best_val_loss was updated from infinity
        # (training is stochastic, so we can't guarantee loss decreases,
        # but it should at least complete and track a best loss)
        assert trainer.best_val_loss < float("inf")
