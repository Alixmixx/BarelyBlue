"""Tests for PyTorch dataset loader."""

import pytest
from pathlib import Path

import chess
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from chess_engine.board.representation import board_to_tensor_18
from chess_engine.data.dataset_writer import DatasetWriter
from chess_engine.training.dataset import ChessDataset
from chess_engine.training.config import TrainingConfig


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a small sample dataset for testing."""
    output_path = tmp_path / "sample.h5"

    # Generate sample data for all splits
    np.random.seed(42)

    # Training data: 100 positions
    train_boards = [chess.Board() for _ in range(100)]
    train_tensors = np.array([board_to_tensor_18(board) for board in train_boards])
    train_evals = np.random.randint(-500, 500, size=100, dtype=np.int16)
    train_results = np.random.choice([1, 0, -1], size=100).astype(np.int8)
    train_ply = np.random.randint(10, 50, size=100, dtype=np.int16)
    train_fens = [board.fen() for board in train_boards]

    # Validation data: 20 positions
    val_boards = [chess.Board() for _ in range(20)]
    val_tensors = np.array([board_to_tensor_18(board) for board in val_boards])
    val_evals = np.random.randint(-500, 500, size=20, dtype=np.int16)
    val_results = np.random.choice([1, 0, -1], size=20).astype(np.int8)
    val_ply = np.random.randint(10, 50, size=20, dtype=np.int16)
    val_fens = [board.fen() for board in val_boards]

    # Test data: 20 positions
    test_boards = [chess.Board() for _ in range(20)]
    test_tensors = np.array([board_to_tensor_18(board) for board in test_boards])
    test_evals = np.random.randint(-500, 500, size=20, dtype=np.int16)
    test_results = np.random.choice([1, 0, -1], size=20).astype(np.int8)
    test_ply = np.random.randint(10, 50, size=20, dtype=np.int16)
    test_fens = [board.fen() for board in test_boards]

    # Write dataset
    with DatasetWriter(output_path) as writer:
        writer.append_batch(
            train_tensors, train_evals, train_results, train_ply, train_fens, split="train"
        )
        writer.append_batch(
            val_tensors, val_evals, val_results, val_ply, val_fens, split="validation"
        )
        writer.append_batch(
            test_tensors, test_evals, test_results, test_ply, test_fens, split="test"
        )

    return output_path


class TestChessDataset:
    """Test ChessDataset class."""

    def test_initialization_train(self, sample_dataset):
        """Test dataset initialization with train split."""
        dataset = ChessDataset(sample_dataset, split="train")

        assert len(dataset) == 100
        assert dataset.split == "train"
        assert dataset.h5_path == sample_dataset

    def test_initialization_validation(self, sample_dataset):
        """Test dataset initialization with validation split."""
        dataset = ChessDataset(sample_dataset, split="validation")

        assert len(dataset) == 20
        assert dataset.split == "validation"

    def test_initialization_test(self, sample_dataset):
        """Test dataset initialization with test split."""
        dataset = ChessDataset(sample_dataset, split="test")

        assert len(dataset) == 20
        assert dataset.split == "test"

    def test_nonexistent_file_raises_error(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ChessDataset(tmp_path / "nonexistent.h5")

    def test_invalid_split_raises_error(self, sample_dataset):
        """Test that invalid split name raises ValueError."""
        with pytest.raises(ValueError, match="Split must be"):
            ChessDataset(sample_dataset, split="invalid")

    def test_getitem_returns_correct_shapes(self, sample_dataset):
        """Test that __getitem__ returns correct tensor shapes."""
        dataset = ChessDataset(sample_dataset, split="train")

        tensor, evaluation = dataset[0]

        # Check types
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(evaluation, torch.Tensor)

        # Check shapes
        assert tensor.shape == (18, 8, 8)
        assert evaluation.shape == ()  # Scalar

        # Check dtypes
        assert tensor.dtype == torch.float32
        assert evaluation.dtype == torch.float32

    def test_evaluation_scaling(self, sample_dataset):
        """Test that evaluations are scaled to [-1, 1]."""
        dataset = ChessDataset(sample_dataset, split="train")

        # Check all evaluations are in valid range
        for i in range(len(dataset)):
            _, evaluation = dataset[i]
            assert -1.0 <= evaluation.item() <= 1.0

    def test_extreme_evaluation_clipping(self, tmp_path):
        """Test that extreme evaluations are clipped."""
        output_path = tmp_path / "extreme.h5"

        # Create dataset with extreme evaluations
        boards = [chess.Board() for _ in range(10)]
        tensors = np.array([board_to_tensor_18(board) for board in boards])
        evals = np.array([10000, -10000, 5000, -5000, 0, 100, -100, 2500, -2500, 1000], dtype=np.int16)
        results = np.zeros(10, dtype=np.int8)
        ply = np.ones(10, dtype=np.int16) * 20
        fens = [board.fen() for board in boards]

        with DatasetWriter(output_path) as writer:
            writer.append_batch(tensors, evals, results, ply, fens, split="train")

        # Load and check clipping
        dataset = ChessDataset(output_path, split="train", max_eval=5000.0)

        tensor, eval_scaled = dataset[0]
        assert eval_scaled.item() == 1.0  # 10000 clipped to 5000, then scaled to 1.0

        tensor, eval_scaled = dataset[1]
        assert eval_scaled.item() == -1.0  # -10000 clipped to -5000, then scaled to -1.0

        tensor, eval_scaled = dataset[4]
        assert eval_scaled.item() == 0.0  # 0 stays 0

    def test_dataloader_integration(self, sample_dataset):
        """Test integration with PyTorch DataLoader."""
        dataset = ChessDataset(sample_dataset, split="train")

        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        )

        # Get first batch
        batch_tensors, batch_evals = next(iter(loader))

        # Check batch shapes
        assert batch_tensors.shape == (16, 18, 8, 8)
        assert batch_evals.shape == (16,)

        # Check all batches
        total_samples = 0
        for tensors, evals in loader:
            assert tensors.shape[1:] == (18, 8, 8)  # Channels, height, width
            assert evals.shape[0] == tensors.shape[0]  # Same batch size
            total_samples += tensors.shape[0]

        assert total_samples == 100

    def test_get_statistics(self, sample_dataset):
        """Test dataset statistics computation."""
        dataset = ChessDataset(sample_dataset, split="train")
        stats = dataset.get_statistics()

        assert "mean_eval" in stats
        assert "std_eval" in stats
        assert "min_eval" in stats
        assert "max_eval" in stats
        assert "num_positions" in stats

        assert stats["num_positions"] == 100
        assert isinstance(stats["mean_eval"], float)
        assert isinstance(stats["std_eval"], float)

    def test_repr(self, sample_dataset):
        """Test string representation."""
        dataset = ChessDataset(sample_dataset, split="train")
        repr_str = repr(dataset)

        assert "ChessDataset" in repr_str
        assert "train" in repr_str
        assert "100" in repr_str


class TestTrainingConfig:
    """Test TrainingConfig class."""

    def test_default_initialization(self):
        """Test config with default values."""
        config = TrainingConfig()

        assert config.model_blocks == 5
        assert config.model_channels == 128
        assert config.batch_size == 256
        assert config.learning_rate == 0.001
        assert config.num_epochs == 50
        assert config.device in ["cpu", "cuda"]

    def test_custom_values(self, tmp_path):
        """Test config with custom values."""
        config = TrainingConfig(
            model_blocks=10,
            model_channels=256,
            batch_size=128,
            learning_rate=0.0001,
            num_epochs=100,
            dataset_path=tmp_path / "custom.h5",
        )

        assert config.model_blocks == 10
        assert config.model_channels == 256
        assert config.batch_size == 128
        assert config.learning_rate == 0.0001
        assert config.num_epochs == 100
        assert config.dataset_path == tmp_path / "custom.h5"

    def test_invalid_model_blocks_raises_error(self):
        """Test that invalid model blocks raises ValueError."""
        with pytest.raises(ValueError, match="model_blocks"):
            TrainingConfig(model_blocks=7)

    def test_invalid_model_channels_raises_error(self):
        """Test that invalid model channels raises ValueError."""
        with pytest.raises(ValueError, match="model_channels"):
            TrainingConfig(model_channels=100)

    def test_invalid_batch_size_raises_error(self):
        """Test that invalid batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=0)

    def test_invalid_learning_rate_raises_error(self):
        """Test that invalid learning rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingConfig(learning_rate=-0.001)

    def test_repr(self):
        """Test string representation."""
        config = TrainingConfig()
        repr_str = repr(config)

        assert "TrainingConfig" in repr_str
        assert "5 blocks" in repr_str
        assert "128 channels" in repr_str
