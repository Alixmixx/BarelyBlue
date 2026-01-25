"""Tests for dataset validator."""

import pytest
from pathlib import Path

import chess
import h5py
import numpy as np

from chess_engine.board.representation import board_to_tensor_18
from chess_engine.data.dataset_writer import DatasetWriter
from chess_engine.data.validator import DatasetValidator


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a small sample dataset for testing."""
    output_path = tmp_path / "sample.h5"

    # Generate sample data
    boards = [chess.Board() for _ in range(50)]
    tensors = np.array([board_to_tensor_18(board) for board in boards])
    evaluations = np.random.randint(-500, 500, size=50, dtype=np.int16)
    game_results = np.random.choice([1, 0, -1], size=50).astype(np.int8)
    ply = np.random.randint(10, 50, size=50, dtype=np.int16)
    fens = [board.fen() for board in boards]

    # Write dataset
    with DatasetWriter(output_path) as writer:
        writer.append_batch(tensors, evaluations, game_results, ply, fens, split="train")

    return output_path


class TestDatasetValidator:
    """Test DatasetValidator class."""

    def test_initialization(self, sample_dataset):
        """Test validator initialization."""
        validator = DatasetValidator(sample_dataset)
        assert validator.dataset_path == sample_dataset

    def test_nonexistent_file_raises_error(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DatasetValidator(tmp_path / "nonexistent.h5")

    def test_invalid_hdf5_raises_error(self, tmp_path):
        """Test that invalid HDF5 file raises ValueError."""
        # Create an invalid HDF5 file (missing required splits)
        invalid_file = tmp_path / "invalid.h5"
        with h5py.File(invalid_file, "w") as f:
            f.create_dataset("dummy", data=[1, 2, 3])

        with pytest.raises(ValueError, match="Dataset missing required splits"):
            DatasetValidator(invalid_file)

    def test_validate_distributions(self, sample_dataset):
        """Test evaluation distribution validation."""
        validator = DatasetValidator(sample_dataset)
        distributions = validator.validate_distributions()

        assert "train" in distributions
        assert "mean" in distributions["train"]
        assert "std" in distributions["train"]
        assert "min" in distributions["train"]
        assert "max" in distributions["train"]
        assert "count" in distributions["train"]
        assert distributions["train"]["count"] == 50

    def test_validate_game_results(self, sample_dataset):
        """Test game result balance validation."""
        validator = DatasetValidator(sample_dataset)
        results = validator.validate_game_results()

        assert "train" in results
        assert 1 in results["train"]
        assert 0 in results["train"]
        assert -1 in results["train"]

        # Total should equal 50
        total = results["train"][1] + results["train"][0] + results["train"][-1]
        assert total == 50

    def test_check_duplicates(self, sample_dataset):
        """Test duplicate detection."""
        validator = DatasetValidator(sample_dataset)
        dup_count, dup_pct = validator.check_duplicates()

        # With starting position boards, should be all duplicates except first
        assert isinstance(dup_count, int)
        assert isinstance(dup_pct, float)
        assert dup_pct >= 0.0
        # Since all positions are starting position, expect 49 duplicates (50 total - 1 first)
        assert dup_count == 49

    def test_validate_tensors(self, sample_dataset):
        """Test tensor validity check."""
        validator = DatasetValidator(sample_dataset)
        all_valid, errors = validator.validate_tensors(sample_size=10)

        assert all_valid is True
        assert len(errors) == 0

    def test_validate_split_balance(self, sample_dataset):
        """Test split balance check."""
        validator = DatasetValidator(sample_dataset)
        split_balance = validator.validate_split_balance()

        assert "train" in split_balance
        assert "validation" in split_balance
        assert "test" in split_balance

        # All data is in train, so it should be 100%
        assert split_balance["train"] == 100.0
        assert split_balance["validation"] == 0.0
        assert split_balance["test"] == 0.0

    def test_generate_report(self, sample_dataset, tmp_path):
        """Test report generation."""
        validator = DatasetValidator(sample_dataset)
        report_path = tmp_path / "report.md"

        report = validator.generate_report(
            output_path=report_path,
            check_duplicates=False,  # Skip for speed
        )

        assert isinstance(report, str)
        assert "# Dataset Validation Report" in report
        assert report_path.exists()

        # Check report content
        report_content = report_path.read_text()
        assert "Split Balance" in report_content
        assert "Evaluation Distribution" in report_content
