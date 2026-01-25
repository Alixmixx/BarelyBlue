"""Tests for data pipeline."""

import pytest
from pathlib import Path

import chess
import h5py

from chess_engine.data.pipeline import DataPipeline, PipelineConfig


@pytest.fixture
def small_pgn_file(tmp_path):
    """Create a small PGN file for testing."""
    pgn_path = tmp_path / "test.pgn"

    # Write minimal PGN
    pgn_content = """
[Event "Test Game"]
[Site "Test"]
[Date "2024.01.01"]
[Round "1"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]
[WhiteElo "2200"]
[BlackElo "2200"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
"""

    pgn_path.write_text(pgn_content.strip())
    return pgn_path


@pytest.fixture
def stockfish_available():
    """Check if Stockfish is available."""
    try:
        from chess_engine.data.stockfish_labeler import StockfishLabeler

        StockfishLabeler(depth=5)
        return True
    except FileNotFoundError:
        pytest.skip("Stockfish not installed")


class TestDataPipeline:
    """Test DataPipeline class."""

    def test_initialization(self, small_pgn_file, tmp_path, stockfish_available):
        """Test pipeline initialization."""
        output_path = tmp_path / "output.h5"

        pipeline = DataPipeline(
            pgn_sources=[small_pgn_file],
            output_path=output_path,
        )

        assert pipeline.output_path == output_path
        assert len(pipeline.pgn_sources) == 1

    def test_nonexistent_pgn_raises_error(self, tmp_path):
        """Test that missing PGN raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DataPipeline(
                pgn_sources=[tmp_path / "nonexistent.pgn"],
                output_path=tmp_path / "output.h5",
            )

    def test_custom_config(self, small_pgn_file, tmp_path, stockfish_available):
        """Test pipeline with custom configuration."""
        config = PipelineConfig(
            min_elo=2500,
            max_positions=5,
            stockfish_depth=10,
        )

        pipeline = DataPipeline(
            pgn_sources=[small_pgn_file],
            output_path=tmp_path / "output.h5",
            config=config,
        )

        assert pipeline.config.min_elo == 2500
        assert pipeline.config.max_positions == 5

    def test_pipeline_run(self, small_pgn_file, tmp_path, stockfish_available):
        """Test complete pipeline execution."""
        output_path = tmp_path / "output.h5"

        config = PipelineConfig(
            max_positions=10,
            stockfish_depth=5,  # Fast evaluation
            batch_size=5,
            validate_output=False,  # Skip validation for speed
        )

        pipeline = DataPipeline(
            pgn_sources=[small_pgn_file],
            output_path=output_path,
            config=config,
        )

        pipeline.run()

        # Verify output file exists
        assert output_path.exists()

        # Check dataset structure
        with h5py.File(output_path, "r") as f:
            assert "train" in f
            assert "validation" in f
            assert "test" in f

            # Check that some data was written
            train_group = f["train"]
            total_positions = (
                train_group["tensors"].shape[0]
                + f["validation"]["tensors"].shape[0]
                + f["test"]["tensors"].shape[0]
            )
            assert total_positions > 0
