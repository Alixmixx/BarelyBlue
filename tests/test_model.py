"""Tests for ChessNet neural network model."""

import pytest
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from chess_engine.training.model import ChessNet, ResidualBlock, create_model


class TestResidualBlock:
    """Test ResidualBlock class."""

    def test_initialization(self):
        """Test block initialization."""
        block = ResidualBlock(channels=128)

        assert isinstance(block.conv1, nn.Conv2d)
        assert isinstance(block.conv2, nn.Conv2d)
        assert isinstance(block.bn1, nn.BatchNorm2d)
        assert isinstance(block.bn2, nn.BatchNorm2d)

    def test_forward_shape(self):
        """Test forward pass preserves shape."""
        block = ResidualBlock(channels=128)
        x = torch.randn(4, 128, 8, 8)  # Batch of 4

        output = block(x)

        assert output.shape == (4, 128, 8, 8)

    def test_gradient_flow(self):
        """Test gradients flow through skip connection."""
        block = ResidualBlock(channels=64)
        x = torch.randn(2, 64, 8, 8, requires_grad=True)

        output = block(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestChessNet:
    """Test ChessNet model."""

    def test_initialization_default(self):
        """Test model with default parameters."""
        model = ChessNet()

        assert model.blocks == 5
        assert model.channels == 128

    def test_initialization_custom(self):
        """Test model with custom parameters."""
        model = ChessNet(blocks=10, channels=256)

        assert model.blocks == 10
        assert model.channels == 256

    def test_invalid_blocks_raises_error(self):
        """Test that invalid block count raises ValueError."""
        with pytest.raises(ValueError, match="blocks must be"):
            ChessNet(blocks=7)

    def test_invalid_channels_raises_error(self):
        """Test that invalid channel count raises ValueError."""
        with pytest.raises(ValueError, match="channels must be"):
            ChessNet(channels=100)

    def test_forward_pass_shape(self):
        """Test forward pass returns correct shape."""
        model = ChessNet(blocks=3, channels=64)

        # Batch of 16 positions
        x = torch.randn(16, 18, 8, 8)

        output = model(x)

        # Should output scalar per position
        assert output.shape == (16,)
        assert output.dtype == torch.float32

    def test_forward_pass_single(self):
        """Test forward pass with single position."""
        model = ChessNet(blocks=3, channels=64)

        # Single position
        x = torch.randn(1, 18, 8, 8)

        output = model(x)

        assert output.shape == (1,)

    def test_output_bounded(self):
        """Test output is bounded to [-1, 1]."""
        model = ChessNet(blocks=3, channels=64)
        model.eval()

        # Test multiple random inputs
        for _ in range(10):
            x = torch.randn(8, 18, 8, 8)
            output = model(x)

            # All outputs should be in [-1, 1] due to tanh
            assert torch.all(output >= -1.0)
            assert torch.all(output <= 1.0)

    def test_parameter_count_small(self):
        """Test parameter count for small model."""
        model = ChessNet(blocks=3, channels=64)
        params = model.count_parameters()

        # Expected: ~750K parameters
        assert 700_000 < params < 850_000

    def test_parameter_count_medium(self):
        """Test parameter count for medium model."""
        model = ChessNet(blocks=5, channels=128)
        params = model.count_parameters()

        # Expected: ~2M parameters
        assert 1_900_000 < params < 2_200_000

    def test_parameter_count_large(self):
        """Test parameter count for large model."""
        model = ChessNet(blocks=10, channels=256)
        params = model.count_parameters()

        # Expected: ~12M parameters
        assert 12_000_000 < params < 13_000_000

    def test_gradient_flow(self):
        """Test gradients flow through entire network."""
        model = ChessNet(blocks=3, channels=64)
        x = torch.randn(4, 18, 8, 8, requires_grad=True)

        # Forward pass
        output = model(x)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check input gradients exist
        assert x.grad is not None

        # Check all model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_training_mode(self):
        """Test model in training mode."""
        model = ChessNet(blocks=3, channels=64)
        model.train()

        x = torch.randn(8, 18, 8, 8)
        output = model(x)

        assert output.shape == (8,)

    def test_eval_mode(self):
        """Test model in evaluation mode."""
        model = ChessNet(blocks=3, channels=64)
        model.eval()

        x = torch.randn(8, 18, 8, 8)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (8,)

    def test_model_serialization(self):
        """Test saving and loading model."""
        model = ChessNet(blocks=3, channels=64)

        # Set some weights to non-random values
        with torch.no_grad():
            model.value_fc2.weight.fill_(0.5)
            model.value_fc2.bias.fill_(0.1)

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_path = Path(f.name)

        try:
            torch.save(model.state_dict(), save_path)

            # Load into new model
            loaded_model = ChessNet(blocks=3, channels=64)
            loaded_model.load_state_dict(torch.load(save_path))

            # Check weights match
            assert torch.allclose(
                model.value_fc2.weight,
                loaded_model.value_fc2.weight
            )

            assert torch.allclose(
                model.value_fc2.bias,
                loaded_model.value_fc2.bias
            )

            # Check outputs match
            x = torch.randn(4, 18, 8, 8)
            with torch.no_grad():
                output1 = model(x)
                output2 = loaded_model(x)

            assert torch.allclose(output1, output2)

        finally:
            save_path.unlink()

    def test_repr(self):
        """Test string representation."""
        model = ChessNet(blocks=5, channels=128)
        repr_str = repr(model)

        assert "ChessNet" in repr_str
        assert "blocks=5" in repr_str
        assert "channels=128" in repr_str
        assert "parameters=" in repr_str

    def test_deterministic_forward(self):
        """Test forward pass is deterministic in eval mode."""
        model = ChessNet(blocks=3, channels=64)
        model.eval()

        x = torch.randn(8, 18, 8, 8)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2)


class TestCreateModel:
    """Test create_model factory function."""

    def test_create_small(self):
        """Test creating small model."""
        model = create_model("small")

        assert model.blocks == 3
        assert model.channels == 64

    def test_create_medium(self):
        """Test creating medium model."""
        model = create_model("medium")

        assert model.blocks == 5
        assert model.channels == 128

    def test_create_large(self):
        """Test creating large model."""
        model = create_model("large")

        assert model.blocks == 10
        assert model.channels == 256

    def test_invalid_config_raises_error(self):
        """Test invalid config name raises ValueError."""
        with pytest.raises(ValueError, match="config_name must be"):
            create_model("extra_large")

    def test_default_config(self):
        """Test default config is medium."""
        model = create_model()

        assert model.blocks == 5
        assert model.channels == 128


class TestModelIntegration:
    """Integration tests for model."""

    def test_batch_processing(self):
        """Test processing batches of different sizes."""
        model = ChessNet(blocks=3, channels=64)
        model.eval()

        batch_sizes = [1, 4, 16, 32]

        with torch.no_grad():
            for batch_size in batch_sizes:
                x = torch.randn(batch_size, 18, 8, 8)
                output = model(x)

                assert output.shape == (batch_size,)

    def test_zero_input(self):
        """Test model handles zero input."""
        model = ChessNet(blocks=3, channels=64)
        model.eval()

        x = torch.zeros(4, 18, 8, 8)

        with torch.no_grad():
            output = model(x)

        # Should still produce bounded output
        assert output.shape == (4,)
        assert torch.all(output >= -1.0)
        assert torch.all(output <= 1.0)

    def test_extreme_input(self):
        """Test model handles extreme input values."""
        model = ChessNet(blocks=3, channels=64)
        model.eval()

        # Large positive values
        x = torch.ones(4, 18, 8, 8) * 10.0

        with torch.no_grad():
            output = model(x)

        # Should still produce bounded output
        assert torch.all(output >= -1.0)
        assert torch.all(output <= 1.0)
