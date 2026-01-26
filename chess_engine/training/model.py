"""
ResNet-based neural network architecture for chess position evaluation.

Components:
    - Conv2d: Convolutional layer that slides filters across the board to detect patterns
      (e.g., piece configurations, pawn structures). Uses 3x3 kernels to learn local features.

    - BatchNorm2d (BN): Normalizes activations during training for stability and faster
      convergence. Helps prevent vanishing/exploding gradients in deep networks.

    - ReLU: Rectified Linear Unit activation (max(0, x)). Introduces non-linearity,
      allowing the network to learn complex functions beyond linear transformations.

    - Skip Connection: Residual path (x + f(x)) that allows gradients to flow directly
      through the network. Essential for training deep networks (>10 layers).

    - Tanh: Hyperbolic tangent activation that bounds output to [-1, 1], mapping to
      win probability (-1 = Black winning, 0 = equal, +1 = White winning).

Architecture inspired by AlphaZero's ResNet design, adapted for position evaluation.
Sources: https://arxiv.org/html/2304.14918v2, https://www.biostat.wisc.edu/~craven/cs760/lectures/AlphaZero.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and skip connection.

    Architecture:
        x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU
    """

    def __init__(self, channels: int):
        """Initialize residual block.

        Args:
            channels: Number of input/output channels
        """
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor (N, channels, 8, 8)

        Returns:
            Output tensor (N, channels, 8, 8)
        """
        residual = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        # Skip connection
        x = x + residual
        x = F.relu(x)

        return x


class ChessNet(nn.Module):
    """ResNet-based chess position evaluator.

    Architecture:
        1. Input: (N, 18, 8, 8) tensor
           - 12 piece planes (6 types x2 colors)
           - 6 metadata planes (castling, en passant, turn)

        2. Initial convolution: 18 -> channels

        3. N residual blocks (depth configurable)

        4. Value head:
           - 1x1 conv: channels -> 32
           - Flatten + Dense layers
           - Output: scalar in [-1, 1]

    The architecture is inspired by AlphaZero but simplified for evaluation only
    """

    def __init__(self, blocks: int = 5, channels: int = 128):
        """Initialize ChessNet.

        Args:
            blocks: Number of residual blocks (3, 5, 10, 15, or 20)
            channels: Number of filters per conv layer (64, 128, 256, or 512)

        Raises:
            ValueError: If blocks or channels not in allowed values
        """
        super().__init__()

        if blocks not in [3, 5, 10, 15, 20]:
            raise ValueError(
                f"blocks must be 3, 5, 10, 15, or 20, got {blocks}"
            )

        if channels not in [64, 128, 256, 512]:
            raise ValueError(
                f"channels must be 64, 128, 256, or 512, got {channels}"
            )

        self.blocks = blocks
        self.channels = channels

        # Initial convolution: 18 input channels -> channels
        self.input_conv = nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(blocks)
        ])

        # Value head
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.

        Args:
            x: Input board tensors (N, 18, 8, 8)

        Returns:
            Evaluation scores (N,) in range [-1, 1]
            Corresponds to win probability:
                +1.0 = White winning
                 0.0 = Equal position
                -1.0 = Black winning
        """
        # Initial convolution
        x = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Value head
        x = F.relu(self.value_bn(self.value_conv(x)))
        x = x.view(x.size(0), -1)  # Flatten: (N, 32, 8, 8) -> (N, 2048)
        x = F.relu(self.value_fc1(x))
        x = torch.tanh(self.value_fc2(x))  # Bound to [-1, 1]

        return x.squeeze(-1)  # (N, 1) -> (N,)

    def count_parameters(self) -> int:
        """Count trainable parameters in model.

        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation of model."""
        params = self.count_parameters()
        return (
            f"ChessNet(\n"
            f"  blocks={self.blocks},\n"
            f"  channels={self.channels},\n"
            f"  parameters={params:,}\n"
            f")"
        )


def create_model(config_name: str = "medium") -> ChessNet:
    """Factory function to create model from preset configurations.

    Args:
        config_name: One of "small", "medium", "large"

    Returns:
        Configured ChessNet model

    Raises:
        ValueError: If config_name not recognized
    """
    configs = {
        "small": {"blocks": 3, "channels": 64},      # ~100K params, fast
        "medium": {"blocks": 5, "channels": 128},    # ~500K params, balanced
        "large": {"blocks": 10, "channels": 256},    # ~2M params, strong
    }

    if config_name not in configs:
        raise ValueError(
            f"config_name must be 'small', 'medium', or 'large', got '{config_name}'"
        )

    return ChessNet(**configs[config_name])
