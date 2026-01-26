"""
Training configuration for chess neural network.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for neural network training.

    This dataclass encapsulates all training hyperparameters, paths,
    and settings in one place for easy experimentation and reproducibility.
    """

    # Model architecture
    model_blocks: int = 5
    """Number of residual blocks in ResNet (3, 5, or 10)"""

    model_channels: int = 128
    """Number of channels per convolutional layer (64, 128, or 256)"""

    # Training hyperparameters
    batch_size: int = 256
    """Batch size for training and validation"""

    learning_rate: float = 0.001
    """Initial learning rate for Adam optimizer"""

    num_epochs: int = 50
    """Maximum number of training epochs"""

    weight_decay: float = 1e-4
    """L2 regularization weight decay"""

    # Data
    dataset_path: Path = Path("data/training.h5")
    """Path to HDF5 dataset file"""

    # Optimization
    device: str = "cpu"
    """Device for training: "cpu" or "cuda" """

    num_workers: int = 4
    """Number of DataLoader worker processes"""

    # Checkpointing
    checkpoint_dir: Path = Path("models/checkpoints")
    """Directory for saving model checkpoints"""

    save_every: int = 5
    """Save checkpoint every N epochs"""

    # Early stopping
    patience: int = 10
    """Number of epochs without improvement before stopping"""

    min_delta: float = 0.001
    """Minimum change in validation loss to be considered improvement"""

    # Learning rate scheduling
    lr_scheduler: str = "plateau"
    """Learning rate scheduler: 'plateau', 'step', or 'cosine'"""

    lr_patience: int = 5
    """Epochs to wait before reducing LR (for plateau scheduler)"""

    lr_factor: float = 0.5
    """Factor to reduce LR by (for plateau scheduler)"""

    # Logging
    log_interval: int = 100
    """Log training progress every N batches"""

    enable_tensorboard: bool = True
    """Enable TensorBoard logging"""

    # Reproducibility
    random_seed: Optional[int] = 42
    """Random seed for reproducibility (None for random)"""

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.dataset_path = Path(self.dataset_path)
        self.checkpoint_dir = Path(self.checkpoint_dir)

        if self.model_blocks not in [3, 5, 10, 15, 20]:
            raise ValueError(
                f"model_blocks should be 3, 5, 10, 15, or 20, got {self.model_blocks}"
            )

        if self.model_channels not in [64, 128, 256, 512]:
            raise ValueError(
                f"model_channels should be 64, 128, 256, or 512, got {self.model_channels}"
            )

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")

        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("Warning: CUDA requested but not available, falling back to CPU")
                    self.device = "cpu"
            except ImportError:
                print("Warning: PyTorch not installed, device setting ignored")

    def __repr__(self) -> str:
        """String representation of config."""
        return (
            f"TrainingConfig(\n"
            f"  Architecture: {self.model_blocks} blocks, {self.model_channels} channels\n"
            f"  Hyperparameters: batch_size={self.batch_size}, lr={self.learning_rate}\n"
            f"  Device: {self.device}\n"
            f"  Dataset: {self.dataset_path}\n"
            f"  Output: {self.checkpoint_dir}\n"
            f")"
        )
