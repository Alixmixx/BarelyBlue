"""
Neural network training module for chess position evaluation.

This module provides PyTorch-based training infrastructure including:
- Dataset loaders for HDF5 chess datasets
- ResNet model architecture
- Training loop with validation and checkpointing
"""

from chess_engine.training.dataset import ChessDataset
from chess_engine.training.config import TrainingConfig
from chess_engine.training.model import ChessNet, ResidualBlock, create_model
from chess_engine.training.trainer import ChessTrainer

__all__ = [
    "ChessDataset",
    "TrainingConfig",
    "ChessNet",
    "ResidualBlock",
    "create_model",
    "ChessTrainer",
]
