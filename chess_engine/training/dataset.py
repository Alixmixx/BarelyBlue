"""
PyTorch Dataset wrapper for HDF5 chess position datasets.

Provides efficient loading of chess positions and evaluations from
HDF5 files created in Phase 2.
"""

import logging
from pathlib import Path
from typing import Tuple, cast

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ChessDataset(Dataset):
    """PyTorch Dataset for chess position training data.

    Loads 18-channel board tensors and Stockfish evaluations from HDF5 files.
    Automatically handles scaling of evaluations from centipawns to [-1, 1]
    range for neural network training.
    """

    def __init__(
        self,
        h5_path: Path,
        split: str = "train",
        max_eval: float = 5000.0,
    ):
        """Initialize chess dataset.

        Args:
            h5_path: Path to HDF5 dataset file
            split: Dataset split to load ("train", "validation", or "test")
            max_eval: Maximum evaluation in centipawns (values clipped to ±max_eval)

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If split name is invalid
        """
        self.h5_path = Path(h5_path)
        self.split = split
        self.max_eval = max_eval

        if not self.h5_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.h5_path}")

        if split not in ["train", "validation", "test"]:
            raise ValueError(
                f"Split must be 'train', 'validation', or 'test', got '{split}'"
            )

        with h5py.File(self.h5_path, "r") as f:
            if split not in f:
                raise ValueError(f"Split '{split}' not found in dataset")

            group = cast(h5py.Group, f[split])

            # Verify required datasets exist
            required_datasets = ["tensors", "evaluations"]
            for ds_name in required_datasets:
                if ds_name not in group:
                    raise ValueError(
                        f"Required dataset '{ds_name}' not found in split '{split}'"
                    )

            # Verify shapes
            tensors_ds = cast(h5py.Dataset, group["tensors"])
            evals_ds = cast(h5py.Dataset, group["evaluations"])

            self.length = tensors_ds.shape[0]
            tensor_shape = tensors_ds.shape
            eval_shape = evals_ds.shape

            if len(tensor_shape) != 4 or tensor_shape[1:] != (18, 8, 8):
                raise ValueError(
                    f"Expected tensor shape (N, 18, 8, 8), got {tensor_shape}"
                )

            if len(eval_shape) != 1 or eval_shape[0] != self.length:
                raise ValueError(
                    f"Evaluation shape mismatch: {eval_shape} vs tensors {tensor_shape}"
                )

        logger.info(
            f"Loaded {split} dataset: {self.length:,} positions from {self.h5_path}"
        )

    def __len__(self) -> int:
        """Return number of positions in dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get position tensor and evaluation label.

        Args:
            idx: Index of position to retrieve

        Returns:
            Tuple of (tensor, evaluation):
                - tensor: (18, 8, 8) float32 board representation
                - evaluation: Scalar float32 in [-1, 1] (scaled from centipawns)

        """
        # Open HDF5 file for this access
        with h5py.File(self.h5_path, "r") as f:
            group = cast(h5py.Group, f[self.split])

            # Load tensor and evaluation
            tensors_ds = cast(h5py.Dataset, group["tensors"])
            evals_ds = cast(h5py.Dataset, group["evaluations"])

            tensor_np = tensors_ds[idx]
            eval_cp = int(evals_ds[idx])

        # Convert to PyTorch tensors
        tensor = torch.from_numpy(tensor_np).float()

        # Scale evaluation to [-1, 1]
        # Clip extreme values to ±max_eval, then normalize
        eval_scaled = np.clip(float(eval_cp), -self.max_eval, self.max_eval)
        eval_scaled = eval_scaled / self.max_eval
        eval_tensor = torch.tensor(eval_scaled, dtype=torch.float32)

        return tensor, eval_tensor

    def get_statistics(self) -> dict:
        """Compute dataset statistics.

        Returns:
            Dictionary with:
                - mean_eval: Mean evaluation in centipawns
                - std_eval: Standard deviation in centipawns
                - min_eval: Minimum evaluation
                - max_eval: Maximum evaluation
                - num_positions: Total positions
        """
        with h5py.File(self.h5_path, "r") as f:
            group = cast(h5py.Group, f[self.split])
            evals_ds = cast(h5py.Dataset, group["evaluations"])
            evals = evals_ds[:]

        return {
            "mean_eval": float(np.mean(evals)),
            "std_eval": float(np.std(evals)),
            "min_eval": float(np.min(evals)),
            "max_eval": float(np.max(evals)),
            "num_positions": len(evals),
        }

    def __repr__(self) -> str:
        """String representation of dataset."""
        return (
            f"ChessDataset(\n"
            f"  split='{self.split}',\n"
            f"  size={self.length:,},\n"
            f"  path={self.h5_path}\n"
            f")"
        )
