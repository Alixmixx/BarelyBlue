"""
HDF5 dataset writer for storing labeled chess positions.

Writes training datasets in HDF5 format with train/validation/test splits.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class DatasetWriter:
    """Write chess position datasets to HDF5 format."""

    def __init__(
        self,
        output_path: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        chunk_size: int = 1000,
    ):
        """
        Initialize dataset writer.

        Args:
            output_path: Path to output HDF5 file
            train_ratio: Fraction of data for training (default: 0.8)
            val_ratio: Fraction of data for validation (default: 0.1)
            test_ratio: Fraction of data for testing (default: 0.1)
            chunk_size: Chunk size for HDF5 datasets (for compression)

        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
            )

        self.output_path = Path(output_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.chunk_size = chunk_size

        self.h5file: Optional[h5py.File] = None
        self._counts = {"train": 0, "validation": 0, "test": 0}
        self._total_count = 0

        logger.info(f"Initialized dataset writer: {output_path}")
        logger.info(
            f"Split ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}"
        )

    def create_datasets(self):
        """
        Create HDF5 file with empty datasets.

        Creates three groups (train, validation, test) each with:
        - tensors: (N, 18, 8, 8) float32
        - evaluations: (N,) int16 (centipawns)
        - game_results: (N,) int8 (1/0/-1)
        - ply: (N,) int16
        - fen: (N,) variable-length string
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open HDF5 file
        self.h5file = h5py.File(self.output_path, "w")

        # Create groups
        for split in ["train", "validation", "test"]:
            group = self.h5file.create_group(split)

            # Tensors (18, 8, 8)
            group.create_dataset(
                "tensors",
                shape=(0, 18, 8, 8),
                maxshape=(None, 18, 8, 8),
                dtype=np.float32,
                chunks=(self.chunk_size, 18, 8, 8),
                compression="gzip",
                compression_opts=4,
            )

            # Evaluations (centipawns)
            group.create_dataset(
                "evaluations",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int16,
                chunks=(self.chunk_size,),
                compression="gzip",
            )

            # Game results
            group.create_dataset(
                "game_results",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int8,
                chunks=(self.chunk_size,),
                compression="gzip",
            )

            # Ply (half-moves from start)
            group.create_dataset(
                "ply",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int16,
                chunks=(self.chunk_size,),
                compression="gzip",
            )

            # FEN strings (variable length)
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(
                "fen",
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
                chunks=(self.chunk_size,),
                compression="gzip",
            )

        self.h5file.attrs["creation_date"] = datetime.now().isoformat()
        self.h5file.attrs["train_ratio"] = self.train_ratio
        self.h5file.attrs["val_ratio"] = self.val_ratio
        self.h5file.attrs["test_ratio"] = self.test_ratio
        self.h5file.attrs["version"] = "1.0"

        logger.info(f"Created HDF5 datasets at {self.output_path}")

    def append_batch(
        self,
        tensors: np.ndarray,
        evaluations: np.ndarray,
        game_results: np.ndarray,
        ply: np.ndarray,
        fens: List[str],
        split: str = "auto",
    ):
        """
        Append a batch of positions to the dataset.

        Args:
            tensors: (N, 18, 8, 8) tensor representations
            evaluations: (N,) centipawn evaluations
            game_results: (N,) game results (1/0/-1)
            ply: (N,) ply counts
            fens: List of N FEN strings
            split: Target split ("train", "validation", "test", or "auto")
                  If "auto", distributes based on split ratios

        Raises:
            ValueError: If arrays have mismatched lengths or invalid split
        """
        if self.h5file is None:
            raise RuntimeError("Must call create_datasets() first")

        # Validate input shapes
        batch_size = len(tensors)
        if not (
            len(evaluations) == len(game_results) == len(ply) == len(fens) == batch_size
        ):
            raise ValueError("All input arrays must have same length")

        # Determine target split
        if split == "auto":
            split = self._auto_assign_split()
        elif split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split: {split}")

        # Get group
        group = cast(h5py.Group, self.h5file[split])

        # Current size
        tensors_ds = cast(h5py.Dataset, group["tensors"])
        current_size = tensors_ds.shape[0]
        new_size = current_size + batch_size

        # Resize datasets
        tensors_ds.resize(new_size, axis=0)
        cast(h5py.Dataset, group["evaluations"]).resize(new_size, axis=0)
        cast(h5py.Dataset, group["game_results"]).resize(new_size, axis=0)
        cast(h5py.Dataset, group["ply"]).resize(new_size, axis=0)
        cast(h5py.Dataset, group["fen"]).resize(new_size, axis=0)

        # Append data
        tensors_ds[current_size:new_size] = tensors
        cast(h5py.Dataset, group["evaluations"])[current_size:new_size] = evaluations
        cast(h5py.Dataset, group["game_results"])[current_size:new_size] = game_results
        cast(h5py.Dataset, group["ply"])[current_size:new_size] = ply
        cast(h5py.Dataset, group["fen"])[current_size:new_size] = fens

        # Update counts
        self._counts[split] += batch_size
        self._total_count += batch_size

        logger.debug(
            f"Appended {batch_size} positions to {split} "
            f"(total: {self._counts[split]})"
        )

    def _auto_assign_split(self) -> str:
        """
        Automatically assign split based on current distribution.

        Uses split ratios to determine which split needs more data.

        Returns:
            Split name ("train", "validation", or "test")
        """
        if self._total_count == 0:
            return "train"

        # Calculate current ratios
        train_current = self._counts["train"] / self._total_count
        val_current = self._counts["validation"] / self._total_count
        test_current = self._counts["test"] / self._total_count

        # Calculate deficits (target - current)
        train_deficit = self.train_ratio - train_current
        val_deficit = self.val_ratio - val_current
        test_deficit = self.test_ratio - test_current

        # Assign to split with largest deficit
        deficits = {
            "train": train_deficit,
            "validation": val_deficit,
            "test": test_deficit,
        }

        return max(deficits, key=lambda split: deficits[split])

    def finalize(self):
        """
        Finalize and close the HDF5 file.

        Updates metadata with final statistics.
        """
        if self.h5file is None:
            return

        # Update metadata with final counts
        self.h5file.attrs["train_count"] = self._counts["train"]
        self.h5file.attrs["validation_count"] = self._counts["validation"]
        self.h5file.attrs["test_count"] = self._counts["test"]
        self.h5file.attrs["total_count"] = self._total_count

        # Close file
        self.h5file.close()
        self.h5file = None

        logger.info(f"Finalized dataset: {self.output_path}")
        logger.info(f"  Train: {self._counts['train']:,} positions")
        logger.info(f"  Validation: {self._counts['validation']:,} positions")
        logger.info(f"  Test: {self._counts['test']:,} positions")
        logger.info(f"  Total: {self._total_count:,} positions")

    def get_counts(self) -> Dict[str, int]:
        """Get current position counts for each split."""
        return self._counts.copy()

    def __enter__(self):
        """Context manager entry."""
        self.create_datasets()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.finalize()
