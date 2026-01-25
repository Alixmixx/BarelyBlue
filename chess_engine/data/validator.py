"""
Dataset quality validation for chess training data.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, cast

import chess
import h5py
import numpy as np
from tqdm import tqdm

from chess_engine.board.representation import board_to_tensor_18
from chess_engine.search.transposition import zobrist_hash

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Results of dataset validation."""

    total_positions: Dict[str, int]
    eval_distribution: Dict[str, Dict[str, float]]  # {split: {mean, std, min, max}}
    game_result_balance: Dict[str, Dict[int, int]]  # {split: {1: count, 0: count, -1: count}}
    duplicate_count: int
    duplicate_percentage: float
    tensor_validity: bool
    tensor_errors: List[str]
    split_balance: Dict[str, float]  # {split: percentage}

    def to_markdown(self) -> str:
        """Generate markdown validation report."""
        lines = [
            "# Dataset Validation Report",
            "",
            f"**Generated**: {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            f"- **Total Positions**: {sum(self.total_positions.values()):,}",
            f"- **Duplicates**: {self.duplicate_count:,} ({self.duplicate_percentage:.2f}%)",
            f"- **Tensor Validity**: {'✅ PASS' if self.tensor_validity else '❌ FAIL'}",
            "",
            "## Split Balance",
            "",
            "| Split | Count | Percentage |",
            "|-------|-------|------------|",
        ]

        for split in ["train", "validation", "test"]:
            count = self.total_positions.get(split, 0)
            pct = self.split_balance.get(split, 0.0)
            lines.append(f"| {split.capitalize()} | {count:,} | {pct:.1f}% |")

        lines.extend(
            [
                "",
                "## Evaluation Distribution",
                "",
                "| Split | Mean | Std Dev | Min | Max |",
                "|-------|------|---------|-----|-----|",
            ]
        )

        for split in ["train", "validation", "test"]:
            if split in self.eval_distribution:
                dist = self.eval_distribution[split]
                lines.append(
                    f"| {split.capitalize()} | {dist['mean']:.1f} | "
                    f"{dist['std']:.1f} | {dist['min']:.0f} | {dist['max']:.0f} |"
                )

        lines.extend(
            [
                "",
                "## Game Result Balance",
                "",
                "| Split | Wins | Draws | Losses |",
                "|-------|------|-------|--------|",
            ]
        )

        for split in ["train", "validation", "test"]:
            if split in self.game_result_balance:
                results = self.game_result_balance[split]
                lines.append(
                    f"| {split.capitalize()} | {results.get(1, 0):,} | "
                    f"{results.get(0, 0):,} | {results.get(-1, 0):,} |"
                )

        if not self.tensor_validity and self.tensor_errors:
            lines.extend(
                [
                    "",
                    "## Tensor Validation Errors",
                    "",
                ]
            )
            for error in self.tensor_errors[:10]:  # Limit to first 10
                lines.append(f"- {error}")

            if len(self.tensor_errors) > 10:
                lines.append(f"- ... and {len(self.tensor_errors) - 10} more")

        return "\n".join(lines)


class DatasetValidator:
    """Validate chess position datasets in HDF5 format."""

    def __init__(self, dataset_path: Path):
        """
        Initialize dataset validator.

        Args:
            dataset_path: Path to HDF5 dataset file

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset file is not valid HDF5
        """
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # Verify it's a valid HDF5 file
        try:
            with h5py.File(dataset_path, "r") as f:
                if not all(split in f for split in ["train", "validation", "test"]):
                    raise ValueError("Dataset missing required splits")
        except OSError as e:
            raise ValueError(f"Invalid HDF5 file: {e}")

        self.dataset_path = dataset_path
        logger.info(f"Initialized validator for: {dataset_path}")

    def validate_distributions(self) -> Dict[str, Dict[str, float]]:
        """
        Check evaluation distribution for each split.

        Returns:
            Dict mapping split names to statistics (mean, std, min, max)
        """
        distributions = {}

        with h5py.File(self.dataset_path, "r") as f:
            for split in ["train", "validation", "test"]:
                group = cast(h5py.Group, f[split])
                evals = cast(h5py.Dataset, group["evaluations"])[:]

                if len(evals) == 0:
                    distributions[split] = {
                        "mean": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "count": 0,
                    }
                else:
                    distributions[split] = {
                        "mean": float(np.mean(evals)),
                        "std": float(np.std(evals)),
                        "min": float(np.min(evals)),
                        "max": float(np.max(evals)),
                        "count": len(evals),
                    }

                logger.debug(
                    f"{split}: mean={distributions[split]['mean']:.1f}, "
                    f"std={distributions[split]['std']:.1f}"
                )

        return distributions

    def validate_game_results(self) -> Dict[str, Dict[int, int]]:
        """
        Check game result balance (wins/draws/losses).

        Returns:
            Dict mapping split names to result counts
        """
        result_counts = {}

        with h5py.File(self.dataset_path, "r") as f:
            for split in ["train", "validation", "test"]:
                group = cast(h5py.Group, f[split])
                results = cast(h5py.Dataset, group["game_results"])[:]

                result_counts[split] = {
                    1: int(np.sum(results == 1)),  # Wins
                    0: int(np.sum(results == 0)),  # Draws
                    -1: int(np.sum(results == -1)),  # Losses
                }

                logger.debug(f"{split}: {result_counts[split]}")

        return result_counts

    def check_duplicates(
        self, sample_size: Optional[int] = None
    ) -> Tuple[int, float]:
        """
        Detect duplicate positions using Zobrist hashing.

        Args:
            sample_size: If provided, only check first N positions (for speed)

        Returns:
            Tuple of (duplicate_count, duplicate_percentage)
        """
        seen_hashes: Set[int] = set()
        duplicate_count = 0
        total_count = 0

        with h5py.File(self.dataset_path, "r") as f:
            for split in ["train", "validation", "test"]:
                group = cast(h5py.Group, f[split])
                fens_dataset = cast(h5py.Dataset, group["fen"])
                fens = fens_dataset[:]

                # Limit sample size if requested
                if sample_size is not None:
                    limit_per_split = sample_size // 3
                    fens = fens[:limit_per_split]

                for fen_bytes in tqdm(fens, desc=f"Checking {split} for duplicates"):
                    fen = (
                        fen_bytes.decode()
                        if isinstance(fen_bytes, bytes)
                        else fen_bytes
                    )
                    board = chess.Board(fen)
                    hash_val = zobrist_hash(board)

                    if hash_val in seen_hashes:
                        duplicate_count += 1
                    else:
                        seen_hashes.add(hash_val)

                    total_count += 1

        duplicate_pct = (
            (duplicate_count / total_count * 100) if total_count > 0 else 0.0
        )
        logger.info(f"Found {duplicate_count} duplicates ({duplicate_pct:.2f}%)")

        return duplicate_count, duplicate_pct

    def validate_tensors(self, sample_size: int = 100) -> Tuple[bool, List[str]]:
        """
        Verify that tensors can be converted back to legal boards.

        Args:
            sample_size: Number of random positions to validate per split

        Returns:
            Tuple of (all_valid, error_messages)
        """
        errors = []

        with h5py.File(self.dataset_path, "r") as f:
            for split in ["train", "validation", "test"]:
                group = cast(h5py.Group, f[split])
                tensors_ds = cast(h5py.Dataset, group["tensors"])
                fens_ds = cast(h5py.Dataset, group["fen"])

                # Sample random indices
                total_size = tensors_ds.shape[0]
                if total_size == 0:
                    continue

                sample_indices = np.random.choice(
                    total_size, size=min(sample_size, total_size), replace=False
                )

                for idx in sample_indices:
                    tensor = tensors_ds[idx]
                    fen_bytes = fens_ds[idx]
                    fen = (
                        fen_bytes.decode()
                        if isinstance(fen_bytes, bytes)
                        else fen_bytes
                    )

                    # Verify shape
                    if tensor.shape != (18, 8, 8):
                        errors.append(f"{split}[{idx}]: Invalid shape {tensor.shape}")
                        continue

                    # Verify FEN is valid
                    try:
                        board = chess.Board(fen)
                    except ValueError as e:
                        errors.append(f"{split}[{idx}]: Invalid FEN: {e}")
                        continue

                    # Verify tensor matches FEN
                    reconstructed_tensor = board_to_tensor_18(board)
                    if not np.allclose(tensor, reconstructed_tensor, atol=1e-6):
                        errors.append(f"{split}[{idx}]: Tensor/FEN mismatch")

        all_valid = len(errors) == 0
        if all_valid:
            logger.info(f"All {sample_size * 3} sampled tensors valid")
        else:
            logger.warning(f"Found {len(errors)} tensor validation errors")

        return all_valid, errors

    def validate_split_balance(self) -> Dict[str, float]:
        """
        Check train/validation/test split percentages.

        Returns:
            Dict mapping split names to percentages
        """
        counts = {}

        with h5py.File(self.dataset_path, "r") as f:
            for split in ["train", "validation", "test"]:
                group = cast(h5py.Group, f[split])
                counts[split] = cast(h5py.Dataset, group["tensors"]).shape[0]

        total = sum(counts.values())
        if total == 0:
            return {"train": 0.0, "validation": 0.0, "test": 0.0}

        percentages = {split: count / total * 100 for split, count in counts.items()}

        logger.info(
            f"Split balance: train={percentages['train']:.1f}%, "
            f"val={percentages['validation']:.1f}%, test={percentages['test']:.1f}%"
        )

        return percentages

    def generate_report(
        self,
        output_path: Optional[Path] = None,
        check_duplicates: bool = True,
        duplicate_sample_size: Optional[int] = 10000,
    ) -> str:
        """
        Generate comprehensive validation report.

        Args:
            output_path: If provided, write report to file
            check_duplicates: Whether to run duplicate check (can be slow)
            duplicate_sample_size: Limit duplicate check to N positions

        Returns:
            Markdown-formatted report string
        """
        logger.info("Generating validation report...")

        # Run all validation checks
        distributions = self.validate_distributions()
        game_results = self.validate_game_results()
        split_balance = self.validate_split_balance()
        tensor_valid, tensor_errors = self.validate_tensors(sample_size=100)

        if check_duplicates:
            dup_count, dup_pct = self.check_duplicates(
                sample_size=duplicate_sample_size
            )
        else:
            dup_count, dup_pct = 0, 0.0

        # Build report
        total_positions = {
            split: distributions[split]["count"] for split in distributions
        }

        report_data = ValidationReport(
            total_positions=total_positions,
            eval_distribution=distributions,
            game_result_balance=game_results,
            duplicate_count=dup_count,
            duplicate_percentage=dup_pct,
            tensor_validity=tensor_valid,
            tensor_errors=tensor_errors,
            split_balance=split_balance,
        )

        report_md = report_data.to_markdown()

        # Write to file if requested
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_md)
            logger.info(f"Validation report written to: {output_path}")

        return report_md
