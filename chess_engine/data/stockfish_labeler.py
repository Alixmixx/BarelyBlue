"""
Stockfish integration for labeling positions with ground truth evaluations.

Uses Stockfish chess engine to evaluate positions at a specified depth,
providing high-quality training labels for the neural network.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import subprocess
import shutil

import chess

logger = logging.getLogger(__name__)


@dataclass
class StockfishEvaluation:
    """Evaluation result from Stockfish."""

    centipawn_score: Optional[int]  # None if mate score
    mate_in: Optional[int]  # None if centipawn score
    depth: int
    nodes: int

    @property
    def is_mate(self) -> bool:
        """Check if evaluation is a mate score."""
        return self.mate_in is not None

    def to_centipawns(self, clamp: int = 10000) -> int:
        """
        Convert evaluation to centipawns with clamping.

        Mate scores are converted to large values (±10000).

        Args:
            clamp: Maximum absolute centipawn value

        Returns:
            Centipawn evaluation
        """
        if self.is_mate:
            # Mate scores: positive for White winning, negative for Black winning
            if self.mate_in > 0:
                return clamp
            else:
                return -clamp
        else:
            return max(-clamp, min(clamp, self.centipawn_score))


class StockfishLabeler:
    """Label positions with Stockfish evaluations."""

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 15,
        threads: int = 1,
    ):
        """
        Initialize Stockfish labeler.

        Args:
            stockfish_path: Path to Stockfish binary (None = auto-detect)
            depth: Search depth for evaluation
            threads: Number of threads per Stockfish instance

        Raises:
            FileNotFoundError: If Stockfish binary not found
        """
        self.depth = depth
        self.threads = threads

        if stockfish_path is None:
            stockfish_path = self._find_stockfish()

        self.stockfish_path = stockfish_path

        if not Path(stockfish_path).exists():
            raise FileNotFoundError(
                f"Stockfish binary not found at: {stockfish_path}\n"
                "Install with: brew install stockfish (macOS) or apt install stockfish (Linux)"
            )

        logger.info(f"Initialized Stockfish labeler: {stockfish_path} (depth={depth})")

    def _find_stockfish(self) -> str:
        """
        Auto-detect Stockfish binary location.

        Returns:
            Path to Stockfish binary

        Raises:
            FileNotFoundError: If Stockfish not found
        """
        # Try common locations
        candidates = [
            "stockfish",
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
        ]

        for candidate in candidates:
            path = shutil.which(candidate)
            if path:
                return path

        raise FileNotFoundError(
            "Stockfish not found. Install with: brew install stockfish (macOS) "
            "or apt install stockfish (Linux)"
        )

    def evaluate_position(self, board: chess.Board) -> StockfishEvaluation:
        """
        Evaluate a single position.

        Args:
            board: Chess position to evaluate

        Returns:
            StockfishEvaluation with score and metadata
        """
        process = subprocess.Popen(
            [self.stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        # Send UCI commands
        commands = [
            "uci",
            f"setoption name Threads value {self.threads}",
            "isready",
            f"position fen {board.fen()}",
            f"go depth {self.depth}",
        ]

        for cmd in commands:
            process.stdin.write(cmd + "\n")
            process.stdin.flush()

        centipawn_score = None
        mate_in = None
        depth = 0
        nodes = 0

        for line in process.stdout:
            line = line.strip()

            # Parse info lines
            if line.startswith("info") and "score" in line:
                parts = line.split()

                if "depth" in parts:
                    depth_idx = parts.index("depth") + 1
                    depth = int(parts[depth_idx])

                if "nodes" in parts:
                    nodes_idx = parts.index("nodes") + 1
                    nodes = int(parts[nodes_idx])

                if "score" in parts:
                    score_idx = parts.index("score") + 1
                    score_type = parts[score_idx]

                    if score_type == "cp":
                        centipawn_score = int(parts[score_idx + 1])
                    elif score_type == "mate":
                        mate_in = int(parts[score_idx + 1])

            if line.startswith("bestmove"):
                break

        process.stdin.write("quit\n")
        process.stdin.flush()
        process.wait(timeout=1.0)

        return StockfishEvaluation(
            centipawn_score=centipawn_score,
            mate_in=mate_in,
            depth=depth,
            nodes=nodes,
        )

    def evaluate_batch(
        self, positions: List[chess.Board], num_workers: int = 1
    ) -> List[StockfishEvaluation]:
        """
        Evaluate multiple positions.

        Args:
            positions: List of chess positions
            num_workers: Number of parallel workers (not used yet)

        Returns:
            List of evaluations
        """
        # TODO: implement parallel execution
        if num_workers > 1:
            logger.warning(
                f"Parallel evaluation not yet implemented. Using sequential evaluation."
            )

        evaluations = []
        for board in positions:
            eval_result = self.evaluate_position(board)
            evaluations.append(eval_result)

        return evaluations
