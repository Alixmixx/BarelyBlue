"""
Abstract Evaluator Interface

This module defines the abstract base class for all position evaluators.
By defining a common interface, we can swap between
evaluators without modifying the search algorithm.

Key Principles:
    1. Evaluators are stateless
    2. evaluate() always returns centipawns from White's perspective
    3. Positive = White advantage, Negative = Black advantage
    4. Checkmate positions should return ±INFINITY

Convention:
    - Material values in centipawns (1/100th of a pawn, pawn = 100, queen = 900)
    - Return 0 for perfectly equal positions
    - Return values are from White's perspective (negate for Black)
"""

from abc import ABC, abstractmethod
import chess
from typing import Optional


# Evaluation constants
INFINITY = 100000  # Represents certain victory/defeat
MATE_SCORE = 50000  # Base score for checkmate


class Evaluator(ABC):
    """
    Abstract base class for position evaluation.

    All evaluator implementations must inherit from this class and implement
    the evaluate() method. This ensures compatibility with the search algorithm.

    Attributes:
        None (evaluators should be stateless)

    Methods:
        evaluate(board): Returns position evaluation in centipawns
    """

    @abstractmethod
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a chess position from White's perspective.

        Args:
            board: python-chess Board object to evaluate

        Returns:
            float: Evaluation in centipawns

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        pass

    def is_draw(self, board: chess.Board) -> bool:
        """
        Check if position is a draw by rule.

        Helper method to detect draws that don't require evaluation:
            - Stalemate
            - Insufficient material
            - Fifty-move rule
            - Threefold repetition

        Args:
            board: python-chess Board object

        Returns:
            bool: True if position is drawn, False otherwise
        """
        return (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_fifty_moves()
            or board.can_claim_threefold_repetition()
        )

    def evaluate_terminal(self, board: chess.Board, ply_from_root: int = 0) -> Optional[float]:
        """
        Evaluate terminal positions (checkmate, stalemate, draw).

        This is a helper method that search algorithms can call to
        know when they can stop searching.

        Args:
            board: python-chess Board object
            ply_from_root: Distance from root (for mate distance calculation)

        Returns:
            float: Evaluation if terminal position
            None: If position is not terminal
        """
        if board.is_checkmate():
            # Negative if current player is checkmated (they lose)
            # Prefer faster mates (subtract ply from root)
            if board.turn == chess.WHITE:
                return -(MATE_SCORE - ply_from_root)  # White is mated
            else:
                return MATE_SCORE - ply_from_root  # Black is mated

        if self.is_draw(board):
            return 0.0

        return None

    def __repr__(self) -> str:
        """String representation of evaluator."""
        return f"{self.__class__.__name__}()"
