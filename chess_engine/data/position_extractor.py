"""
Position extraction from chess games.
"""

import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Set

import chess
import chess.pgn

from chess_engine.search.transposition import zobrist_hash

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPosition:
    """A position extracted from a chess game."""

    board: chess.Board
    game_result: int  # 1 (White win), 0 (draw), -1 (Black win)
    ply: int
    fen: str
    game_id: str

    def __post_init__(self):
        """Compute zobrist hash for deduplication."""
        self._zobrist_hash = zobrist_hash(self.board)

    def get_zobrist_hash(self) -> int:
        """Get cached Zobrist hash."""
        return self._zobrist_hash


class PositionExtractor:
    """Extract positions from chess games with quality filtering."""

    def __init__(
        self,
        min_ply: int = 10,
        max_ply: int = 100,
        min_pieces: int = 6,
    ):
        """
        Initialize position extractor.

        Args:
            min_ply: Minimum ply to start extracting
            max_ply: Maximum ply to extract
            min_pieces: Minimum piece count
        """
        self.min_ply = min_ply
        self.max_ply = max_ply
        self.min_pieces = min_pieces

    def extract_positions(
        self, game: chess.pgn.Game, game_id: Optional[str] = None
    ) -> Iterator[ExtractedPosition]:
        """
        Extract positions from a single game.

        Args:
            game: Chess game to extract from
            game_id: Optional unique identifier for the game

        Yields:
            ExtractedPosition objects that pass quality filters
        """
        result_str = game.headers.get("Result", "*")
        game_result = self._parse_result(result_str)

        if game_result is None:
            return

        if game_id is None:
            game_id = f"{game.headers.get('Site', 'unknown')}"

        # Iterate through game moves
        board = game.board()
        ply = 0

        for move in game.mainline_moves():
            board.push(move)
            ply += 1

            # Apply quality filters
            if not self._passes_filters(board, ply):
                continue

            position = ExtractedPosition(
                board=board.copy(),
                game_result=game_result,
                ply=ply,
                fen=board.fen(),
                game_id=game_id,
            )

            yield position

    def deduplicate(
        self, positions: Iterable[ExtractedPosition]
    ) -> Iterator[ExtractedPosition]:
        """
        Remove duplicate positions using Zobrist hashing.

        Args:
            positions: Iterable of positions (may contain duplicates)

        Yields:
            Unique positions
        """
        seen_hashes: Set[int] = set()
        duplicate_count = 0

        for position in positions:
            position_hash = position.get_zobrist_hash()

            if position_hash not in seen_hashes:
                seen_hashes.add(position_hash)
                yield position
            else:
                duplicate_count += 1

        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate positions")

    def _passes_filters(self, board: chess.Board, ply: int) -> bool:
        """
        Check if a position passes quality filters.

        Args:
            board: Chess position to check
            ply: Ply count

        Returns:
            True if position should be extracted
        """
        # Filter by ply range
        if ply < self.min_ply or ply > self.max_ply:
            return False

        # Filter by piece count
        piece_count = len(board.piece_map())
        if piece_count < self.min_pieces:
            return False

        # Skip terminal positions (checkmate, stalemate)
        if board.is_game_over():
            return False

        return True

    def _parse_result(self, result_str: str) -> Optional[int]:
        """
        Parse PGN result string to integer.

        Args:
            result_str: PGN result ("1-0", "0-1", "1/2-1/2", "*")

        Returns:
            1 (White win), 0 (draw), -1 (Black win)
        """
        if result_str == "1-0":
            return 1
        elif result_str == "0-1":
            return -1
        elif result_str == "1/2-1/2":
            return 0
        else:
            return None


def count_positions_in_game(game: chess.pgn.Game) -> int:
    """
    Count total positions in a game (for statistics).

    Args:
        game: Chess game

    Returns:
        Number of positions
    """
    return sum(1 for _ in game.mainline_moves())
