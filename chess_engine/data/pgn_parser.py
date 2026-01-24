"""
PGN Parser for streaming large chess databases.
"""

import logging
from pathlib import Path
from typing import Iterator, Optional

import chess.pgn

logger = logging.getLogger(__name__)


class PGNParser:
    """Stream large PGN files efficiently without loading into memory."""

    def __init__(self, min_elo: int = 2000, max_games: Optional[int] = None):
        """
        Initialize PGN parser with filtering criteria.

        Args:
            min_elo: Minimum ELO rating for both players (default: 2000)
            max_games: Maximum number of games to parse (None = unlimited)
        """
        self.min_elo = min_elo
        self.max_games = max_games
        self._games_parsed = 0

    def parse_file(self, pgn_path: Path) -> Iterator[chess.pgn.Game]:
        """
        Stream games from a single PGN file.

        Args:
            pgn_path: Path to PGN file

        Yields:
            chess.pgn.Game objects that pass ELO filter
        """
        if not pgn_path.exists():
            raise FileNotFoundError(f"PGN file not found: {pgn_path}")

        logger.info(f"Parsing PGN file: {pgn_path}")
        self._games_parsed = 0

        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_file:
            while True:
                if self.max_games is not None and self._games_parsed >= self.max_games:
                    logger.info(f"Reached max_games limit: {self.max_games}")
                    break

                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break

                    if self._passes_elo_filter(game):
                        self._games_parsed += 1
                        yield game

                except Exception as e:
                    logger.warning(f"Error parsing game: {e}")
                    continue

        logger.info(f"Parsed {self._games_parsed} games from {pgn_path}")

    def parse_directory(self, dir_path: Path) -> Iterator[chess.pgn.Game]:
        """
        Stream games from all PGN files in a directory.

        Args:
            dir_path: Path to directory containing .pgn files

        Yields:
            chess.pgn.Game objects that pass ELO filter
        """
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        logger.info(f"Parsing PGN files from directory: {dir_path}")

        pgn_files = sorted(dir_path.glob("*.pgn"))
        logger.info(f"Found {len(pgn_files)} PGN files")

        for pgn_file in pgn_files:
            yield from self.parse_file(pgn_file)

    def _passes_elo_filter(self, game: chess.pgn.Game) -> bool:
        """
        Check if a game passes the ELO filter.

        Both White and Black players must have ELO >= min_elo.

        Args:
            game: Chess game to check

        Returns:
            True if game passes filter, False otherwise
        """
        try:
            white_elo = game.headers.get("WhiteElo", "?")
            black_elo = game.headers.get("BlackElo", "?")

            if white_elo == "?" or black_elo == "?":
                return False

            white_elo = int(white_elo)
            black_elo = int(black_elo)

            return white_elo >= self.min_elo and black_elo >= self.min_elo

        except (ValueError, TypeError):
            return False

    def get_games_parsed(self) -> int:
        """Get count of games parsed so far."""
        return self._games_parsed
