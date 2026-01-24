"""
Tests for position extraction module.
"""

import pytest
import chess
import chess.pgn
from io import StringIO

from chess_engine.data.position_extractor import (
    PositionExtractor,
    ExtractedPosition,
    count_positions_in_game,
)


@pytest.fixture
def sample_game():
    """Create a sample chess game for testing."""
    pgn = """
[Event "Test Game"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5
7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 c5 11. d4 Qc7 12. Nbd2 Bd7 1-0
"""
    return chess.pgn.read_game(StringIO(pgn))


@pytest.fixture
def short_game():
    """Create a short game (< 10 plies) for testing."""
    pgn = """
[Event "Short Game"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Qxf7# 1-0
"""
    return chess.pgn.read_game(StringIO(pgn))


@pytest.fixture
def draw_game():
    """Create a drawn game for testing."""
    pgn = """
[Event "Draw Game"]
[White "Player1"]
[Black "Player2"]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1/2-1/2
"""
    return chess.pgn.read_game(StringIO(pgn))


class TestPositionExtractor:
    """Test suite for PositionExtractor class."""

    def test_extract_basic(self, sample_game):
        """Test basic position extraction."""
        extractor = PositionExtractor(min_ply=10, max_ply=100)
        positions = list(extractor.extract_positions(sample_game))

        # Game has 24 plies (12 full moves)
        # Should extract plies 10-24 (15 positions)
        assert len(positions) > 0
        assert all(pos.ply >= 10 for pos in positions)

    def test_game_result_white_win(self, sample_game):
        """Test game result parsing for White win."""
        extractor = PositionExtractor(min_ply=0, max_ply=100)
        positions = list(extractor.extract_positions(sample_game))

        # All positions should have game_result = 1 (White win)
        assert all(pos.game_result == 1 for pos in positions)

    def test_game_result_draw(self, draw_game):
        """Test game result parsing for draw."""
        extractor = PositionExtractor(min_ply=0, max_ply=100)
        positions = list(extractor.extract_positions(draw_game))

        # All positions should have game_result = 0 (draw)
        assert all(pos.game_result == 0 for pos in positions)

    def test_min_ply_filter(self, short_game):
        """Test minimum ply filter."""
        # Extract with min_ply = 10 (game only has 5 plies)
        extractor = PositionExtractor(min_ply=10, max_ply=100)
        positions = list(extractor.extract_positions(short_game))

        # Should extract no positions
        assert len(positions) == 0

        # Extract with min_ply = 2
        extractor = PositionExtractor(min_ply=2, max_ply=100)
        positions = list(extractor.extract_positions(short_game))

        # Should extract plies 2-5
        assert len(positions) > 0
        assert all(pos.ply >= 2 for pos in positions)

    def test_max_ply_filter(self, sample_game):
        """Test maximum ply filter."""
        extractor = PositionExtractor(min_ply=0, max_ply=15)
        positions = list(extractor.extract_positions(sample_game))

        # Should only extract positions up to ply 15
        assert all(pos.ply <= 15 for pos in positions)

    def test_min_pieces_filter(self):
        """Test minimum pieces filter."""
        # Create endgame position with 4 pieces (2 kings + 2 pawns)
        # Use valid king and pawn endgame
        pgn = """
[Event "Endgame"]
[White "Player1"]
[Black "Player2"]
[Result "1/2-1/2"]
[FEN "8/8/8/3k4/3P4/3K4/8/8 w - - 0 1"]

1. Kd2 Kd6 2. Ke3 Ke7 3. Kf4 Kf6 1/2-1/2
"""
        game = chess.pgn.read_game(StringIO(pgn))

        # With min_pieces=6, should extract no positions (only 4 pieces)
        extractor = PositionExtractor(min_ply=0, max_ply=100, min_pieces=6)
        positions = list(extractor.extract_positions(game))
        assert len(positions) == 0

        # With min_pieces=3, should extract positions
        extractor = PositionExtractor(min_ply=0, max_ply=100, min_pieces=3)
        positions = list(extractor.extract_positions(game))
        assert len(positions) > 0

    def test_skip_terminal_positions(self):
        """Test that checkmate positions are skipped."""
        pgn = """
[Event "Mate Game"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. f3 e5 2. g4 Qh4# 1-0
"""
        game = chess.pgn.read_game(StringIO(pgn))

        extractor = PositionExtractor(min_ply=0, max_ply=100)
        positions = list(extractor.extract_positions(game))

        # Final position (checkmate) should be skipped
        # Game has 4 plies, but checkmate position is excluded
        assert all(not pos.board.is_checkmate() for pos in positions)

    def test_extracted_position_fields(self, sample_game):
        """Test that extracted positions have all required fields."""
        extractor = PositionExtractor(min_ply=10, max_ply=15)
        positions = list(extractor.extract_positions(sample_game, game_id="test123"))

        for pos in positions:
            assert isinstance(pos.board, chess.Board)
            assert pos.game_result in [-1, 0, 1]
            assert isinstance(pos.ply, int)
            assert isinstance(pos.fen, str)
            assert pos.game_id == "test123"
            assert pos.board.fen() == pos.fen

    def test_board_is_legal(self, sample_game):
        """Test that extracted boards are legal positions."""
        extractor = PositionExtractor(min_ply=0, max_ply=100)
        positions = list(extractor.extract_positions(sample_game))

        for pos in positions:
            # Board should be valid
            assert pos.board.is_valid()
            # Should have legal moves (not terminal)
            assert list(pos.board.legal_moves)

    def test_deduplication(self):
        """Test position deduplication using Zobrist hashing."""
        # Create positions with duplicates
        board1 = chess.Board()
        board2 = chess.Board()  # Same as board1
        board3 = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")

        positions = [
            ExtractedPosition(board1, 0, 1, board1.fen(), "game1"),
            ExtractedPosition(board2, 0, 1, board2.fen(), "game2"),  # Duplicate
            ExtractedPosition(board3, 0, 2, board3.fen(), "game3"),
        ]

        extractor = PositionExtractor()
        unique_positions = list(extractor.deduplicate(positions))

        # Should have 2 unique positions (board1/board2 are same)
        assert len(unique_positions) == 2

    def test_zobrist_hash_consistency(self):
        """Test that Zobrist hashing is consistent."""
        board = chess.Board()
        pos1 = ExtractedPosition(board, 0, 1, board.fen(), "game1")
        pos2 = ExtractedPosition(board, 0, 1, board.fen(), "game2")

        # Same board should have same hash
        assert pos1.get_zobrist_hash() == pos2.get_zobrist_hash()


class TestHelperFunctions:
    """Test helper functions."""

    def test_count_positions_in_game(self, sample_game):
        """Test counting positions in a game."""
        count = count_positions_in_game(sample_game)

        # Game has 24 plies (12 full moves)
        assert count == 24

    def test_count_positions_short_game(self, short_game):
        """Test counting positions in short game."""
        count = count_positions_in_game(short_game)

        # Short game has 5 plies
        assert count == 5
