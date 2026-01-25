"""
Tests for Stockfish labeling module.

Tests Stockfish integration for evaluating chess positions.
"""

import pytest
import chess

from chess_engine.data.stockfish_labeler import (
    StockfishLabeler,
    StockfishEvaluation,
)


@pytest.fixture
def stockfish_labeler():
    """Create a Stockfish labeler with low depth for fast tests."""
    try:
        labeler = StockfishLabeler(depth=10)
        return labeler
    except FileNotFoundError:
        pytest.skip("Stockfish not installed")


class TestStockfishEvaluation:
    """Test StockfishEvaluation dataclass."""

    def test_centipawn_evaluation(self):
        """Test centipawn evaluation."""
        eval_result = StockfishEvaluation(
            centipawn_score=150, mate_in=None, depth=10, nodes=1000
        )

        assert not eval_result.is_mate
        assert eval_result.to_centipawns() == 150

    def test_mate_evaluation_positive(self):
        """Test mate evaluation (White winning)."""
        eval_result = StockfishEvaluation(
            centipawn_score=None, mate_in=3, depth=10, nodes=1000
        )

        assert eval_result.is_mate
        assert eval_result.to_centipawns() == 10000  # Clamped to max

    def test_mate_evaluation_negative(self):
        """Test mate evaluation (Black winning)."""
        eval_result = StockfishEvaluation(
            centipawn_score=None, mate_in=-2, depth=10, nodes=1000
        )

        assert eval_result.is_mate
        assert eval_result.to_centipawns() == -10000  # Clamped to min

    def test_clamping_positive(self):
        """Test clamping of large positive evaluations."""
        eval_result = StockfishEvaluation(
            centipawn_score=15000, mate_in=None, depth=10, nodes=1000
        )

        assert eval_result.to_centipawns(clamp=5000) == 5000

    def test_clamping_negative(self):
        """Test clamping of large negative evaluations."""
        eval_result = StockfishEvaluation(
            centipawn_score=-15000, mate_in=None, depth=10, nodes=1000
        )

        assert eval_result.to_centipawns(clamp=5000) == -5000


class TestStockfishLabeler:
    """Test StockfishLabeler class."""

    def test_initialization(self, stockfish_labeler):
        """Test Stockfish initialization."""
        assert stockfish_labeler.depth == 10
        assert stockfish_labeler.stockfish_path is not None

    def test_starting_position(self, stockfish_labeler):
        """Test evaluation of starting position."""
        board = chess.Board()
        eval_result = stockfish_labeler.evaluate_position(board)

        # Starting position should be roughly equal (within ±50 centipawns)
        assert eval_result.centipawn_score is not None
        assert abs(eval_result.centipawn_score) < 50
        assert eval_result.depth == 10
        assert eval_result.nodes > 0

    def test_white_advantage(self, stockfish_labeler):
        """Test position with clear White advantage."""
        # Position after 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 (White wins piece)
        fen = "r1bqkbnr/1ppp1ppp/p1B5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4"
        board = chess.Board(fen)
        eval_result = stockfish_labeler.evaluate_position(board)

        # White should have significant advantage (won a knight)
        assert eval_result.centipawn_score > 200

    def test_black_advantage(self, stockfish_labeler):
        """Test position with clear Black advantage."""
        # Mirror of previous position (Black won piece)
        fen = "rnbqk2r/pppp1ppp/5n2/4p3/4P3/P1b5/1PPP1PPP/R1BQKBNR w KQkq - 0 4"
        board = chess.Board(fen)
        eval_result = stockfish_labeler.evaluate_position(board)

        # Black should have significant advantage
        assert eval_result.centipawn_score < -200

    def test_mate_in_one(self, stockfish_labeler):
        """Test detection of mate in one."""
        # Back rank mate: Qh8# is mate
        fen = "6k1/5ppp/8/8/8/8/5PPP/4Q1K1 w - - 0 1"
        board = chess.Board(fen)
        eval_result = stockfish_labeler.evaluate_position(board)

        # Should find mate
        assert eval_result.is_mate
        assert eval_result.mate_in > 0  # White is mating

    def test_getting_mated(self, stockfish_labeler):
        """Test detection of getting mated."""
        # Black has back rank mate threat
        fen = "4q1k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"
        board = chess.Board(fen)
        eval_result = stockfish_labeler.evaluate_position(board)

        # Should detect Black is mating (or very large negative score)
        # Might be mate_in or just very negative centipawn
        if eval_result.is_mate:
            assert eval_result.mate_in < 0  # Black is mating
        else:
            assert eval_result.centipawn_score < -500

    def test_batch_evaluation(self, stockfish_labeler):
        """Test evaluating multiple positions."""
        positions = [
            chess.Board(),  # Starting position
            chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),  # After e4
        ]

        evaluations = stockfish_labeler.evaluate_batch(positions)

        assert len(evaluations) == 2
        assert all(isinstance(e, StockfishEvaluation) for e in evaluations)

        # Both positions should be roughly equal
        assert abs(evaluations[0].centipawn_score) < 50
        assert abs(evaluations[1].centipawn_score) < 50

    def test_evaluation_consistency(self, stockfish_labeler):
        """Test that same position evaluates consistently."""
        board = chess.Board()
        eval1 = stockfish_labeler.evaluate_position(board)
        eval2 = stockfish_labeler.evaluate_position(board)

        # Should be identical (deterministic at same depth)
        assert eval1.centipawn_score == eval2.centipawn_score
        assert eval1.mate_in == eval2.mate_in


class TestStockfishAutoDetection:
    """Test auto-detection of Stockfish binary."""

    def test_auto_detect_stockfish(self):
        """Test that Stockfish can be auto-detected."""
        try:
            labeler = StockfishLabeler()  # No path provided
            assert labeler.stockfish_path is not None
        except FileNotFoundError:
            pytest.skip("Stockfish not installed")

    def test_invalid_path_raises_error(self):
        """Test that invalid path raises error."""
        with pytest.raises(FileNotFoundError):
            StockfishLabeler(stockfish_path="/nonexistent/stockfish")
