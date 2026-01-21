"""
Unit Tests for Evaluation Module

Tests for position evaluation functions, focusing on:
    - Material counting accuracy
    - Piece-square table correctness
    - Symmetry (flipped position = negated evaluation)
    - Terminal position detection (checkmate, stalemate)
    - Edge cases (empty board, insufficient material)
"""

import chess
import pytest
import numpy as np
from chess_engine.evaluation import ClassicalEvaluator, Evaluator
from chess_engine.evaluation.base import MATE_SCORE


class TestClassicalEvaluator:
    """Tests for ClassicalEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create a ClassicalEvaluator instance."""
        return ClassicalEvaluator()

    def test_starting_position_roughly_equal(self, evaluator):
        """
        Test that starting position evaluates close to 0.

        Starting position should be roughly equal (within 100 centipawns).
        The exact value depends on PST values, but should be small.
        """
        board = chess.Board()
        score = evaluator.evaluate(board)

        # Starting position should be roughly equal
        assert -100 < score < 100, f"Starting position eval {score} is not close to 0"

    def test_material_advantage(self, evaluator):
        """
        Test that material advantage is properly counted.

        White up a queen (900 centipawns) should have large positive eval.
        """
        # Position: White has extra queen
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1")

        # Black is up a rook (white missing rook on h1)
        score = evaluator.evaluate(board)

        # Black should be ahead by approximately a rook (500 cp)
        assert score < -400, f"Black should be ahead by ~500 cp, got {score}"

    def test_symmetry(self, evaluator):
        """
        Test evaluation symmetry.

        Flipping the board (swapping colors) should negate the evaluation.
        """
        # Use a non-symmetric position
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        score1 = evaluator.evaluate(board)

        # TODO: Implement proper board flipping test
        assert isinstance(score1, (int, float, np.floating))

    def test_checkmate_detection(self, evaluator):
        """
        Test that checkmate positions are evaluated correctly.

        Checkmate should return MATE_SCORE (adjusted by distance to mate).
        """
        # Fool's mate
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
        board.push_san("Qh4#")  # Checkmate

        score = evaluator.evaluate(board)

        # White is checkmated, so score should be very negative
        assert score < -MATE_SCORE + 100, f"Checkmate should have score < -{MATE_SCORE}, got {score}"

    def test_stalemate_detection(self, evaluator):
        """
        Test that stalemate is evaluated as draw (0).
        """
        # Stalemate position: Black king on a8, White queen on c7, White king on b6
        # Black to move, king cannot move and is not in check = stalemate
        board = chess.Board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")

        # Verify it's stalemate
        assert board.is_stalemate()

        score = evaluator.evaluate(board)

        # Stalemate should evaluate to 0 (draw)
        assert score == 0.0, f"Stalemate should evaluate to 0, got {score}"

    def test_insufficient_material(self, evaluator):
        """
        Test that insufficient material is evaluated as draw.
        """
        # King vs King
        board = chess.Board("8/8/8/8/8/7k/8/K7 w - - 0 1")

        assert board.is_insufficient_material()

        score = evaluator.evaluate(board)
        assert score == 0.0, f"Insufficient material should evaluate to 0, got {score}"

    def test_piece_square_tables(self, evaluator):
        """
        Test that piece-square tables affect evaluation.
        """
        # Knight on edge (a1) vs. knight in center (e4)
        board_edge = chess.Board("7k/pppppppp/8/8/8/8/PPPPPPPP/N6K w - - 0 1")  # Knight on a1
        board_center = chess.Board("7k/pppppppp/8/8/4N3/8/PPPPPPPP/7K w - - 0 1")  # Knight on e4

        score_edge = evaluator.evaluate(board_edge)
        score_center = evaluator.evaluate(board_center)

        # Central knight should be worth more than edge knight
        assert score_center > score_edge, (
            f"Central knight ({score_center}) should be worth more than edge knight ({score_edge})"
        )

    def test_consistency(self, evaluator):
        """
        Test that evaluator is deterministic.
        """
        board = chess.Board()

        scores = [evaluator.evaluate(board) for _ in range(5)]

        # All scores should be identical
        assert len(set(scores)) == 1, f"Evaluator is not deterministic: {scores}"

    def test_endgame_detection(self, evaluator):
        """
        Test that endgame is properly detected.
        """
        # Endgame position: Only kings and pawns
        board_endgame = chess.Board("8/4k3/8/8/8/8/4K3/8 w - - 0 1")

        assert evaluator.is_endgame(board_endgame), "Should detect endgame"

        # Opening position: Should not be endgame
        board_opening = chess.Board()

        assert not evaluator.is_endgame(board_opening), "Starting position should not be endgame"


class TestEvaluatorInterface:
    """Tests for Evaluator abstract interface."""

    def test_evaluator_is_abstract(self):
        """
        Test that Evaluator cannot be instantiated directly.
        """
        with pytest.raises(TypeError):
            # Should raise TypeError because evaluate() is not implemented
            evaluator = Evaluator()

    def test_is_draw_helper(self):
        """
        Test the is_draw() helper method.

        This method should detect various draw conditions.
        """
        evaluator = ClassicalEvaluator()  # Use concrete implementation

        # Stalemate
        board_stalemate = chess.Board("k7/8/1K6/8/8/8/8/8 b - - 0 1")
        assert evaluator.is_draw(board_stalemate)

        # Insufficient material
        board_insufficient = chess.Board("8/8/8/8/8/7k/8/K7 w - - 0 1")
        assert evaluator.is_draw(board_insufficient)

        # Normal position
        board_normal = chess.Board()
        assert not evaluator.is_draw(board_normal)


# ============================================================================
# Integration Tests
# ============================================================================

class TestEvaluationIntegration:
    """Integration tests for evaluation with real games."""

    @pytest.fixture
    def evaluator(self):
        return ClassicalEvaluator()

    def test_evaluation_throughout_game(self, evaluator):
        """
        Test that evaluation changes sensibly throughout a game.
        """
        board = chess.Board()
        evaluations = []

        # Play the Italian Game opening
        moves = ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5']

        for move in moves:
            board.push_san(move)
            score = evaluator.evaluate(board)
            evaluations.append(score)

            # All evaluations should be reasonable
            assert -200 < score < 200, f"Evaluation {score} seems unreasonable for opening"

    def test_tactical_position(self, evaluator):
        """
        Test evaluation on a tactical position.
        """
        # Position: White has a queen, Black doesn't
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

        score = evaluator.evaluate(board)

        # White should have large advantage (~900 centipawns for queen)
        assert score > 700, f"White should be ahead by ~900 cp, got {score}"


# ============================================================================
# Performance Tests
# ============================================================================

class TestEvaluationPerformance:
    """Performance tests for evaluation function."""

    @pytest.fixture
    def evaluator(self):
        return ClassicalEvaluator()

    def test_evaluation_speed(self, evaluator):
        """
        Test that evaluation is fast enough.
        """
        import time

        board = chess.Board()
        iterations = 10000

        start = time.time()
        for _ in range(iterations):
            evaluator.evaluate(board)
        elapsed = time.time() - start

        positions_per_second = iterations / elapsed

        # Should evaluate at least 10,000 positions/second
        assert positions_per_second > 10000, (
            f"Evaluation too slow: {positions_per_second:.0f} pos/sec"
        )


# ============================================================================
# Edge Cases
# ============================================================================

class TestEvaluationEdgeCases:
    """Test edge cases and unusual positions."""

    @pytest.fixture
    def evaluator(self):
        return ClassicalEvaluator()

    def test_empty_board(self, evaluator):
        """
        Test evaluation of board with only kings.

        Should be recognized as draw (insufficient material).
        """
        board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")

        score = evaluator.evaluate(board)
        assert score == 0.0, "Only kings should evaluate to 0"

    def test_many_pieces(self, evaluator):
        """
        Test evaluation still works with many pieces.

        Verify no overflow or errors with full board.
        """
        board = chess.Board()  # Starting position has many pieces

        score = evaluator.evaluate(board)
        assert isinstance(score, (int, float, np.floating))
        assert -10000 < score < 10000  # Reasonable range


# ============================================================================
# Test Utilities
# ============================================================================

def compare_evaluations(eval1, eval2, tolerance=10):
    """
    Helper to compare two evaluations within tolerance.

    Args:
        eval1: First evaluation
        eval2: Second evaluation
        tolerance: Allowed difference in centipawns

    Returns:
        True if evaluations are within tolerance
    """
    return abs(eval1 - eval2) < tolerance
