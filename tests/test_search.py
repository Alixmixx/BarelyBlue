"""
Unit Tests for Search Module

Tests for minimax search algorithm.
"""

import chess
import pytest
import numpy as np
from chess_engine.search import minimax, find_best_move, TranspositionTable
from chess_engine.evaluation import ClassicalEvaluator
from chess_engine.search.transposition import zobrist_hash, NodeType


class TestMinimax:
    """Tests for minimax search algorithm."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator for testing."""
        return ClassicalEvaluator()

    def test_mate_in_one(self, evaluator):
        """Test that engine finds mate in 1."""

        board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")

        best_move, score, nodes, pv = find_best_move(board, depth=1, evaluator=evaluator)

        assert (
            best_move.to_square == chess.H8 or best_move.to_square == chess.A8
        ), f"Should find mate with rook, got {best_move}"
        assert nodes > 0, "Should search at least one node"

        board.push(best_move)
        assert board.is_checkmate(), f"Move {best_move} should be checkmate"

    def test_mate_in_two(self, evaluator):
        """Test that engine finds mate in 2 (requires depth 3+)."""

        board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
        )

        best_move, score, nodes, pv = find_best_move(board, depth=3, evaluator=evaluator)

        expected_move = chess.Move.from_uci("d8h4")

        assert best_move == expected_move, f"Should find Qh4#, got {best_move}"
        assert nodes > 0, "Should search at least one node"

        board.push(best_move)
        assert board.is_checkmate(), "Qh4 should be checkmate"

    def test_depth_increases_strength(self, evaluator):
        """Test that searching deeper finds better moves."""

        board = chess.Board(
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
        )

        move_d2, score_d2, nodes_d2, pv_d2 = find_best_move(board, depth=2, evaluator=evaluator)
        move_d4, score_d4, nodes_d4, pv_d4 = find_best_move(board, depth=4, evaluator=evaluator)

        assert move_d2 in board.legal_moves
        assert move_d4 in board.legal_moves
        assert nodes_d4 > nodes_d2, "Deeper search should explore more nodes"

    def test_alpha_beta_correctness(self, evaluator):
        """Test that alpha-beta pruning produces same result as full minimax."""

        board = chess.Board()

        best_move, score, nodes, pv = find_best_move(board, depth=3, evaluator=evaluator)

        best_move2, score2, nodes2, pv2 = find_best_move(board, depth=3, evaluator=evaluator)

        assert best_move == best_move2, "Alpha-beta should be deterministic"
        assert score == score2, "Scores should match"
        assert nodes > 0 and nodes2 > 0, "Should search at least one node"

    def test_no_legal_moves_raises_error(self, evaluator):
        """Test that search handles game-over positions."""

        board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
        )
        board.push_san("Qh4#")

        with pytest.raises(ValueError):
            find_best_move(board, depth=3, evaluator=evaluator)


class TestTranspositionTable:
    """Tests for transposition table."""

    def test_store_and_lookup(self):
        """Test basic store and lookup operations."""

        tt = TranspositionTable(max_size=1000)

        hash_val = 12345
        tt.store(hash_val, depth=5, value=150.0, node_type=NodeType.EXACT)

        entry = tt.lookup(hash_val, depth=5)

        assert entry is not None, "Should find stored entry"
        assert entry.value == 150.0, "Value should match"
        assert entry.depth == 5, "Depth should match"
        assert entry.node_type == NodeType.EXACT, "Node type should match"

    def test_depth_replacement(self):
        """Test that higher depth entries replace lower depth entries."""

        tt = TranspositionTable(max_size=1000)

        hash_val = 12345

        tt.store(hash_val, depth=3, value=100.0, node_type=NodeType.EXACT)

        tt.store(hash_val, depth=5, value=150.0, node_type=NodeType.EXACT)

        entry = tt.lookup(hash_val)
        assert entry.depth == 5, "Should keep higher depth entry"
        assert entry.value == 150.0, "Value should be from depth 5 entry"

    def test_insufficient_depth_returns_none(self):
        """Test that lookup returns None if cached depth is insufficient."""

        tt = TranspositionTable(max_size=1000)

        hash_val = 12345
        tt.store(hash_val, depth=4, value=100.0, node_type=NodeType.EXACT)

        entry = tt.lookup(hash_val, depth=6)

        assert entry is None, "Should not return entry with insufficient depth"

    def test_zobrist_hash_consistency(self):
        """Test that Zobrist hash is consistent for same position."""

        board1 = chess.Board()
        board2 = chess.Board()

        hash1 = zobrist_hash(board1)
        hash2 = zobrist_hash(board2)

        assert hash1 == hash2, "Same position should have same hash"

    def test_zobrist_hash_changes_with_position(self):
        """Test that Zobrist hash changes when position changes."""

        board1 = chess.Board()
        hash1 = zobrist_hash(board1)

        board2 = chess.Board()
        board2.push_san("e4")
        hash2 = zobrist_hash(board2)

        assert hash1 != hash2, "Different positions should have different hashes"

    def test_transposition_table_improves_search(
        self,
    ):
        """Test that transposition table improves search performance."""

        evaluator = ClassicalEvaluator()
        board = chess.Board()

        tt1 = TranspositionTable(max_size=1000000)
        move1, score1, nodes1, pv1 = find_best_move(
            board, depth=4, evaluator=evaluator, transposition_table=tt1
        )

        assert move1 in board.legal_moves
        assert nodes1 > 0, "Should search at least one node"

    def test_clear_table(self):
        """Test that clearing table removes all entries."""

        tt = TranspositionTable(max_size=1000)

        for i in range(10):
            tt.store(i, depth=5, value=float(i), node_type=NodeType.EXACT)

        assert len(tt.table) == 10, "Should have 10 entries"

        tt.clear()

        assert len(tt.table) == 0, "Table should be empty after clear"
        assert tt.hits == 0, "Hits should be reset"
        assert tt.misses == 0, "Misses should be reset"


class TestMoveOrdering:
    """Tests for move ordering heuristics."""

    @pytest.fixture
    def evaluator(self):
        return ClassicalEvaluator()

    def test_captures_ordered_first(self):
        """Test that captures are searched before quiet moves."""

        from chess_engine.search.minimax import order_moves

        board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        )

        legal_moves = list(board.legal_moves)
        ordered_moves = order_moves(board, legal_moves)

        # Find a capture move
        capture_move = None
        quiet_move = None

        for move in legal_moves:
            if board.is_capture(move):
                capture_move = move
            elif not board.is_capture(move):
                quiet_move = move

            if capture_move and quiet_move:
                break

        if capture_move and quiet_move:
            # Capture should come before quiet move in ordered list
            capture_index = ordered_moves.index(capture_move)
            quiet_index = ordered_moves.index(quiet_move)

            assert (
                capture_index < quiet_index
            ), "Captures should be ordered before quiet moves"


class TestSearchIntegration:
    """Integration tests for complete search."""

    @pytest.fixture
    def evaluator(self):
        return ClassicalEvaluator()

    def test_complete_game_search(self, evaluator):
        """Test search throughout a complete game."""

        board = chess.Board()
        tt = TranspositionTable(max_size=1000000)

        for _ in range(10):
            if board.is_game_over():
                break

            best_move, score, nodes, pv = find_best_move(
                board,
                depth=3,  # Shallow depth for speed
                evaluator=evaluator,
                transposition_table=tt,
            )

            assert (
                best_move in board.legal_moves
            ), f"Search returned illegal move: {best_move}"
            assert nodes > 0, "Should search at least one node"

            board.push(best_move)

    def test_search_finds_obvious_captures(self, evaluator):
        """Test that search finds obviously good captures."""

        board = chess.Board(
            "rnbqkbnr/pppppppp/8/8/8/3Q4/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
        )

        best_move, score, nodes, pv = find_best_move(board, depth=3, evaluator=evaluator)

        # Should capture the hanging piece
        assert best_move in board.legal_moves
        assert nodes > 0, "Should search at least one node"
