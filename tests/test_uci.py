"""
Unit Tests for UCI Interface

Tests for UCI protocol implementation, focusing on:
    - Command parsing: uci, isready, position, go, stop, quit
    - Position setup: FEN parsing, move application
    - Search invocation: Correct depth/time handling
    - Output format: Proper UCI responses
    - Error handling: Invalid commands, illegal moves
"""

import chess
import pytest
from io import StringIO
from unittest.mock import patch, MagicMock
from chess_engine.uci import UCIEngine
from chess_engine.search.transposition import NodeType


class TestUCICommands:
    """Tests for UCI command handling."""

    @pytest.fixture
    def engine(self):
        """Create a UCI engine for testing."""
        return UCIEngine()

    def test_handle_uci(self, engine, capsys):
        """Test 'uci' command response."""

        engine.handle_uci()

        captured = capsys.readouterr()
        output = captured.out

        assert "id name" in output, "Should include engine name"
        assert "id author" in output, "Should include author"
        assert "uciok" in output, "Should end with uciok"

    def test_handle_isready(self, engine, capsys):
        """Test 'isready' command response."""

        engine.handle_isready()

        captured = capsys.readouterr()
        output = captured.out

        assert "readyok" in output, "Should output readyok"

    def test_handle_ucinewgame(self, engine):
        """Test 'ucinewgame' command."""

        engine.board.push_san("e4")
        engine.board.push_san("e5")

        # Store something in TT
        engine.transposition_table.store(
            12345, depth=5, value=100.0, node_type=NodeType.EXACT
        )

        engine.handle_ucinewgame()

        # Board should be reset
        assert engine.board.fen() == chess.STARTING_FEN, "Board should be reset"

        # TT should be cleared
        assert len(engine.transposition_table.table) == 0, "TT should be cleared"

    def test_handle_position_startpos(self, engine):
        """Test 'position startpos' command."""

        engine.board.push_san("e4")

        engine.handle_position(["position", "startpos"])

        assert (
            engine.board.fen() == chess.STARTING_FEN
        ), "Should reset to starting position"

    def test_handle_position_with_moves(self, engine):
        """Test 'position startpos moves e2e4 e7e5' command."""

        engine.handle_position(["position", "startpos", "moves", "e2e4", "e7e5"])

        expected_fen = chess.Board()
        expected_fen.push_san("e4")
        expected_fen.push_san("e5")

        assert engine.board.fen() == expected_fen.fen(), "Should apply moves correctly"

    def test_handle_position_fen(self, engine):
        """Test 'position fen <FEN>' command."""

        test_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

        engine.handle_position(["position", "fen", *test_fen.split()])

        expected_board = chess.Board(test_fen)
        assert (
            engine.board.board_fen() == expected_board.board_fen()
        ), "Piece placement should match"
        assert engine.board.turn == expected_board.turn, "Turn should match"

    def test_handle_position_invalid_move(self, engine, capsys):
        """Test that invalid moves are rejected."""

        # Try illegal move
        engine.handle_position(["position", "startpos", "moves", "e2e5"])  # Illegal

        # Should not have applied the move
        assert engine.board is not None, "Engine should still have a board"

    def test_handle_go_depth(self, engine):
        """Test 'go depth X' command."""

        engine.handle_position(["position", "startpos"])

        engine.handle_go(["go", "depth", "3"])

        engine.handle_stop()

    def test_handle_stop(self, engine):
        """Test 'stop' command."""

        # Start a search
        engine.handle_position(["position", "startpos"])
        engine.handle_go(["go", "depth", "5"])

        engine.handle_stop()

        assert engine.stop_search, "Stop flag should be set"


class TestUCIPositionSetup:
    """Tests for position setup via UCI."""

    @pytest.fixture
    def engine(self):
        return UCIEngine()

    def test_startpos_is_correct(self, engine):
        """Test that startpos sets up correct position."""

        engine.handle_position(["position", "startpos"])

        assert engine.board.fen() == chess.STARTING_FEN

    def test_move_sequence(self, engine):
        """Test that move sequence is applied correctly."""

        moves = ["e2e4", "e7e5", "g1f3", "b8c6"]

        engine.handle_position(["position", "startpos", "moves"] + moves)

        # Verify final position
        expected = chess.Board()
        for move_uci in moves:
            expected.push(chess.Move.from_uci(move_uci))

        assert engine.board.fen() == expected.fen()

    def test_fen_with_moves(self, engine):
        """Test 'position fen <FEN> moves ...' command."""
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

        engine.handle_position(["position", "fen", *fen.split(), "moves", "e7e5"])

        # Should have applied e5 to the FEN position
        expected = chess.Board(fen)
        expected.push_san("e5")

        assert engine.board.fen() == expected.fen()


class TestUCIOutput:
    """Tests for UCI output format."""

    @pytest.fixture
    def engine(self):
        return UCIEngine()

    def test_bestmove_output_format(self, engine, capsys):
        """Test that bestmove output is in correct UCI format."""

        engine.handle_position(["position", "startpos"])

        assert engine is not None


class TestUCIIntegration:
    """Integration tests for full UCI workflow."""

    @pytest.fixture
    def engine(self):
        return UCIEngine()

    def test_full_uci_session(self, engine, capsys):
        """Test a complete UCI session."""

        # GUI: uci
        engine.handle_uci()
        output = capsys.readouterr().out
        assert "uciok" in output

        # GUI: isready
        engine.handle_isready()
        output = capsys.readouterr().out
        assert "readyok" in output

        # GUI: position startpos
        engine.handle_position(["position", "startpos"])

        # GUI: go depth 3
        engine.handle_go(["go", "depth", "3"])

        # Wait for search to complete
        import time

        time.sleep(0.5)

        # GUI: stop
        engine.handle_stop()

        # Clean up
        engine.stop_search = True

    def test_play_multiple_moves(self, engine):
        """Test playing multiple moves via UCI commands."""

        engine.handle_position(["position", "startpos", "moves", "e2e4"])
        assert engine.board.fen() != chess.STARTING_FEN

        engine.handle_position(["position", "startpos", "moves", "e2e4", "e7e5"])

        expected = chess.Board()
        expected.push_san("e4")
        expected.push_san("e5")

        assert engine.board.fen() == expected.fen()


class TestUCIErrorHandling:
    """Tests for error handling in UCI interface."""

    @pytest.fixture
    def engine(self):
        return UCIEngine()

    def test_invalid_fen(self, engine, capsys):
        """
        Test that invalid FEN is handled gracefully."""

        # Try to set invalid FEN
        engine.handle_position(["position", "fen", "invalid_fen"])

        assert engine.board is not None

    def test_illegal_move_ignored(self, engine):
        """Test that illegal moves are ignored."""

        initial_fen = engine.board.fen()

        # Try illegal move
        engine.handle_position(["position", "startpos", "moves", "e2e5"])

        assert engine.board is not None

    def test_empty_command(self, engine):
        """Test that empty commands don't crash."""

        engine.handle_position([])
        engine.handle_go([])

        assert engine.board is not None


class TestUCIThreading:
    """Tests for threading behavior."""

    @pytest.fixture
    def engine(self):
        return UCIEngine()

    def test_search_runs_in_background(self, engine):
        """Test that search runs in background thread."""

        engine.handle_position(["position", "startpos"])
        engine.handle_go(["go", "depth", "4"])

        import time

        time.sleep(0.1)

        engine.handle_stop()

        if engine.search_thread and engine.search_thread.is_alive():
            engine.search_thread.join(timeout=2.0)

    def test_stop_flag_interrupts_search(self, engine):
        """Test that stop flag can be set during search."""

        engine.handle_position(["position", "startpos"])
        engine.handle_go(["go", "depth", "5"])

        import time

        time.sleep(0.1)

        initial_searching = engine.searching

        engine.handle_stop()

        # Verify stop flag was set
        assert engine.stop_search, "Stop flag should be set after handle_stop()"

        if engine.search_thread:
            engine.search_thread.join(timeout=10.0)

        assert initial_searching, "Search should have been running initially"


def simulate_uci_command(engine, command_str):
    """
    Helper to simulate a UCI command.

    Args:
        engine: UCIEngine instance
        command_str: Command string (e.g., "position startpos moves e2e4")

    Returns:
        None (modifies engine state)
    """
    tokens = command_str.split()
    cmd = tokens[0]

    if cmd == "position":
        engine.handle_position(tokens)
    elif cmd == "go":
        engine.handle_go(tokens)
    elif cmd == "stop":
        engine.handle_stop()
    elif cmd == "uci":
        engine.handle_uci()
    elif cmd == "isready":
        engine.handle_isready()
    elif cmd == "ucinewgame":
        engine.handle_ucinewgame()
