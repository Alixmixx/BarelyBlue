"""
Tests for 18-channel tensor representation with metadata planes.

Tests the extended tensor format that includes castling rights,
en passant target, and side to move information.
"""

import pytest
import chess
import numpy as np

from chess_engine.board.representation import (
    board_to_tensor,
    board_to_tensor_18,
    add_metadata_planes,
)


class TestTensor18Channels:
    """Test suite for 18-channel tensor representation."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        board = chess.Board()
        tensor = board_to_tensor_18(board)

        assert tensor.shape == (18, 8, 8)
        assert tensor.dtype == np.float32

    def test_first_12_channels_match(self):
        """Test that first 12 channels match board_to_tensor output."""
        board = chess.Board()
        tensor_12 = board_to_tensor(board)
        tensor_18 = board_to_tensor_18(board)

        # First 12 channels should be identical
        np.testing.assert_array_equal(tensor_18[:12], tensor_12)

    def test_starting_position_castling(self):
        """Test castling rights in starting position."""
        board = chess.Board()
        tensor = board_to_tensor_18(board)

        # All castling rights available
        assert np.all(tensor[12] == 1.0)  # White kingside
        assert np.all(tensor[13] == 1.0)  # White queenside
        assert np.all(tensor[14] == 1.0)  # Black kingside
        assert np.all(tensor[15] == 1.0)  # Black queenside

    def test_no_castling_rights(self):
        """Test position with no castling rights."""
        # Position after kings have moved
        fen = "rnbq1bnr/ppppkppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR w - - 2 3"
        board = chess.Board(fen)
        tensor = board_to_tensor_18(board)

        # No castling rights
        assert np.all(tensor[12] == 0.0)  # White kingside
        assert np.all(tensor[13] == 0.0)  # White queenside
        assert np.all(tensor[14] == 0.0)  # Black kingside
        assert np.all(tensor[15] == 0.0)  # Black queenside

    def test_partial_castling_rights(self):
        """Test position with partial castling rights."""
        # White can only castle kingside, Black can only castle queenside
        fen = "r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w Kq - 0 1"
        board = chess.Board(fen)
        tensor = board_to_tensor_18(board)

        assert np.all(tensor[12] == 1.0)  # White kingside: yes
        assert np.all(tensor[13] == 0.0)  # White queenside: no
        assert np.all(tensor[14] == 0.0)  # Black kingside: no
        assert np.all(tensor[15] == 1.0)  # Black queenside: yes

    def test_en_passant_plane(self):
        """Test en passant target square encoding."""
        # Position with en passant on e3
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        board = chess.Board(fen)
        tensor = board_to_tensor_18(board)

        # En passant plane (channel 16) should have 1 at e3
        ep_plane = tensor[16]

        # e3 is column 4 (e-file), row 5 (rank 3)
        assert ep_plane[5, 4] == 1.0

        # All other squares should be 0
        ep_plane[5, 4] = 0.0
        assert np.all(ep_plane == 0.0)

    def test_no_en_passant(self):
        """Test position without en passant."""
        board = chess.Board()
        tensor = board_to_tensor_18(board)

        # En passant plane should be all zeros
        assert np.all(tensor[16] == 0.0)

    def test_side_to_move_white(self):
        """Test side to move plane for White."""
        board = chess.Board()
        assert board.turn == chess.WHITE

        tensor = board_to_tensor_18(board)

        # Side to move plane should be all 1s for White
        assert np.all(tensor[17] == 1.0)

    def test_side_to_move_black(self):
        """Test side to move plane for Black."""
        board = chess.Board()
        board.push_san("e4")
        assert board.turn == chess.BLACK

        tensor = board_to_tensor_18(board)

        # Side to move plane should be all 0s for Black
        assert np.all(tensor[17] == 0.0)

    def test_complex_position(self):
        """Test a complex mid-game position."""
        # Italian Game position
        fen = "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        board = chess.Board(fen)
        tensor = board_to_tensor_18(board)

        # Check shape
        assert tensor.shape == (18, 8, 8)

        # Check first 12 channels are valid
        tensor_12 = board_to_tensor(board)
        np.testing.assert_array_equal(tensor[:12], tensor_12)

        # All castling rights available
        assert np.all(tensor[12] == 1.0)
        assert np.all(tensor[13] == 1.0)
        assert np.all(tensor[14] == 1.0)
        assert np.all(tensor[15] == 1.0)

        # No en passant
        assert np.all(tensor[16] == 0.0)

        # White to move
        assert np.all(tensor[17] == 1.0)

    def test_multiple_en_passant_positions(self):
        """Test en passant on different squares."""
        test_cases = [
            ("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", chess.D6, 2, 3),
            ("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR w KQkq e3 0 2", chess.E3, 5, 4),
            ("rnbqkbnr/pppppppp/8/8/Pp6/8/1PPPPPPP/RNBQKBNR w KQkq b3 0 2", chess.B3, 5, 1),
        ]

        for fen, ep_square, expected_row, expected_col in test_cases:
            board = chess.Board(fen)
            assert board.ep_square == ep_square

            tensor = board_to_tensor_18(board)

            # Check en passant plane
            ep_plane = tensor[16]
            assert ep_plane[expected_row, expected_col] == 1.0

            # All other squares should be 0
            ep_plane[expected_row, expected_col] = 0.0
            assert np.all(ep_plane == 0.0)


class TestAddMetadataPlanes:
    """Test the add_metadata_planes function."""

    def test_add_metadata_planes_matches_tensor_18(self):
        """Test that add_metadata_planes produces same output as board_to_tensor_18."""
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")

        tensor_12 = board_to_tensor(board)
        tensor_extended = add_metadata_planes(tensor_12, board)
        tensor_18 = board_to_tensor_18(board)

        # Should be identical
        np.testing.assert_array_equal(tensor_extended, tensor_18)

    def test_add_metadata_planes_shape(self):
        """Test that add_metadata_planes returns correct shape."""
        board = chess.Board()
        tensor_12 = board_to_tensor(board)
        tensor_extended = add_metadata_planes(tensor_12, board)

        assert tensor_extended.shape == (18, 8, 8)

    def test_add_metadata_preserves_piece_planes(self):
        """Test that piece planes are preserved."""
        board = chess.Board()
        tensor_12 = board_to_tensor(board)
        tensor_extended = add_metadata_planes(tensor_12, board)

        # First 12 channels should match original
        np.testing.assert_array_equal(tensor_extended[:12], tensor_12)


class TestTensor18Integration:
    """Integration tests for 18-channel tensors."""

    def test_different_positions_produce_different_metadata(self):
        """Test that different positions produce different metadata planes."""
        # Position 1: Starting position
        board1 = chess.Board()
        tensor1 = board_to_tensor_18(board1)

        # Position 2: After e4
        board2 = chess.Board()
        board2.push_san("e4")
        tensor2 = board_to_tensor_18(board2)

        # Metadata should differ (at least side to move)
        assert not np.array_equal(tensor1[17], tensor2[17])

    def test_game_progression(self):
        """Test tensor representation through a game."""
        board = chess.Board()

        # Starting position: White to move, all castling rights
        tensor = board_to_tensor_18(board)
        assert np.all(tensor[17] == 1.0)  # White to move
        assert np.all(tensor[12:16] == 1.0)  # All castling rights

        # After e4: Black to move
        board.push_san("e4")
        tensor = board_to_tensor_18(board)
        assert np.all(tensor[17] == 0.0)  # Black to move

        # After e4 e5: White to move
        board.push_san("e5")
        tensor = board_to_tensor_18(board)
        assert np.all(tensor[17] == 1.0)  # White to move

        # Castle kingside
        board.push_san("Nf3")
        board.push_san("Nf6")
        board.push_san("Be2")
        board.push_san("Be7")
        board.push_san("O-O")

        tensor = board_to_tensor_18(board)
        # White has castled, no longer has castling rights
        assert np.all(tensor[12] == 0.0)  # White kingside lost
        assert np.all(tensor[13] == 0.0)  # White queenside lost
