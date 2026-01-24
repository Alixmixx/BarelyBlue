"""
Board Representation for Neural Network Input

This module converts python-chess Board objects into tensor representations
that can be fed into neural networks.

12-Channel Representation (piece positions only):
    0: White Pawns      6: Black Pawns
    1: White Knights    7: Black Knights
    2: White Bishops    8: Black Bishops
    3: White Rooks      9: Black Rooks
    4: White Queens    10: Black Queens
    5: White Kings     11: Black Kings

18-Channel Representation (pieces + metadata):
    0-11: Same as above (piece positions)
    12: White kingside castling rights
    13: White queenside castling rights
    14: Black kingside castling rights
    15: Black queenside castling rights
    16: En passant target square (1 at target square, 0 elsewhere)
    17: Side to move (all 1s if White, all 0s if Black)

Each channel is an 8*8 binary mask where 1 indicates piece presence.

Board Orientation:
    - Row 0 = Rank 8 (Black's back rank)
    - Row 7 = Rank 1 (White's back rank)
    - Column 0 = A-file
    - Column 7 = H-file
"""

import chess
import numpy as np
from typing import Tuple

# Piece type to channel index mapping
# White pieces: channels 0-5
# Black pieces: channels 6-11
PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def square_to_coordinates(square: int) -> Tuple[int, int]:
    """
    Convert python-chess square index to (row, column) coordinates.

    Args:
        square: Square index (0-63) where 0=A1, 63=H8

    Returns:
        Tuple of (row, col) where:
            - row 0 = rank 8 (index 56-63)
            - row 7 = rank 1 (index 0-7)
            - col 0 = A-file
            - col 7 = H-file
    """
    rank = square // 8
    file = square % 8
    row = 7 - rank
    col = file
    return row, col


def coordinates_to_square(row: int, col: int) -> int:
    """
    Convert (row, column) coordinates to python-chess square index.

    Args:
        row: Row index (0-7) where 0 is rank 8
        col: Column index (0-7) where 0 is A-file

    Returns:
        Square index (0-63)
    """
    rank = 7 - row
    file = col
    return rank * 8 + file


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert a chess board to a 12-channel tensor representation.

    Args:
        board: python-chess Board object

    Returns:
        numpy array of shape (12, 8, 8) with dtype float32
        - 12 channels: 6 piece types * 2 colors
        - 8*8 dimensions
        - Binary values: 1.0 piece exists, 0.0 no piece
    """
    # Initialize 12-channel tensor with zeros
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    # Iterate through all squares on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Get channel index for this piece
            channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]

            # Convert square to (row, col)
            row, col = square_to_coordinates(square)

            tensor[channel, row, col] = 1.0

    return tensor


def tensor_to_board(tensor: np.ndarray) -> chess.Board:
    """
    Convert a 12-channel tensor back to a python-chess Board object.

    This is the inverse of board_to_tensor().

    Args:
        tensor: numpy array of shape (12, 8, 8)

    Returns:
        python-chess Board object

    Raises:
        ValueError: If tensor has invalid shape or multiple pieces on one square
    """
    if tensor.shape != (12, 8, 8):
        raise ValueError(f"Invalid tensor shape: {tensor.shape}. Expected (12, 8, 8)")

    # Create empty board
    board = chess.Board(fen=None)

    # Reverse mapping
    channel_to_piece = {v: k for k, v in PIECE_TO_CHANNEL.items()}

    # Iterate through all channels
    for channel in range(12):
        piece_type, color = channel_to_piece[channel]

        positions = np.argwhere(tensor[channel] > 0.5)

        for row, col in positions:
            square = coordinates_to_square(row, col)
            piece = chess.Piece(piece_type, color)

            # Check if square already occupied
            if board.piece_at(square) is not None:
                raise ValueError(
                    f"Multiple pieces on square {chess.square_name(square)}"
                )

            board.set_piece_at(square, piece)

    return board


def board_to_tensor_18(board: chess.Board) -> np.ndarray:
    """
    Convert board to 18-channel tensor with metadata planes.

    Channels 0-11: Piece positions (from board_to_tensor)
    Channel 12: White kingside castling (all 1s if available)
    Channel 13: White queenside castling (all 1s if available)
    Channel 14: Black kingside castling (all 1s if available)
    Channel 15: Black queenside castling (all 1s if available)
    Channel 16: En passant target square (1 at target square)
    Channel 17: Side to move (all 1s for White, all 0s for Black)

    Args:
        board: python-chess Board object

    Returns:
        numpy array of shape (18, 8, 8) with dtype float32
    """
    # Get base 12-channel representation
    tensor_12 = board_to_tensor(board)

    # Create 6 metadata planes
    metadata = np.zeros((6, 8, 8), dtype=np.float32)

    # Castling rights (all squares 1.0 if right available)
    if board.has_kingside_castling_rights(chess.WHITE):
        metadata[0, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.WHITE):
        metadata[1, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.BLACK):
        metadata[2, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.BLACK):
        metadata[3, :, :] = 1.0

    # En passant target square
    if board.ep_square is not None:
        row, col = square_to_coordinates(board.ep_square)
        metadata[4, row, col] = 1.0

    # Side to move (all 1s for White, all 0s for Black)
    if board.turn == chess.WHITE:
        metadata[5, :, :] = 1.0

    return np.concatenate([tensor_12, metadata], axis=0)


def add_metadata_planes(tensor: np.ndarray, board: chess.Board) -> np.ndarray:
    """
    Add metadata planes to tensor representation.
    
    Args:
        tensor: Unused (kept for API compatibility)
        board: Chess board to convert

    Returns:
        Extended tensor with shape (18, 8, 8)
    """
    return board_to_tensor_18(board)
