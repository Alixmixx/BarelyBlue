"""
Board Representation Module

This module provides utilities for converting chess board states into tensor
representations suitable for neural network processing.

Key Components:
    - board_to_tensor: Converts python-chess Board to 12-channel tensor representation (12-8-8)
    - Coordinate mapping utilities for square indexing
    - Future: Metadata planes for game state (castling, en passant, etc.)

Data Flow:
    python-chess Board → board_to_tensor() → (12, 8, 8) numpy array → NN model
"""

from chess_engine.board.representation import board_to_tensor

__all__ = ['board_to_tensor']
