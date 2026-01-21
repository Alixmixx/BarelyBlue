"""
Classical Piece-Square Table Evaluation

This module implements a traditional chess evaluation function using:
    1. Material counting (piece values)
    2. Piece-Square Tables (positional bonuses/penalties)

This is the baseline evaluator. It should achieve reasonable
tactical play (solve ~8/24 Bratko-Kopec tests at depth 5).

Evaluation Components:
    - Material: P=100, N=320, B=330, R=500, Q=900, K=0
    - Position: PST bonuses for each piece type

Reference:
    Simplified Evaluation Function
    https://www.chessprogramming.org/Simplified_Evaluation_Function
"""

import chess
import numpy as np
from chess_engine.evaluation.base import Evaluator

#fmt: off
# ============================================================================
# Material Values (centipawns)
# ============================================================================
# These are standard values used in most chess engines

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


# ============================================================================
# Piece-Square Tables (PSTs)
# ============================================================================
# These tables assign bonuses/penalties based on piece placement.
# Values are from White's perspective (row 0 = rank 8, row 7 = rank 1).
# For Black pieces, we'll flip the table vertically.
#
# Convention: Higher values = better squares
# Units: Centipawns (added to material value)
#
# ============================================================================

# Pawn PST: Encourage central pawns, discourage edge pawns
# Bonuses for advanced pawns
PAWN_TABLE = np.array([
    [  0,   0,   0,   0,   0,   0,   0,   0],  # Rank 8 (promotion)
    [ 50,  50,  50,  50,  50,  50,  50,  50],  # Rank 7
    [ 10,  10,  20,  30,  30,  20,  10,  10],  # Rank 6
    [  5,   5,  10,  25,  25,  10,   5,   5],  # Rank 5
    [  0,   0,   0,  20,  20,   0,   0,   0],  # Rank 4
    [  5,  -5, -10,   0,   0, -10,  -5,   5],  # Rank 3
    [  5,  10,  10, -20, -20,  10,  10,   5],  # Rank 2
    [  0,   0,   0,   0,   0,   0,   0,   0],  # Rank 1
], dtype=np.float32)

# Knight PST: "Knights on the rim are dim"
# Central knights are powerful, edge knights are weak
KNIGHT_TABLE = np.array([
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   0,   0,   0,   0, -20, -40],
    [-30,   0,  10,  15,  15,  10,   0, -30],
    [-30,   5,  15,  20,  20,  15,   5, -30],
    [-30,   0,  15,  20,  20,  15,   0, -30],
    [-30,   5,  10,  15,  15,  10,   5, -30],
    [-40, -20,   0,   5,   5,   0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
], dtype=np.float32)

# Bishop PST: Prefer central positions, avoid being trapped
# Fianchettoed bishops (b2/g2 for white) get bonuses
BISHOP_TABLE = np.array([
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
], dtype=np.float32)

# Rook PST: Prefer 7th rank, open files
# Rooks should be developed late
ROOK_TABLE = np.array([
    [  0,   0,   0,   0,   0,   0,   0,   0],
    [  5,  10,  10,  10,  10,  10,  10,   5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [ -5,   0,   0,   0,   0,   0,   0,  -5],
    [  0,   0,   0,   5,   5,   0,   0,   0],
], dtype=np.float32)

# Queen PST: Avoid early development, prefer central control
QUEEN_TABLE = np.array([
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [  0,   0,   5,   5,   5,   5,   0,  -5],
    [-10,   5,   5,   5,   5,   5,   0, -10],
    [-10,   0,   5,   0,   0,   0,   0, -10],
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
], dtype=np.float32)

# King PST (Middlegame): Stay safe, prefer castled position
# Strong penalty for exposed king
KING_MIDDLEGAME_TABLE = np.array([
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [ 20,  30,  10,   0,   0,  10,  30,  20],
], dtype=np.float32)

# King PST (Endgame): Centralize king, help with pawn promotion
# In endgame, king becomes an active piece
KING_ENDGAME_TABLE = np.array([
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10,   0,   0, -10, -20, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -30,   0,   0,   0,   0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50],
], dtype=np.float32)
#fmt: on


class ClassicalEvaluator(Evaluator):
    """
    Classical evaluation using material and piece-square tables.

    This evaluator combines:
        1. Material counting (sum of piece values)
        2. Positional evaluation (piece-square table bonuses)
        3. Simple endgame detection (different king table)

    Attributes:
        piece_tables: Dictionary mapping piece types to PST arrays
    """

    def __init__(self):
        """Initialize the classical evaluator with piece-square tables."""
        self.piece_tables = {
            chess.PAWN: PAWN_TABLE,
            chess.KNIGHT: KNIGHT_TABLE,
            chess.BISHOP: BISHOP_TABLE,
            chess.ROOK: ROOK_TABLE,
            chess.QUEEN: QUEEN_TABLE,
        }

        # Endgame if total material < queen + rook (1400 centipawns)
        self.endgame_threshold = 1400

    def is_endgame(self, board: chess.Board) -> bool:
        """
        Detect if position is in endgame phase.

        Simple heuristic: Endgame if both sides have limited material
        (less than rook + queen = 1400 centipawns of non-pawn pieces)

        Args:
            board: Chess board to analyze

        Returns:
            bool: True if endgame, False otherwise
        """
        # Count non-pawn material for both sides
        material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.PAWN:
                material += PIECE_VALUES.get(piece.piece_type, 0)

        return material < self.endgame_threshold

    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate position using material + PST.

        Args:
            board: Chess board to evaluate

        Returns:
            float: Evaluation in centipawns (White's perspective)
        """
        # Check for terminal positions first
        terminal_score = self.evaluate_terminal(board)
        if terminal_score is not None:
            return terminal_score

        # Determine game phase for king PST
        endgame = self.is_endgame(board)
        king_table = KING_ENDGAME_TABLE if endgame else KING_MIDDLEGAME_TABLE

        score = 0.0

        # Iterate through all squares
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            # Get material value
            piece_value = PIECE_VALUES[piece.piece_type]

            # Get positional value from PST
            # Convert square to (row, col)
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            row = 7 - rank
            col = file

            # Get PST value
            if piece.piece_type == chess.KING:
                pst_value = king_table[row, col]
            else:
                pst_table = self.piece_tables.get(piece.piece_type)
                if pst_table is not None:
                    pst_value = pst_table[row, col]
                else:
                    pst_value = 0

            # For black pieces, flip the table vertically
            if piece.color == chess.BLACK:
                flipped_row = 7 - row
                if piece.piece_type == chess.KING:
                    pst_value = king_table[flipped_row, col]
                else:
                    pst_table = self.piece_tables.get(piece.piece_type)
                    if pst_table is not None:
                        pst_value = pst_table[flipped_row, col]
                    else:
                        pst_value = 0

            total_value = piece_value + pst_value

            if piece.color == chess.WHITE:
                score += total_value
            else:
                score -= total_value

        return score
