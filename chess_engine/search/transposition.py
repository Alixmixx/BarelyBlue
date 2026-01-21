"""
Transposition Table with Zobrist Hashing

This module implements a transposition table (TT) - a hash table that caches
position evaluations to avoid re-searching the same positions. This is one of
the most important optimizations in chess engines.

References:
    - Zobrist Hashing: https://www.chessprogramming.org/Zobrist_Hashing
    - Transposition Table: https://www.chessprogramming.org/Transposition_Table
"""

import chess
import random
from typing import Optional, Dict, Tuple
from enum import Enum


class NodeType(Enum):
    """
    Type of node in search tree.

    This determines how we can use the cached value:
        - EXACT: The exact evaluation (all moves searched)
        - LOWER_BOUND: Alpha cutoff occurred (eval >= beta)
        - UPPER_BOUND: Beta cutoff occurred (eval <= alpha)
    """
    EXACT = 0
    LOWER_BOUND = 1  # Alpha cutoff (eval is at least this good)
    UPPER_BOUND = 2  # Beta cutoff (eval is at most this good)


class TTEntry:
    """
    Entry in the transposition table.

    Attributes:
        zobrist_hash: 64-bit hash of position
        depth: Search depthof this entry
        value: Evaluation score (centipawns)
        node_type: EXACT, LOWER_BOUND, or UPPER_BOUND
        best_move: Best move found in this position

    Size: ~32 bytes per entry
        - zobrist_hash: 8 bytes (64-bit int)
        - depth: 4 bytes (int)
        - value: 4 bytes (float32)
        - node_type: 1 byte (enum)
        - best_move: ~4 bytes (move encoding)
        - Padding: ~11 bytes

    Memory Usage Examples:
        - 1 million entries ≈ 32 MB
        - 10 million entries ≈ 320 MB
        - 100 million entries ≈ 3.2 GB
    """

    def __init__(
        self,
        zobrist_hash: int,
        depth: int,
        value: float,
        node_type: NodeType,
        best_move: Optional[chess.Move] = None,
    ):
        self.zobrist_hash = zobrist_hash
        self.depth = depth
        self.value = value
        self.node_type = node_type
        self.best_move = best_move

    def __repr__(self) -> str:
        return (
            f"TTEntry(hash={self.zobrist_hash}, depth={self.depth}, "
            f"value={self.value:.2f}, type={self.node_type}, move={self.best_move})"
        )


# ============================================================================
# Zobrist Hashing
# ============================================================================
# Zobrist hashing uses random 64-bit numbers for each piece/square combination.
# Hash = XOR of all piece positions.
# Incremental updates: Moving a piece = XOR old square, XOR new square.
#
# Hash components:
#   - 12 piece types (6 pieces * 2 colors) * 64 squares = 768 random numbers
#   - Castling rights (4 bits) = 4 random numbers
#   - En passant file (8 files) = 8 random numbers
#   - Side to move (1 bit) = 1 random number
#   Total: 781 random 64-bit numbers
# ============================================================================

# Initialize Zobrist random numbers
random.seed(42)  # Fixed seed for reproducibility

# Piece hashes: [piece_type][color][square]
# piece_type: 1-6 (PAWN to KING)
# color: 0-1 (WHITE, BLACK)
# square: 0-63
ZOBRIST_PIECES = [
    [[random.getrandbits(64) for _ in range(64)] for _ in range(2)]
    for _ in range(7)  # 0 is unused, 1-6 are piece types
]

# Castling rights hashes (4 bits: WK, WQ, BK, BQ)
ZOBRIST_CASTLING = [random.getrandbits(64) for _ in range(16)]

# En passant file hashes (8 files)
ZOBRIST_EN_PASSANT = [random.getrandbits(64) for _ in range(8)]

# Side to move hash (XOR this if black to move)
ZOBRIST_SIDE_TO_MOVE = random.getrandbits(64)


# TODO:
# Current implementation recomputes hash from scratch each time (slow).
# Incremental updates are much faster
def zobrist_hash(board: chess.Board) -> int:
    """
    Compute Zobrist hash for a chess position.
    The hash uniquely identifies a position (with low enough collisions).

    Args:
        board: python-chess Board object

    Returns:
        64-bit integer hash
    """
    hash_value = 0

    # Hash all pieces on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            hash_value ^= ZOBRIST_PIECES[piece.piece_type][piece.color][square]

    # Hash castling rights
    castling_index = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_index |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_index |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_index |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_index |= 8
    hash_value ^= ZOBRIST_CASTLING[castling_index]

    # Hash en passant square
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        hash_value ^= ZOBRIST_EN_PASSANT[ep_file]

    # Hash side to move
    if board.turn == chess.BLACK:
        hash_value ^= ZOBRIST_SIDE_TO_MOVE

    return hash_value


class TranspositionTable:
    """
    Transposition table for caching position evaluations.

    The TT stores previously evaluated positions to avoid re-searching different move order.

    Attributes:
        max_size: Maximum number of entries (memory limit)
        table: Dictionary mapping hash → TTEntry
    """

    def __init__(self, max_size: int = 10000000):
        """
        Initialize transposition table.

        Args:
            max_size: Maximum number of entries (default 10M ≈ 320MB)

        """
        self.max_size = max_size
        self.table: Dict[int, TTEntry] = {}
        self.hits = 0
        self.misses = 0
        self.collisions = 0

    def store(
        self,
        zobrist_hash: int,
        depth: int,
        value: float,
        node_type: NodeType,
        best_move: Optional[chess.Move] = None,
    ):
        """
        Store a position evaluation in the transposition table.

        Args:
            zobrist_hash: Zobrist hash of the position
            depth: Search depth this evaluation was performed at
            value: Evaluation score (centipawns)
            node_type: EXACT, LOWER_BOUND, or UPPER_BOUND
            best_move: Best move found (optional)
        """
        # Check if we should replace existing entry
        if zobrist_hash in self.table:
            existing = self.table[zobrist_hash]

            # detect collisions
            if existing.zobrist_hash != zobrist_hash:
                self.collisions += 1

            # Only replace if new depth >= old depth
            if depth < existing.depth:
                return

        # Store new entry
        entry = TTEntry(zobrist_hash, depth, value, node_type, best_move)
        self.table[zobrist_hash] = entry

        # TODO: Improve eviction
        if len(self.table) > self.max_size:
            random_key = random.choice(list(self.table.keys()))
            del self.table[random_key]

    def lookup(self, zobrist_hash: int, depth: int = 0) -> Optional[TTEntry]:
        """
        Look up a position in the transposition table.

        Args:
            zobrist_hash: Zobrist hash of the position
            depth: Current search depth (only use if cached depth >= this)

        Returns:
            TTEntry if found and usable, None otherwise
        """
        if zobrist_hash in self.table:
            entry = self.table[zobrist_hash]

            # Only use entry if it was searched to at least the current depth
            if entry.depth >= depth:
                self.hits += 1
                return entry

        self.misses += 1
        return None

    def clear(self):
        """Clear all entries from the transposition table."""

        self.table.clear()
        self.hits = 0
        self.misses = 0
        self.collisions = 0

    def get_stats(self) -> Dict[str, int | float]:
        """Get statistics about transposition table usage."""

        total_lookups = self.hits + self.misses
        hit_rate = (self.hits / total_lookups * 100) if total_lookups > 0 else 0

        return {
            'entries': len(self.table),
            'hits': self.hits,
            'misses': self.misses,
            'collisions': self.collisions,
            'hit_rate': hit_rate,
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"TranspositionTable(entries={stats['entries']}, "
            f"hit_rate={stats['hit_rate']:.1f}%)"
        )
