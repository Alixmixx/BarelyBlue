"""
Search Module

This module implements chess search algorithms. The primary algorithm is
minimax with alpha-beta pruning, enhanced with a transposition table for
caching previously evaluated positions.

Key Components:
    - minimax: Core search algorithm with alpha-beta pruning
    - find_best_move: Root-level search function
    - TranspositionTable: Zobrist hashing and position cache
    - Move ordering: Heuristics to improve alpha-beta efficiency

"""

from chess_engine.search.minimax import minimax, find_best_move
from chess_engine.search.transposition import TranspositionTable, zobrist_hash

__all__ = ['minimax', 'find_best_move', 'TranspositionTable', 'zobrist_hash']
