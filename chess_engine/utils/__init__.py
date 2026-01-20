"""
Utilities Module

This module provides utility functions for testing and benchmarking the
chess engine.

Key Components:
    - Bratko-Kopec test suite: 24 tactical positions
    - Win At Chess (WAC) test suite: 300 tactical positions
    - Perft: Move generation verification

Testing Methodology:
    Test suites like Bratko-Kopec are standard benchmarks for chess engines.
    They contain positions where the best move is known, and the engine's
    task is to find it within a reasonable depth/time.

Success Metrics:
    - Bratko-Kopec: 8/24 at depth 5 (reasonable classical engine)
    - WAC: 150/300 at depth 5 (good tactical strength)
"""

from chess_engine.utils.testing import (
    run_bratko_kopec,
    run_wac,
    evaluate_position,
)

__all__ = [
    'run_bratko_kopec',
    'run_wac',
    'evaluate_position',
]
