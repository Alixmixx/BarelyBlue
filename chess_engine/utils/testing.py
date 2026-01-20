"""
Chess Engine Testing and Benchmarking

This module provides test suites and benchmarking tools for evaluating
chess engine performance.

Test Suites:
    1. Bratko-Kopec Test: 24 tactical positions
       - Created by Danny Kopec and Ivan Bratko (1982)
       - Tests tactical vision and search effectiveness
       - Each position has a known best move
       - Standard benchmark for chess engines

    2. Win At Chess (WAC): 300 tactical positions
       - More extensive tactical test suite
       - Positions require finding forcing moves (checks, captures, threats)
       - Good for testing tactical strength

Evaluation Metrics:
    - Correct Moves: Number of positions where engine found best move
    - Time per Position: Average thinking time
    - Nodes Searched: Total nodes evaluated
    - Depth Reached: Average search depth

References:
    - Bratko-Kopec: https://www.chessprogramming.org/Bratko-Kopec_Test
    - WAC: https://www.chessprogramming.org/Win_at_Chess
"""

import chess
from typing import List, Any, Optional, Dict
from dataclasses import dataclass
from chess_engine.evaluation.base import Evaluator
from chess_engine.search.minimax import find_best_move
from chess_engine.search.transposition import TranspositionTable
import time


@dataclass
class TestPosition:
    """
    A test position with expected best move(s).

    Attributes:
        fen: Board position in FEN notation
        best_moves: List of acceptable best moves (UCI format)
        description: Human-readable description of the position
        id: Position identifier (e.g., "BK.01" for Bratko-Kopec #1)

    """
    fen: str
    best_moves: List[str]  # UCI move strings
    description: str = ""
    id: str = ""


@dataclass
class TestResult:
    """
    Result of testing a single position.

    Attributes:
        position: The test position
        found_move: Move the engine found (UCI format)
        score: Evaluation score for the move
        correct: Whether the engine found a best move
        time_taken: Time spent searching (seconds)
        nodes_searched: Number of nodes evaluated
        depth: Search depth used

    """
    position: TestPosition
    found_move: str
    score: float
    correct: bool
    time_taken: float
    nodes_searched: int = 0
    depth: int = 0


# ============================================================================
# Bratko-Kopec Test Suite
# ============================================================================

BRATKO_KOPEC_POSITIONS = [
    TestPosition(
        id="BK.01",
        fen="1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1",
        best_moves=["d6d1"],
        description="Black forces checkmate with Qd1+"
    ),
    TestPosition(
        id="BK.02",
        fen="3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - - 0 1",
        best_moves=["d4d5"],
        description="White breaks through with d5"
    ),
    TestPosition(
        id="BK.03",
        fen="2q1rr1k/3bbnnp/p2p1pp1/2pPp3/PpP1P1P1/1P2BNNP/2BQ1PRK/7R b - - 0 1",
        best_moves=["f6f5"],
        description="Black counterattacks with f5"
    ),
    # TODO: Add remaining 21 Bratko-Kopec positions
]


# ============================================================================
# Win At Chess (WAC) Test Suite
# ============================================================================

WAC_POSITIONS = [
    TestPosition(
        id="WAC.001",
        fen="2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
        best_moves=["g3g6"],
        description="White wins with Qg6"
    ),
    # TODO: Add more WAC positions
]


def evaluate_position(
    position: TestPosition,
    depth: int,
    evaluator: Evaluator,
    transposition_table: Optional[TranspositionTable] = None,
    verbose: bool = False,
) -> TestResult:
    """
    Evaluate a single test position.

    Args:
        position: Test position to evaluate
        depth: Search depth
        evaluator: Position evaluator
        transposition_table: Optional TT for caching
        verbose: If True, print detailed output

    Returns:
        TestResult with engine's move and whether it was correct
    """
    board = chess.Board(position.fen)

    if verbose:
        print(f"\nTesting {position.id}: {position.description}")
        print(f"FEN: {position.fen}")
        print(f"Expected moves: {position.best_moves}")

    start_time = time.time()

    try:
        best_move, score = find_best_move(
            board,
            depth,
            evaluator,
            transposition_table,
            verbose=False
        )

        time_taken = time.time() - start_time
        found_move_uci = best_move.uci()
        correct = found_move_uci in position.best_moves

        if verbose:
            print(f"Engine found: {found_move_uci} (score: {score:.2f})")
            print(f"Time: {time_taken:.2f}s")
            print(f"Result: {'✓ CORRECT' if correct else '✗ WRONG'}")

        return TestResult(
            position=position,
            found_move=found_move_uci,
            score=score,
            correct=correct,
            time_taken=time_taken,
            depth=depth,
        )

    except Exception as e:
        print(f"Error evaluating position {position.id}: {e}")
        return TestResult(
            position=position,
            found_move="",
            score=0.0,
            correct=False,
            time_taken=time.time() - start_time,
            depth=depth,
        )


def run_bratko_kopec(
    evaluator: Evaluator,
    depth: int = 5,
    transposition_table: Optional[TranspositionTable] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the Bratko-Kopec test suite.

    Args:
        evaluator: Position evaluator
        depth: Search depth (default: 5)
        transposition_table: Optional TT for caching
        verbose: If True, print detailed results

    Returns:
        Dictionary with test results:
            - score: Number of correct positions
            - total: Total number of positions
            - percentage: Success percentage
            - results: List of TestResult objects
            - avg_time: Average time per position
    """
    if verbose:
        print("=" * 70)
        print("BRATKO-KOPEC TEST SUITE")
        print("=" * 70)

    results = []
    correct_count = 0
    total_time = 0.0

    for position in BRATKO_KOPEC_POSITIONS:
        result = evaluate_position(
            position,
            depth,
            evaluator,
            transposition_table,
            verbose=verbose
        )
        results.append(result)

        if result.correct:
            correct_count += 1

        total_time += result.time_taken

    avg_time = total_time / len(BRATKO_KOPEC_POSITIONS) if BRATKO_KOPEC_POSITIONS else 0
    percentage = (correct_count / len(BRATKO_KOPEC_POSITIONS) * 100) if BRATKO_KOPEC_POSITIONS else 0

    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Score: {correct_count}/{len(BRATKO_KOPEC_POSITIONS)} ({percentage:.1f}%)")
        print(f"Average time: {avg_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")

    return {
        'score': correct_count,
        'total': len(BRATKO_KOPEC_POSITIONS),
        'percentage': percentage,
        'results': results,
        'avg_time': avg_time,
        'total_time': total_time,
    }


def run_wac(
    evaluator: Evaluator,
    depth: int = 5,
    transposition_table: Optional[TranspositionTable] = None,
    max_positions: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the Win At Chess (WAC) test suite.

    Args:
        evaluator: Position evaluator
        depth: Search depth (default: 5)
        transposition_table: Optional TT for caching
        max_positions: Limit number of positions to test (None = all)
        verbose: If True, print detailed results

    Returns:
        Dictionary with test results (same format as run_bratko_kopec)
    """
    if verbose:
        print("=" * 70)
        print("WIN AT CHESS (WAC) TEST SUITE")
        print("=" * 70)

    positions = WAC_POSITIONS
    if max_positions:
        positions = positions[:max_positions]

    results = []
    correct_count = 0
    total_time = 0.0

    for position in positions:
        result = evaluate_position(
            position,
            depth,
            evaluator,
            transposition_table,
            verbose=verbose
        )
        results.append(result)

        if result.correct:
            correct_count += 1

        total_time += result.time_taken

    avg_time = total_time / len(positions) if positions else 0
    percentage = (correct_count / len(positions) * 100) if positions else 0

    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Score: {correct_count}/{len(positions)} ({percentage:.1f}%)")
        print(f"Average time: {avg_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")

    return {
        'score': correct_count,
        'total': len(positions),
        'percentage': percentage,
        'results': results,
        'avg_time': avg_time,
        'total_time': total_time,
    }