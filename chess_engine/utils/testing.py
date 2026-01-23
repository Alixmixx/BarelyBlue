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
    TestPosition(
        id="BK.04",
        fen="rnbqkb1r/p3pppp/1p6/2ppP3/3N4/2P5/PPP1QPPP/R1B1KB1R w KQkq - 0 1",
        best_moves=["e5e6"],
        description="White breaks through with e6"
    ),
    TestPosition(
        id="BK.05",
        fen="r1b2rk1/2q1b1pp/p2ppn2/1p6/3QP3/1BN1B3/PPP3PP/R4RK1 w - - 0 1",
        best_moves=["d4d7", "c3d5"],
        description="White wins material with Qd7 or Nd5"
    ),
    TestPosition(
        id="BK.06",
        fen="2r3k1/pppR1pp1/4p3/4P1P1/5P2/1P4K1/P1P5/8 w - - 0 1",
        best_moves=["g5g6"],
        description="White advances with g6"
    ),
    TestPosition(
        id="BK.07",
        fen="1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1 w - - 0 1",
        best_moves=["h5f6"],
        description="White plays Nf6"
    ),
    TestPosition(
        id="BK.08",
        fen="4b3/p3kp2/6p1/3pP2p/2pP1P2/4K1P1/P3N2P/8 w - - 0 1",
        best_moves=["f4f5"],
        description="White advances with f5"
    ),
    TestPosition(
        id="BK.09",
        fen="2kr1bnr/pbpq4/2n1pp2/3p3p/3P1P1B/2N2N1Q/PPP3PP/2KR1B1R w - - 0 1",
        best_moves=["f4f5"],
        description="White attacks with f5"
    ),
    TestPosition(
        id="BK.10",
        fen="3rr1k1/pp3pp1/1qn2np1/8/3p4/PP1R1P2/2P1NQPP/R1B3K1 b - - 0 1",
        best_moves=["c6e5"],
        description="Black plays Ne5"
    ),
    TestPosition(
        id="BK.11",
        fen="2r1nrk1/p2q1ppp/bp1p4/n1pPp3/P1P1P3/2PBB1N1/4QPPP/R4RK1 w - - 0 1",
        best_moves=["f2f4"],
        description="White advances with f4"
    ),
    TestPosition(
        id="BK.12",
        fen="r3r1k1/ppqb1ppp/8/4p1NQ/8/2P5/PP3PPP/R3R1K1 b - - 0 1",
        best_moves=["d7f5"],
        description="Black defends with Bf5"
    ),
    TestPosition(
        id="BK.13",
        fen="r2q1rk1/4bppp/p2p4/2pP4/3pP3/3Q4/PP1B1PPP/R3R1K1 w - - 0 1",
        best_moves=["b2b4"],
        description="White plays b4"
    ),
    TestPosition(
        id="BK.14",
        fen="rnb2r1k/pp2p2p/2pp2p1/q2P1p2/8/1Pb2NP1/PB2PPBP/R2Q1RK1 w - - 0 1",
        best_moves=["d5c6", "d5d6"],
        description="White advances with dxc6 or d6"
    ),
    TestPosition(
        id="BK.15",
        fen="2r3k1/1p2q1pp/2b1pr2/p1pp4/6Q1/1P1PP1R1/P1PN2PP/5RK1 w - - 0 1",
        best_moves=["g4g7"],
        description="White plays Qg7+"
    ),
    TestPosition(
        id="BK.16",
        fen="r1bqkb1r/4npp1/p1p4p/1p1pP1B1/3N1P2/2N5/PPP3PP/R2QK2R w KQkq - 0 1",
        best_moves=["e5e6"],
        description="White breaks with e6"
    ),
    TestPosition(
        id="BK.17",
        fen="r2q1rk1/1ppnbppp/p2p1nb1/3Pp3/2P1P1P1/2N2N1P/PPB1QP2/R1B2RK1 b - - 0 1",
        best_moves=["h7h5"],
        description="Black attacks with h5"
    ),
    TestPosition(
        id="BK.18",
        fen="r1bq1rk1/pp2ppbp/2np2p1/2n5/P3PP2/N1P2N2/1PB3PP/R1B1QRK1 b - - 0 1",
        best_moves=["c6b4"],
        description="Black plays Nb4"
    ),
    TestPosition(
        id="BK.19",
        fen="3rr3/2pq2pk/p2p1pnp/8/2QBPP2/1P6/P5PP/4RRK1 b - - 0 1",
        best_moves=["e8e4"],
        description="Black plays Rxe4"
    ),
    TestPosition(
        id="BK.20",
        fen="r4k2/pb2bp1r/1p1qp2p/3pNp2/3P1P2/2N3P1/PPP1Q2P/2KRR3 w - - 0 1",
        best_moves=["g3g4"],
        description="White attacks with g4"
    ),
    TestPosition(
        id="BK.21",
        fen="3rn2k/ppb2rpp/2ppqp2/5N2/2P1P3/1P5Q/PB3PPP/3RR1K1 w - - 0 1",
        best_moves=["h3h7"],
        description="White sacrifices with Qxh7+"
    ),
    TestPosition(
        id="BK.22",
        fen="2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 b - - 0 1",
        best_moves=["b7e4"],
        description="Black plays Bxe4"
    ),
    TestPosition(
        id="BK.23",
        fen="r1bqk2r/pp2bppp/2p5/3pP3/P2Q1P2/2N1B3/1PP3PP/R4RK1 b kq - 0 1",
        best_moves=["e8g8"],
        description="Black castles kingside"
    ),
    TestPosition(
        id="BK.24",
        fen="r2qnrnk/p2b2b1/1p1p2pp/2pPpp2/1PP1P3/PRNBB3/3QNPPP/5RK1 w - - 0 1",
        best_moves=["f2f4"],
        description="White advances with f4"
    ),
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
        best_move, score, nodes, pv = find_best_move(
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
            print(f"Nodes searched: {nodes:,}")
            print(f"Principal variation: {' '.join([m.uci() for m in pv[:5]])}")
            print(f"Time: {time_taken:.2f}s")
            print(f"Result: {'✓ CORRECT' if correct else '✗ WRONG'}")

        return TestResult(
            position=position,
            found_move=found_move_uci,
            score=score,
            correct=correct,
            time_taken=time_taken,
            nodes_searched=nodes,
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