#!/usr/bin/env python3
"""
Bratko-Kopec Benchmark Runner

Runs the Bratko-Kopec test suite at multiple depths to establish
baseline performance metrics for the chess engine.

Usage:
    python tools/run_benchmark.py [--depths 3,4,5] [--verbose]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.evaluation.classical import ClassicalEvaluator
from chess_engine.search.transposition import TranspositionTable
from chess_engine.utils.testing import run_bratko_kopec
import time


def format_time(seconds: float) -> str:
    """Format time"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


def run_benchmark(depths: list[int], verbose: bool = False):
    """
    Run Bratko-Kopec benchmark at multiple depths.

    Args:
        depths: List of depths to test
        verbose: If True, print detailed results for each position
    """
    evaluator = ClassicalEvaluator()

    print("=" * 80)
    print("BRATKO-KOPEC BENCHMARK - BarelyBlue Chess Engine")
    print("=" * 80)
    print(f"Evaluator: Classical (Piece-Square Tables)")
    print(f"Search: Minimax with Alpha-Beta Pruning + Transposition Table")
    print(f"Depths: {depths}")
    print("=" * 80)
    print()

    all_results = []

    for depth in depths:
        print(f"\n{'=' * 80}")
        print(f"DEPTH {depth}")
        print("=" * 80)

        # Create a transposition table for each depth
        tt = TranspositionTable(max_size=10_000_000)

        start_time = time.time()
        result = run_bratko_kopec(
            evaluator=evaluator,
            depth=depth,
            transposition_table=tt,
            verbose=verbose
        )
        total_time = time.time() - start_time

        total_nodes = sum(r.nodes_searched for r in result['results'])
        nodes_per_sec = total_nodes / total_time if total_time > 0 else 0

        all_results.append({
            'depth': depth,
            'score': result['score'],
            'total': result['total'],
            'percentage': result['percentage'],
            'avg_time': result['avg_time'],
            'total_time': total_time,
            'total_nodes': total_nodes,
            'nodes_per_sec': nodes_per_sec,
            'tt_hits': tt.hits,
            'tt_misses': tt.misses,
            'results': result['results']
        })

        print(f"\nResults at depth {depth}:")
        print(f"  Correct: {result['score']}/{result['total']} ({result['percentage']:.1f}%)")
        print(f"  Total time: {format_time(total_time)}")
        print(f"  Avg time per position: {format_time(result['avg_time'])}")
        print(f"  Total nodes: {total_nodes:,}")
        print(f"  Nodes/sec: {nodes_per_sec:,.0f}")
        print(f"  TT hits: {tt.hits:,} ({100 * tt.hits / (tt.hits + tt.misses):.1f}%)" if (tt.hits + tt.misses) > 0 else "")

        failed = [r for r in result['results'] if not r.correct]
        if failed and verbose:
            print(f"\n  Failed positions:")
            for r in failed:
                print(f"    {r.position.id}: Expected {r.position.best_moves}, got {r.found_move}")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Depth':<8} {'Correct':<12} {'%':<8} {'Avg Time':<12} {'Nodes/sec':<15} {'TT Hit %':<10}")
    print("-" * 80)

    for r in all_results:
        tt_hit_rate = 100 * r['tt_hits'] / (r['tt_hits'] + r['tt_misses']) if (r['tt_hits'] + r['tt_misses']) > 0 else 0
        print(f"{r['depth']:<8} {r['score']}/{r['total']:<8} {r['percentage']:<7.1f}% {format_time(r['avg_time']):<12} {r['nodes_per_sec']:>12,.0f}  {tt_hit_rate:>8.1f}%")

    print("=" * 80)

    position_results = {}
    for r in all_results:
        for pos_result in r['results']:
            pos_id = pos_result.position.id
            if pos_id not in position_results:
                position_results[pos_id] = []
            position_results[pos_id].append(pos_result.correct)

    always_failed = [pos_id for pos_id, results in position_results.items()
                     if not any(results)]

    if always_failed:
        print(f"\nPositions that failed at all depths: {', '.join(sorted(always_failed))}")

    # Best improvement across depths
    if len(all_results) >= 2:
        improvement = all_results[-1]['score'] - all_results[0]['score']
        print(f"\nImprovement from depth {depths[0]} to {depths[-1]}: +{improvement} positions")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run Bratko-Kopec benchmark at multiple depths"
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="3,4,5",
        help="Comma-separated list of depths to test (default: 3,4,5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each position"
    )

    args = parser.parse_args()

    try:
        depths = [int(d.strip()) for d in args.depths.split(",")]
    except ValueError:
        print("Error: depths must be comma-separated integers")
        sys.exit(1)

    try:
        run_benchmark(depths, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
