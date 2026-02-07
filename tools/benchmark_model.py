#!/usr/bin/env python3
"""
Benchmark chess model on Bratko-Kopec test suite.

Compares a trained neural network model against the classical PST evaluator
to measure improvement.

Usage:
    # Benchmark classical baseline
    python tools/benchmark_model.py --model classical --depth 5

    # Benchmark trained neural network
    python tools/benchmark_model.py --model models/checkpoints/best_model.pt --depth 5

    # Compare both
    python tools/benchmark_model.py --compare --depth 5
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import chess
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.evaluation.classical import ClassicalEvaluator
from chess_engine.evaluation.neural import NeuralEvaluator
from chess_engine.search.minimax import find_best_move
from chess_engine.training.model import ChessNet

# Simple test positions for benchmarking
TEST_POSITIONS = [
    # (id, fen, best_move, description)
    ("pos1", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "d2d4", "Opening: Control center"),
    ("pos2", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "d2d4", "Opening: Challenge center"),
    ("pos3", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", "g1f3", "Opening: Develop knight"),
    ("pos4", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "a7a6", "Opening: Attack bishop"),
    ("pos5", "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "b1c3", "Opening: Develop"),
]


def load_bratko_kopec() -> List[Tuple[str, str, str]]:
    """Load test positions for benchmarking.

    Returns:
        List of (id, fen, best_move) tuples
    """
    return [(pos_id, fen, best_move) for pos_id, fen, best_move, _ in TEST_POSITIONS]


def benchmark_model(
    model_type: str,
    model_path: Path = None,
    depth: int = 5,
    verbose: bool = False
) -> Dict[str, any]:
    """Benchmark a model on Bratko-Kopec suite.

    Args:
        model_type: "classical" or "neural"
        model_path: Path to neural model checkpoint (required if model_type="neural")
        depth: Search depth
        verbose: Print detailed results

    Returns:
        Dict with benchmark results
    """
    # Load evaluator
    if model_type == "classical":
        evaluator = ClassicalEvaluator()
        model_name = "Classical (PST)"
    elif model_type == "neural":
        if model_path is None or not model_path.exists():
            raise ValueError(f"Neural model path required and must exist: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Extract model config from checkpoint
        config = checkpoint.get("config")
        if config:
            blocks = config.model_blocks
            channels = config.model_channels
        else:
            # Fallback defaults
            blocks = 5
            channels = 128

        # Create model
        model = ChessNet(blocks=blocks, channels=channels)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        evaluator = NeuralEvaluator(model)
        model_name = f"Neural ({blocks}b{channels}ch)"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load test positions
    positions = load_bratko_kopec()

    # Run benchmark
    correct = 0
    total = len(positions)
    results = []

    logger = logging.getLogger(__name__)
    logger.info(f"Benchmarking {model_name} at depth {depth}")
    logger.info(f"Test suite: {total} test positions")
    logger.info("=" * 60)

    for position_id, fen, expected_move in positions:
        board = chess.Board(fen)

        # Search for best move
        try:
            best_move, eval_score, nodes, pv = find_best_move(board, depth, evaluator)
            predicted_move = best_move.uci() if best_move else None

            # Check if correct
            is_correct = predicted_move == expected_move
            if is_correct:
                correct += 1

            results.append({
                "id": position_id,
                "fen": fen,
                "expected": expected_move,
                "predicted": predicted_move,
                "eval": eval_score,
                "correct": is_correct,
            })

            if verbose:
                status = "✓" if is_correct else "✗"
                logger.info(
                    f"{status} {position_id}: {predicted_move} "
                    f"(expected {expected_move}, eval={eval_score})"
                )

        except Exception as e:
            logger.error(f"Error on {position_id}: {e}")
            results.append({
                "id": position_id,
                "fen": fen,
                "expected": expected_move,
                "predicted": None,
                "eval": None,
                "correct": False,
            })

    accuracy = correct / total * 100 if total > 0 else 0.0

    logger.info("=" * 60)
    logger.info(f"Results: {correct}/{total} correct ({accuracy:.1f}%)")
    logger.info("=" * 60)

    return {
        "model_name": model_name,
        "model_type": model_type,
        "depth": depth,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


def compare_models(depth: int = 5, neural_model: Path = None):
    """Compare classical and neural models side-by-side.

    Args:
        depth: Search depth
        neural_model: Path to neural model checkpoint
    """
    logger = logging.getLogger(__name__)

    # Benchmark classical
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE: Classical Evaluator")
    logger.info("=" * 60)
    classical_results = benchmark_model("classical", depth=depth, verbose=False)

    # Benchmark neural if available
    if neural_model and neural_model.exists():
        logger.info("\n" + "=" * 60)
        logger.info("TRAINED: Neural Network")
        logger.info("=" * 60)
        neural_results = benchmark_model("neural", model_path=neural_model, depth=depth, verbose=False)

        # Compare
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON")
        logger.info("=" * 60)

        logger.info(f"Classical: {classical_results['correct']}/{classical_results['total']} "
                   f"({classical_results['accuracy']:.1f}%)")
        logger.info(f"Neural:    {neural_results['correct']}/{neural_results['total']} "
                   f"({neural_results['accuracy']:.1f}%)")

        improvement = neural_results['accuracy'] - classical_results['accuracy']
        if improvement > 0:
            logger.info(f"Improvement: +{improvement:.1f}% ✓")
        elif improvement < 0:
            logger.info(f"Regression: {improvement:.1f}% ✗")
        else:
            logger.info("No change")

        # Show position-by-position differences
        logger.info("\nPosition-by-position differences:")
        for cls_res, neu_res in zip(classical_results['results'], neural_results['results']):
            if cls_res['correct'] != neu_res['correct']:
                if neu_res['correct']:
                    logger.info(f"  {cls_res['id']}: Classical ✗ → Neural ✓")
                else:
                    logger.info(f"  {cls_res['id']}: Classical ✓ → Neural ✗")
    else:
        logger.warning("Neural model not found, skipping comparison")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark chess model on Bratko-Kopec suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="classical",
        help="Model type: 'classical' or path to neural model checkpoint",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Search depth",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare classical vs neural models",
    )
    parser.add_argument(
        "--neural-model",
        type=str,
        default="models/checkpoints/best_model.pt",
        help="Path to neural model for comparison",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    try:
        if args.compare:
            # Compare mode
            neural_path = Path(args.neural_model) if args.neural_model else None
            compare_models(depth=args.depth, neural_model=neural_path)
        else:
            # Single model mode
            if args.model == "classical":
                results = benchmark_model("classical", depth=args.depth, verbose=args.verbose)
            else:
                model_path = Path(args.model)
                results = benchmark_model("neural", model_path=model_path, depth=args.depth, verbose=args.verbose)

    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
