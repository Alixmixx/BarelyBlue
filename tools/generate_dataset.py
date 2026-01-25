#!/usr/bin/env python3
"""
CLI tool for generating chess training datasets.

Usage:
    python tools/generate_dataset.py generate \\
        --pgn data/lichess_elite.pgn \\
        --output data/training.h5 \\
        --max-positions 100000 \\
        --stockfish-depth 15

    python tools/generate_dataset.py validate \\
        data/training.h5 \\
        --output-report data/validation.md
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.data.pipeline import DataPipeline, PipelineConfig
from chess_engine.data.validator import DatasetValidator


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def generate_dataset(args):
    """Run dataset generation pipeline."""
    # Parse PGN sources
    pgn_sources = []
    for pgn_str in args.pgn:
        pgn_path = Path(pgn_str)
        if not pgn_path.exists():
            print(f"Error: PGN file not found: {pgn_path}")
            sys.exit(1)
        pgn_sources.append(pgn_path)

    output_path = Path(args.output)

    # Check if output exists
    if output_path.exists() and not args.overwrite:
        print(f"Error: Output file already exists: {output_path}")
        print("Use --overwrite to replace it")
        sys.exit(1)

    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 < total_ratio < 1.01):
        print(f"Error: Split ratios must sum to 1.0, got {total_ratio}")
        sys.exit(1)

    # Build config
    config = PipelineConfig(
        min_elo=args.min_elo,
        max_games=args.max_games,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        min_pieces=args.min_pieces,
        stockfish_path=args.stockfish_path,
        stockfish_depth=args.stockfish_depth,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_positions=args.max_positions,
        batch_size=args.batch_size,
        validate_output=not args.no_validate,
        enable_dedup=not args.no_dedup,
    )

    # Run pipeline
    pipeline = DataPipeline(
        pgn_sources=pgn_sources,
        output_path=output_path,
        config=config,
    )

    pipeline.run()

    print(f"\nDataset generated successfully: {output_path}")


def validate_dataset(args):
    """Run validation on existing dataset."""
    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    validator = DatasetValidator(dataset_path)

    output_path = None
    if args.output_report:
        output_path = Path(args.output_report)

    report = validator.generate_report(
        output_path=output_path,
        check_duplicates=not args.no_duplicates,
        duplicate_sample_size=args.duplicate_sample,
    )

    print(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate or validate chess training datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate new dataset")

    gen_parser.add_argument(
        "--pgn",
        nargs="+",
        required=True,
        help="PGN file(s) to process (space-separated for multiple files)",
    )
    gen_parser.add_argument(
        "--output",
        required=True,
        help="Output HDF5 dataset path",
    )
    gen_parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Maximum positions to generate (default: unlimited)",
    )
    gen_parser.add_argument(
        "--min-elo",
        type=int,
        default=2000,
        help="Minimum player ELO",
    )
    gen_parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum games to parse (default: unlimited)",
    )
    gen_parser.add_argument(
        "--min-ply",
        type=int,
        default=10,
        help="Minimum ply (skip opening book)",
    )
    gen_parser.add_argument(
        "--max-ply",
        type=int,
        default=100,
        help="Maximum ply (skip long games)",
    )
    gen_parser.add_argument(
        "--min-pieces",
        type=int,
        default=6,
        help="Minimum pieces (skip endgames)",
    )
    gen_parser.add_argument(
        "--stockfish-path",
        type=str,
        default=None,
        help="Path to Stockfish binary (default: auto-detect)",
    )
    gen_parser.add_argument(
        "--stockfish-depth",
        type=int,
        default=15,
        help="Stockfish search depth",
    )
    gen_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split ratio",
    )
    gen_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    gen_parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio",
    )
    gen_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for Stockfish evaluation",
    )
    gen_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after generation",
    )
    gen_parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable position deduplication",
    )
    gen_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )
    gen_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    val_parser = subparsers.add_parser("validate", help="Validate existing dataset")

    val_parser.add_argument(
        "dataset",
        help="HDF5 dataset path to validate",
    )
    val_parser.add_argument(
        "--output-report",
        help="Path to write validation report (default: print to stdout)",
    )
    val_parser.add_argument(
        "--no-duplicates",
        action="store_true",
        help="Skip duplicate checking (faster)",
    )
    val_parser.add_argument(
        "--duplicate-sample",
        type=int,
        default=10000,
        help="Sample size for duplicate check",
    )
    val_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    setup_logging(verbose=args.verbose)

    try:
        if args.command == "generate":
            generate_dataset(args)
        elif args.command == "validate":
            validate_dataset(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
