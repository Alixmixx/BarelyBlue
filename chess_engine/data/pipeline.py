"""
End-to-end data pipeline for chess position dataset generation.

Orchestrates PGN parsing, position extraction, Stockfish labeling,
and HDF5 dataset writing.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from tqdm import tqdm

from chess_engine.board.representation import board_to_tensor_18
from chess_engine.data.dataset_writer import DatasetWriter
from chess_engine.data.pgn_parser import PGNParser
from chess_engine.data.position_extractor import ExtractedPosition, PositionExtractor
from chess_engine.data.stockfish_labeler import StockfishLabeler
from chess_engine.data.validator import DatasetValidator

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for data pipeline."""

    # PGN filtering
    min_elo: int = 2000
    max_games: Optional[int] = None

    # Position extraction
    min_ply: int = 10
    max_ply: int = 100
    min_pieces: int = 6

    # Stockfish labeling
    stockfish_path: Optional[str] = None
    stockfish_depth: int = 15

    # Dataset writing
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    chunk_size: int = 1000

    # Pipeline control
    max_positions: Optional[int] = None
    batch_size: int = 100
    validate_output: bool = True

    # Deduplication
    enable_dedup: bool = True


class DataPipeline:
    """End-to-end pipeline for generating chess training datasets."""

    def __init__(
        self,
        pgn_sources: List[Path],
        output_path: Path,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize data pipeline.

        Args:
            pgn_sources: List of PGN file paths to process
            output_path: Path to output HDF5 dataset
            config: Pipeline configuration (uses defaults if None)

        Raises:
            FileNotFoundError: If any PGN source doesn't exist
        """
        for pgn_path in pgn_sources:
            if not pgn_path.exists():
                raise FileNotFoundError(f"PGN file not found: {pgn_path}")

        self.pgn_sources = pgn_sources
        self.output_path = output_path
        self.config = config or PipelineConfig()

        self.pgn_parser = PGNParser(
            min_elo=self.config.min_elo,
            max_games=self.config.max_games,
        )

        self.position_extractor = PositionExtractor(
            min_ply=self.config.min_ply,
            max_ply=self.config.max_ply,
            min_pieces=self.config.min_pieces,
        )

        self.stockfish_labeler = StockfishLabeler(
            stockfish_path=self.config.stockfish_path,
            depth=self.config.stockfish_depth,
        )

        self.writer = DatasetWriter(
            output_path=self.output_path,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            chunk_size=self.config.chunk_size,
        )

        logger.info(f"Initialized pipeline: {len(pgn_sources)} PGN sources")
        logger.info(f"Output: {output_path}")

    def run(self):
        """
        Execute the complete data pipeline.

        Raises:
            RuntimeError: If pipeline fails at any step
        """
        try:
            logger.info("Starting data pipeline...")

            # Create HDF5 datasets
            self.writer.create_datasets()

            total_positions = 0
            batch_positions = []

            for position in self._extract_all_positions():
                batch_positions.append(position)

                if len(batch_positions) >= self.config.batch_size:
                    self._process_batch(batch_positions)
                    total_positions += len(batch_positions)
                    batch_positions = []

                    if (
                        self.config.max_positions
                        and total_positions >= self.config.max_positions
                    ):
                        logger.info(
                            f"Reached max positions limit: {self.config.max_positions}"
                        )
                        break

            if batch_positions:
                self._process_batch(batch_positions)
                total_positions += len(batch_positions)

            self.writer.finalize()

            logger.info(f"Pipeline complete: {total_positions:,} positions processed")

            if self.config.validate_output:
                logger.info("Running validation...")
                validator = DatasetValidator(self.output_path)
                report = validator.generate_report(
                    output_path=self.output_path.parent / "validation_report.md",
                    check_duplicates=True,
                    duplicate_sample_size=10000,
                )
                print(report)

        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user")
            self.writer.finalize()
            raise

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.writer.finalize()
            raise RuntimeError(f"Pipeline failed: {e}") from e

    def _extract_all_positions(self) -> Iterator[ExtractedPosition]:
        """
        Extract positions from all PGN sources.

        Yields:
            ExtractedPosition objects
        """
        for pgn_path in self.pgn_sources:
            logger.info(f"Processing PGN: {pgn_path}")

            for game in self.pgn_parser.parse_file(pgn_path):
                positions = self.position_extractor.extract_positions(game)

                if self.config.enable_dedup:
                    positions = self.position_extractor.deduplicate(positions)

                for position in positions:
                    yield position

    def _process_batch(self, positions: List[ExtractedPosition]):
        """
        Process a batch of positions: label with Stockfish and write to HDF5.

        Args:
            positions: List of extracted positions
        """
        if not positions:
            return

        # Convert to boards
        boards = [pos.board for pos in positions]

        # Evaluate with Stockfish (with progress bar)
        evaluations_objs = []
        for board in tqdm(
            boards, desc=f"Evaluating batch ({len(positions)} positions)", leave=False
        ):
            eval_obj = self.stockfish_labeler.evaluate_position(board)
            evaluations_objs.append(eval_obj)

        # Convert to tensors
        tensors = np.array([board_to_tensor_18(board) for board in boards])

        # Extract labels
        evaluations = np.array(
            [eval_obj.to_centipawns() for eval_obj in evaluations_objs],
            dtype=np.int16,
        )

        game_results = np.array([pos.game_result for pos in positions], dtype=np.int8)
        ply = np.array([pos.ply for pos in positions], dtype=np.int16)
        fens = [pos.fen for pos in positions]

        # Write to HDF5
        self.writer.append_batch(
            tensors=tensors,
            evaluations=evaluations,
            game_results=game_results,
            ply=ply,
            fens=fens,
            split="auto",
        )

        logger.debug(f"Processed batch: {len(positions)} positions")
