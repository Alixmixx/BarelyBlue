#!/usr/bin/env python3
"""
Train chess neural network.

Usage:
    python tools/train_model.py \\
        --dataset data/training.h5 \\
        --blocks 5 \\
        --channels 128 \\
        --epochs 50 \\
        --batch-size 256
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine.training.config import TrainingConfig
from chess_engine.training.dataset import ChessDataset
from chess_engine.training.model import ChessNet
from chess_engine.training.trainer import ChessTrainer


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train chess neural network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model architecture
    parser.add_argument(
        "--blocks",
        type=int,
        default=5,
        choices=[3, 5, 10, 15, 20],
        help="Number of residual blocks",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=128,
        choices=[64, 128, 256, 512],
        help="Number of channels per conv layer",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to HDF5 dataset file",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization weight decay",
    )

    # Optimization
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training (cuda/cpu)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.001,
        help="Minimum change to qualify as improvement",
    )

    # Learning rate scheduling
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="LR scheduler patience (epochs before reducing LR)",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Factor to reduce LR by",
    )

    # Logging
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log every N batches (0 to disable progress bar)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (0 for random)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # Set random seed
    if args.seed > 0:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        logger.info(f"Set random seed: {args.seed}")

    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    # Create config
    config = TrainingConfig(
        model_blocks=args.blocks,
        model_channels=args.channels,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        weight_decay=args.weight_decay,
        dataset_path=dataset_path,
        device=args.device,
        num_workers=args.num_workers,
        checkpoint_dir=Path(args.checkpoint_dir),
        save_every=args.save_every,
        patience=args.patience,
        min_delta=args.min_delta,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        log_interval=args.log_interval,
        enable_tensorboard=not args.no_tensorboard,
        random_seed=args.seed if args.seed > 0 else None,
    )

    logger.info("Configuration:")
    logger.info(config)

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = ChessDataset(dataset_path, split="train")
    val_dataset = ChessDataset(dataset_path, split="validation")

    logger.info(f"Training samples: {len(train_dataset):,}")
    logger.info(f"Validation samples: {len(val_dataset):,}")

    # Print dataset statistics
    logger.info("Dataset statistics:")
    train_stats = train_dataset.get_statistics()
    for key, value in train_stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
    )

    # Create model
    logger.info(f"Creating model: {args.blocks} blocks, {args.channels} channels")
    model = ChessNet(blocks=args.blocks, channels=args.channels)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Create trainer
    trainer = ChessTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            trainer.load_checkpoint(resume_path)
        else:
            logger.error(f"Checkpoint not found: {resume_path}")
            sys.exit(1)

    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("\n\nTraining interrupted by user")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nTraining failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Print final results
    logger.info("\n" + "=" * 60)
    logger.info("Training completed successfully!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {config.checkpoint_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
