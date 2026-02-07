# BarelyBlue Chess Engine - Development Journal

---

## 2026-01-26: Training Loop and Checkpointing

Implemented ChessTrainer for complete training pipeline. Features: training loop with tqdm progress bars, validation with sign accuracy metric (correctly predicting winning side), MSE loss optimization, Adam optimizer with weight decay, ReduceLROnPlateau scheduler, early stopping based on validation loss, checkpoint saving (regular + best model), TensorBoard logging. Created CLI tool (tools/train_model.py) for model training with argparse interface. Handles checkpoint resume, configurable hyperparameters, and graceful interruption. Test suite: 16 tests (epoch training, validation, checkpointing, early stopping, LR scheduling, device placement). Total: 170 tests passing (12 skipped).

## 2026-01-26: ResNet Model Architecture

Implemented ChessNet ResNet architecture for chess position evaluation. Created ResidualBlock with skip connections (Conv->BN->ReLU->Conv->BN->(+x)->ReLU) for gradient flow through deep networks. ChessNet takes 18-channel board tensors (12 piece planes + 6 metadata), processes through configurable residual tower (3/5/10/15/20 blocks), and outputs scalar evaluation in [-1, 1] via tanh. Value head uses 1x1 conv + dense layers. Model sizes: small (~750K params), medium (~2M params), large (~12M params). Added create_model() factory function for preset configurations. Architecture inspired by AlphaZero.

## 2026-01-26: PyTorch Infrastructure and Dataset Loader

Implemented PyTorch training infrastructure with ChessDataset and TrainingConfig. Added dependencies (torch, torchvision, tensorboard) to requirements.txt. Created chess_engine/training/ module with lazy HDF5 loading, automatic evaluation scaling from centipawns to [-1, 1], and support for train/validation/test splits. TrainingConfig dataclass includes hyperparameter validation, CUDA auto-detection, and configurable model architecture (3/5/10/15/20 blocks, 64/128/256/512 channels).

## 2026-01-25: Data Pipeline Complete

Added DatasetValidator with comprehensive quality checks (evaluation distributions, game result balance, duplicate detection via Zobrist hashing, tensor validity). Created DataPipeline for end-to-end orchestration from PGN -> Stockfish labeling -> HDF5 writing. Added CLI tool (tools/generate_dataset.py) with generate/validate subcommands.
Dataset format: 18-channel tensors with train/val/test splits (80/10/10).

## 2026-01-25: Stockfish integration and HDF5 writer

Added StockfishLabeler for ground truth position evaluation at configurable depth (default 15). Supports batch evaluation with multiple workers for parallel processing. Created DatasetWriter for HDF5 dataset creation with train/validation/test splits, automatic ratio-based assignment, and efficient chunked writing. Dataset schema includes tensors (18,8,8), evaluations (centipawns), game_results (1/0/-1), ply, and FEN strings.

## 2026-01-24: Extended tensor representation to 18 channels

Extended board_to_tensor to support 18-channel representation with metadata planes. Added channels 12-17 for castling rights (4 planes), en passant target square, and side to move. These metadata planes provide essential game state information for neural network training. Implemented add_metadata_planes() function.

## 2026-01-24: Position extraction from games

Added PositionExtractor to extract training positions from PGN games with quality filtering. Filters: min_ply (skip opening book), max_ply, min_pieces (skip bare endgames). Skips terminal positions and deduplicates using Zobrist hashing.

## 2026-01-24: PGN parser module

Created chess_engine/data/ module with PGNParser for streaming large PGN files without loading into memory. Supports ELO filtering (min_elo >= 2000) to ensure high-quality training positions from master-level games. 

## 2026-01-22: UCI protocol interface

Implemented Universal Chess Interface protocol for GUI compatibility. Supports standard UCI commands (uci, isready, position, go, stop, quit). Uses threading to handle concurrent search and command processing. Engine can now be used with chess GUIs like Arena, En-croissant, or Lichess bots.
Implementation of the logger and small updates to the chess_engine

## 2026-01-21: Minimax search with alpha-beta pruning and transposition tables

Implemented core search algorithm using minimax with alpha-beta pruning to dramatically reduce nodes explored. Added transposition table using Zobrist hashing to cache previously evaluated positions. Move ordering (captures first, then checks, then quiet moves) improves pruning efficiency. Expected performance: depth 4-5 in 1-3 seconds.

## 2026-01-21: Classical piece-square table evaluator

Implemented traditional evaluation using material counting and piece-square tables. Material values: P=100, N=320, B=330, R=500, Q=900. PSTs provide positional bonuses (e.g., central knights are stronger, edge knights weaker). This becomes the baseline evaluator for the classical engine. This was possible thanks to this amazing chess wiki page: [Simplified Evaluation Function](https://www.chessprogramming.org/Simplified_Evaluation_Function)

## 2026-01-21: Abstract evaluator interface

Created abstract base class for position evaluators. Defines evaluate() method that returns centipawns from White's perspective. Includes helper methods for terminal position detection (checkmate, draws). This interface allows swapping between classical and neural evaluators without touching search code.

## 2026-01-21: Board tensor representation

Implemented 12-channel tensor encoding for chess positions (6 piece types * 2 colors). Each channel is an 8*8 binary mask showing where pieces are located. This prepares the codebase for neural network integration while remaining unused in the classical engine. Includes bidirectional conversion (board ↔ tensor) and coordinate mapping utilities.

## 2026-01-21: Project foundation

Added project dependencies and package structure. The engine uses a modular design where evaluation functions are swappable - the search algorithm doesn't care if positions are evaluated classically or with a neural network. This abstraction lets us start with traditional piece-square tables and upgrade to ML later without rewriting the search code.
