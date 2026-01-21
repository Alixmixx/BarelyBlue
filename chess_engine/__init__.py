"""
BarelyBlue Chess Engine

A UCI-compliant chess engine with classical evaluation and minimax search,
designed for future neural network integration.

## Architecture

The engine is organized into several key modules:

1. **board**: Board representation and tensor conversion
   - Convert chess positions to 12-channel tensors for neural networks
   - Coordinate mapping utilities

2. **evaluation**: Position evaluation functions
   - Abstract Evaluator interface (swappable design)
   - ClassicalEvaluator: Piece-Square Table based evaluation
   - Future: NeuralEvaluator for deep learning integration

3. **search**: Search algorithms
   - Minimax with alpha-beta pruning
   - Transposition table with Zobrist hashing
   - Move ordering heuristics

4. **uci**: Universal Chess Interface protocol
   - UCI command handling
   - Thread-safe search management
   - Compatible with chess GUIs

5. **utils**: Testing and benchmarking utilities
   - Bratko-Kopec test suite
   - Win At Chess (WAC) test suite

## Quick Start

### As a Python Library

```python
import chess
from chess_engine.evaluation import ClassicalEvaluator
from chess_engine.search import find_best_move, TranspositionTable

# Create evaluator and board
evaluator = ClassicalEvaluator()
board = chess.Board()

# Find best move
best_move, score = find_best_move(board, depth=5, evaluator=evaluator)
print(f"Best move: {best_move} (score: {score:.2f})")
```

### As a UCI Engine

```bash
python -m chess_engine.uci.interface
```

Then connect with a chess GUI (Arena, CuteChess, etc.)

## Version

0.1.0
"""

__version__ = "0.1.0"
__author__ = "Alix Muller"
__license__ = "MIT"

# Import key components for easy access
# NOTE: Imports will be uncommented as modules are implemented in subsequent commits
# from chess_engine.evaluation import Evaluator, ClassicalEvaluator
# from chess_engine.search import find_best_move, minimax, TranspositionTable
# from chess_engine.uci import UCIEngine

__all__ = [
    # 'Evaluator',
    # 'ClassicalEvaluator',
    # 'find_best_move',
    # 'minimax',
    # 'TranspositionTable',
    # 'UCIEngine',
]
