"""
Evaluation Module

This module provides position evaluation functions for the chess engine.
The key design principle is that evaluators are SWAPPABLE - the search
algorithm should work with any evaluator that implements the base interface.
This will allow modular optimization and easy comparison

Key Components:
    - Evaluator (ABC): Abstract base class defining the evaluation interface
    - ClassicalEvaluator: Traditional piece-square table evaluation
    - [Future] NeuralEvaluator: Deep learning based evaluation

Data Flow:
    chess.Board → evaluator.evaluate() → float (centipawns)
                                          Positive = White advantage
                                          Negative = Black advantage

"""

from chess_engine.evaluation.base import Evaluator
# from chess_engine.evaluation.classical import ClassicalEvaluator

__all__ = ['Evaluator']  # ClassicalEvaluator will be added in next commit
