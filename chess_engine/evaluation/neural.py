"""
Neural network-based position evaluator.

Uses a trained ResNet model to evaluate chess positions.
"""

import torch
import chess

from chess_engine.evaluation.base import Evaluator
from chess_engine.training.model import ChessNet
from chess_engine.board.representation import board_to_tensor_18


class NeuralEvaluator(Evaluator):
    """Neural network position evaluator.

    Uses a trained ChessNet model to evaluate positions. The model outputs
    a value in [-1, 1] range which is scaled to centipawns.
    """

    def __init__(self, model: ChessNet, device: str = "cpu"):
        """Initialize neural evaluator.

        Args:
            model: Trained ChessNet model
            device: Device to run model on ("cpu" or "cuda")
        """
        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode
        self.device = device

    def evaluate(self, board: chess.Board) -> float:
        """Evaluate position using neural network.

        Args:
            board: Chess position to evaluate

        Returns:
            Evaluation in centipawns from White's perspective
        """
        # Handle terminal positions
        if board.is_checkmate():
            return -10000.0 if board.turn == chess.WHITE else 10000.0
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        # Convert board to tensor
        tensor = board_to_tensor_18(board)

        # Add batch dimension and convert to torch tensor
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

        # Run through model
        with torch.no_grad():
            output = self.model(tensor)

        # Convert to centipawns
        # Model outputs in [-1, 1] range, scale to centipawns
        # We use a scaling factor of 500 (so ±1 = ±500 centipawns)
        eval_score = output.item() * 500.0

        return eval_score
