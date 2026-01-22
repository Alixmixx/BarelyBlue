"""
Minimax Search with Alpha-Beta Pruning

This module implements the core search algorithm for the chess engine.
Minimax explores the game tree to find the best move, and alpha-beta
pruning dramatically reduces the number of nodes evaluated.

Key Concepts:
    - Minimax: Recursive algorithm that assumes optimal play by both sides
    - Alpha-Beta: Optimization that prunes branches that can't affect result
    - Move Ordering: Evaluate better moves first to maximize pruning
    - Principal Variation (PV): Best line of play found

Algorithm Complexity:
    - Minimax: O(b^d) where b=branching factor (~35), d=depth
    - Alpha-Beta: O(b^(d/2)) with perfect move ordering

References:
    - Minimax: https://www.chessprogramming.org/Minimax
    - Alpha-Beta: https://www.chessprogramming.org/Alpha-Beta
    - Move Ordering: https://www.chessprogramming.org/Move_Ordering
"""

import chess
from typing import Optional, Tuple, List
from chess_engine.evaluation.base import Evaluator, MATE_SCORE
from chess_engine.search.transposition import TranspositionTable, NodeType

MAX_DEPTH = 100  # Maximum search depth


def order_moves(board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
    """
    Order moves to improve alpha-beta pruning efficiency.

    Move ordering is CRITICAL for alpha-beta performance. Good moves should
    be searched first to cause more cutoffs (prunes).

    Ordering Priority:
        Captures (MVV-LVA: Most Valuable Victim - Least Valuable Aggressor)

    Args:
        board: Current board position
        moves: List of legal moves to order

    Returns:
        Sorted list of moves (best moves first)
    """

    def move_score(move: chess.Move) -> int:
        """
        Assign a score to a move for ordering purposes.
        Higher score = searched earlier.
        """
        score = 0

        # Captures: Score by MVV-LVA
        if board.is_capture(move):
            # Get captured piece value (victim)
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                victim_value = get_piece_value(captured_piece.piece_type)
            else:
                victim_value = 100

            # Get attacking piece value (aggressor)
            attacker_piece = board.piece_at(move.from_square)
            if attacker_piece:
                attacker_value = get_piece_value(attacker_piece.piece_type)
            else:
                attacker_value = 100

            # MVV-LVA: High victim value, low attacker value
            score = 10000 + (victim_value - attacker_value // 10)

        # Checks
        board.push(move)
        if board.is_check():
            score += 5000
        board.pop()

        # Promotions
        if move.promotion:
            score += 8000

        # Castling (usually good)
        if board.is_castling(move):
            score += 3000

        return score

    # Sort moves by score (descending)
    return sorted(moves, key=move_score, reverse=True)


def get_piece_value(piece_type: int) -> int:
    """
    Get approximate piece value for move ordering.

    Args:
        piece_type: chess.PAWN, chess.KNIGHT, etc.

    Returns:
        Piece value in centipawns
    """
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,
    }
    return values.get(piece_type, 0)


def minimax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    evaluator: Evaluator,
    transposition_table: Optional[TranspositionTable] = None,
    ply_from_root: int = 0,
    nodes_searched: Optional[List[int]] = None,
) -> float:
    """
    Minimax search with alpha-beta pruning.

    This is the core search function. It recursively explores the game tree,
    assuming both players play optimally, and returns the evaluation of the
    best line found.

    Args:
        board: Current chess position
        depth: Remaining search depth (decrements each recursive call)
        alpha: Alpha value for pruning (best score for maximizer)
        beta: Beta value for pruning (best score for minimizer)
        maximizing_player: True if current player wants to maximize score
        evaluator: Position evaluation function
        transposition_table: Optional cache for previously evaluated positions
        ply_from_root: Distance from root (for mate distance calculation)
        nodes_searched: Optional mutable list [count] to track positions evaluated

    Returns:
        float: Evaluation of the position in centipawns

    Algorithm:
        1. Check if depth = 0 (leaf node) → evaluate position
        2. Generate all legal moves
        3. For each move:
            a. Make move on board
            b. Recursively search (depth - 1)
            c. Undo move
            d. Update alpha/beta
            e. Prune if alpha >= beta
        4. Return best score found

    Example:
        If maximizing_player has found a move with score = 5 (alpha=5),
        and minimizing_player finds a move with score = 3,
        then minimizing_player won't choose this branch (beta cutoff).

    """
    if nodes_searched is not None:
        nodes_searched[0] += 1

    # TODO: Implement transposition table lookup here

    # Base case: Reached leaf node (depth = 0)
    if depth == 0:
        return evaluator.evaluate(board)

    terminal_score = evaluator.evaluate_terminal(board, ply_from_root)
    if terminal_score is not None:
        return terminal_score

    # Generate and order legal moves
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return evaluator.evaluate(board)

    ordered_moves = order_moves(board, legal_moves)

    if maximizing_player:
        # Maximizing player (wants highest score)
        max_eval = -float("inf")
        for move in ordered_moves:
            board.push(move)

            # Recursive call (switch to minimizing player, decrement depth)
            eval_score = minimax(
                board,
                depth - 1,
                alpha,
                beta,
                False,  # Switch to minimizing
                evaluator,
                transposition_table,
                ply_from_root + 1,
                nodes_searched,
            )

            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)

            # Beta cutoff: Minimizing player won't allow this branch
            if beta <= alpha:
                break  # Prune remaining moves

        # TODO: Store result in transposition table
        return max_eval

    else:
        # Minimizing player (wants lowest score)
        min_eval = float("inf")
        for move in ordered_moves:
            board.push(move)

            # Recursive call (switch to maximizing player, decrement depth)
            eval_score = minimax(
                board,
                depth - 1,
                alpha,
                beta,
                True,  # Switch to maximizing
                evaluator,
                transposition_table,
                ply_from_root + 1,
                nodes_searched,
            )

            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)

            # Alpha cutoff: Maximizing player won't allow this branch
            if beta <= alpha:
                break  # Prune remaining moves

        # TODO: Store result in transposition table
        return min_eval


def find_best_move(
    board: chess.Board,
    depth: int,
    evaluator: Evaluator,
    transposition_table: Optional[TranspositionTable] = None,
    verbose: bool = False,
) -> Tuple[chess.Move | None, float, int, List[chess.Move]]:
    """
    Find the best move in the current position.

    Args:
        board: Current chess position
        depth: Search depth (higher = stronger but slower)
        evaluator: Position evaluation function
        transposition_table: Optional cache for position evaluations
        verbose: If True, print search statistics

    Returns:
        Tuple of (best_move, evaluation, nodes, pv)
            - best_move: The best move found
            - evaluation: Score of the best move
            - nodes: Number of positions evaluated
            - pv: Principal variation

    Raises:
        ValueError: If no legal moves available (game over)
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves available")

    best_move = None
    best_score = -float("inf") if board.turn == chess.WHITE else float("inf")

    maximizing = board.turn == chess.WHITE

    ordered_moves = order_moves(board, legal_moves)

    nodes = [0]
    pv = []
    # Search each move
    for move in ordered_moves:
        board.push(move)

        if maximizing:
            # White to move: maximize score
            score = minimax(
                board,
                depth - 1,
                -float("inf"),
                float("inf"),
                False,
                evaluator,
                transposition_table,
                ply_from_root=1,
                nodes_searched=nodes,
            )
        else:
            # Black to move: minimize score
            score = minimax(
                board,
                depth - 1,
                -float("inf"),
                float("inf"),
                True,
                evaluator,
                transposition_table,
                ply_from_root=1,
                nodes_searched=nodes,
            )

        board.pop()

        if maximizing:
            if score > best_score:
                best_score = score
                best_move = move
                pv = [move]
        else:
            if score < best_score:
                best_score = score
                best_move = move
                pv = [move]

        if verbose:
            print(f"Move: {move}, Score: {score:.2f}")

    if verbose:
        print(f"\nNodes searched: {nodes[0]}")
        print(f"Best move: {best_move}, Score: {best_score:.2f}")

    return best_move, best_score, nodes[0], pv
