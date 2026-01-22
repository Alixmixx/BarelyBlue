"""
UCI Protocol Implementation

This module implements the Universal Chess Interface (UCI) protocol for
communication between the chess engine and GUI applications.
The engine receives total time remaining and must allocate time per move.

UCI Commands Supported:
    - uci: Identify engine
    - isready: Synchronization check
    - ucinewgame: Start new game
    - position: Set board position
    - go: Start searching
    - stop: Stop searching
    - quit: Shutdown engine

Threading:
    - Main thread: Listen for UCI commands
    - Search thread: Run minimax search
    - Communication: Thread-safe stop flag

References:
    - UCI Protocol: https://www.chessprogramming.org/UCI
    - Example Engines: python-chess includes a UCI wrapper
"""

import chess
import sys
import threading
from typing import Optional
from chess_engine.search.minimax import find_best_move
from chess_engine.search.transposition import TranspositionTable
from chess_engine.evaluation.classical import ClassicalEvaluator


class UCIEngine:
    """
    UCI-compliant chess engine interface.

    This class handles all UCI communication and coordinates the search
    algorithm with the evaluation function.

    Attributes:
        board: Current chess position
        evaluator: Position evaluation function
        transposition_table: Cache for position evaluations
        searching: Flag indicating if search is in progress
        stop_search: Flag to stop ongoing search
        search_thread: Background thread for search

    Methods:
        run: Main UCI command loop
        handle_uci: Respond to 'uci' command
        handle_isready: Respond to 'isready' command
        handle_position: Set board position
        handle_go: Start search
        handle_stop: Stop search
        handle_quit: Shutdown engine
    """

    def __init__(self, evaluator=None, tt_size=10000000):
        """
        Initialize UCI engine.

        Args:
            evaluator: Position evaluator (default: ClassicalEvaluator)
            tt_size: Transposition table size in entries (default: 10M)
        """
        self.board = chess.Board()
        self.evaluator = evaluator if evaluator else ClassicalEvaluator()
        self.transposition_table = TranspositionTable(max_size=tt_size)

        # Search state
        self.searching = False
        self.stop_search = False
        self.search_thread: Optional[threading.Thread] = None

        # Engine info
        self.name = "BarelyBlue"
        self.version = "1.0"
        self.author = "Alix Muller"

    def run(self):
        """
        Main UCI command loop.

        Listens for UCI commands on stdin and responds on stdout.
        Runs until 'quit' command is received.

        Commands:
            - uci: Identify engine
            - isready: Sync check
            - ucinewgame: Reset for new game
            - position [fen | startpos] moves ...
            - go [depth X] [movetime X] [wtime X btime X]
            - stop: Stop search
            - quit: Exit
        """
        while True:
            try:
                # Read command from stdin
                command = input().strip()

                if not command:
                    continue

                # Parse command
                tokens = command.split()
                cmd = tokens[0].lower()

                # Handle commands
                if cmd == "uci":
                    self.handle_uci()

                elif cmd == "isready":
                    self.handle_isready()

                elif cmd == "ucinewgame":
                    self.handle_ucinewgame()

                elif cmd == "position":
                    self.handle_position(tokens)

                elif cmd == "go":
                    self.handle_go(tokens)

                elif cmd == "stop":
                    self.handle_stop()

                elif cmd == "quit":
                    self.handle_quit()
                    break

                else:
                    # Unknown command - UCI spec says to ignore
                    pass

            except EOFError:
                break
            except Exception as e:
                print(f"# Error: {e}", file=sys.stderr)

    def handle_uci(self):
        """
        Handle 'uci' command - identify engine.

        Response:
            id name BarelyBlue 1.0
            id author Alix Muller
            uciok
        """
        print(f"id name {self.name} {self.version}")
        print(f"id author {self.author}")

        # TODO: Add engine options here

        print("uciok")

    def handle_isready(self):
        """
        Handle 'isready' command - synchronization.
        
        Response:
            readyok
        """
        print("readyok")

    def handle_ucinewgame(self):
        """Handle 'ucinewgame' command - reset for new game."""

        self.board = chess.Board()
        self.transposition_table.clear()
        self.stop_search = False

    def handle_position(self, tokens):
        """
        Handle 'position' command - set board position.

        Formats:
            position startpos
            position startpos moves e2e4 e7e5
            position fen <FEN string>
            position fen <FEN string> moves e2e4

        Args:
            tokens: Command tokens (e.g., ['position', 'startpos', 'moves', 'e2e4'])
        """
        if len(tokens) < 2:
            return

        # Parse position type
        if tokens[1] == "startpos":
            self.board = chess.Board()
            move_index = 2
        elif tokens[1] == "fen":
            try:
                moves_index = tokens.index("moves")
                fen = " ".join(tokens[2:moves_index])
                move_index = moves_index
            except ValueError:
                fen = " ".join(tokens[2:])
                move_index = len(tokens)

            try:
                self.board = chess.Board(fen)
            except ValueError as e:
                print(f"# Invalid FEN: {e}", file=sys.stderr)
                return
        else:
            return

        # Apply moves
        if move_index < len(tokens) and tokens[move_index] == "moves":
            for move_str in tokens[move_index + 1:]:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                    else:
                        print(f"# Illegal move: {move_str}", file=sys.stderr)
                        break
                except ValueError as e:
                    print(f"# Invalid move format: {move_str} - {e}", file=sys.stderr)
                    break

    def handle_go(self, tokens):
        """
        Handle 'go' command - start search.

        Formats:
            go depth 5
            go movetime 5000 (search for 5 seconds)
            go wtime 300000 btime 300000 (3 minutes per side)
            go infinite (search until 'stop')

        Args:
            tokens: Command tokens (e.g., ['go', 'depth', '5'])
        """
        depth = None
        movetime = None
        wtime = None
        btime = None
        infinite = False

        i = 1
        while i < len(tokens):
            if tokens[i] == "depth" and i + 1 < len(tokens):
                depth = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "movetime" and i + 1 < len(tokens):
                movetime = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "wtime" and i + 1 < len(tokens):
                wtime = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "btime" and i + 1 < len(tokens):
                btime = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "infinite":
                infinite = True
                i += 1
            else:
                i += 1


        if depth is None:
            depth = 5  # default depth

        # TODO: Implement time management

        self.stop_search = False
        self.searching = True
        self.search_thread = threading.Thread(
            target=self._search_thread,
            args=(depth,)
        )
        self.search_thread.start()

    def _search_thread(self, depth: int):
        """
        Background thread for search.

        Runs minimax search and sends result via UCI protocol.

        Args:
            depth: Search depth

        Output:
            info depth X score cp Y nodes Z
            bestmove <move>
        """
        try:
            best_move, score = find_best_move(
                self.board,
                depth,
                self.evaluator,
                self.transposition_table,
            )

            # Send result
            if not self.stop_search:
                print(f"info depth {depth} score cp {int(score)}")
                print(f"bestmove {best_move.uci()}")

        except Exception as e:
            print(f"# Search error: {e}", file=sys.stderr)
            # Send a legal move as fallback
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                print(f"bestmove {legal_moves[0].uci()}")

        finally:
            self.searching = False

    def handle_stop(self):
        """
        Handle 'stop' command - stop ongoing search.

        Sets stop_search flag and waits for search thread to finish.
        Search should return best move found so far.
        """
        #TODO: implement stopping mid-search.
        self.stop_search = True

        if self.search_thread and self.search_thread.is_alive():
            self.search_thread.join(timeout=1.0)

    def handle_quit(self):
        """Handle 'quit' command - shutdown engine."""

        self.handle_stop()
        sys.exit(0)


# Example: Running the Engine
#
# To run the engine as a standalone UCI program:
#
#   if __name__ == "__main__":
#       engine = UCIEngine()
#       engine.run()
#
# Then use with a GUI:
#   1. Install Arena or any chess GUI
#   2. Add engine: python -m chess_engine.uci.interface
#   3. Play against engine
