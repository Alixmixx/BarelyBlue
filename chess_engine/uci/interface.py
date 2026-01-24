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
import logging
from pathlib import Path
from typing import Optional
from chess_engine.search.minimax import find_best_move
from chess_engine.search.transposition import TranspositionTable
from chess_engine.evaluation.classical import ClassicalEvaluator


def setup_logger(debug=True):
    """
    Setup file-based logger for UCI debugging.

    Args:
        debug: If True, log at DEBUG level; otherwise INFO level

    Returns:
        Configured logger instance
    """
    log_dir = Path.home() / ".barelyblue"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "engine.log"

    logger = logging.getLogger("barelyblue")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    logger.handlers.clear()

    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


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

    def __init__(self, evaluator=None, tt_size=10000000, debug=True):
        """
        Initialize UCI engine.

        Args:
            evaluator: Position evaluator (default: ClassicalEvaluator)
            tt_size: Transposition table size in entries (default: 10M)
            debug: Enable debug logging (default: True)
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

        self.logger = setup_logger(debug=debug)
        self.logger.info("=== BarelyBlue Engine Started ===")
        self.logger.info(f"Log file: {Path.home() / '.barelyblue' / 'engine.log'}")

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

                self.logger.debug(f">>> {command}")

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
                    self.logger.debug(f"Unknown command ignored: {command}")

            except EOFError:
                self.logger.info("EOF received, shutting down")
                break
            except Exception as e:
                self.logger.error(f"Command error: {e}", exc_info=True)
                print(f"# Error: {e}", file=sys.stderr)

    def handle_uci(self):
        """
        Handle 'uci' command - identify engine.

        Response:
            id name BarelyBlue 1.0
            id author Alix Muller
            uciok
        """
        self.logger.info("Handling: uci")

        print(f"id name {self.name} {self.version}")
        print(f"id author {self.author}")

        # Send UCI options (at least one required for protocol compliance)
        print("option name Hash type spin default 16 min 1 max 1024")

        print("uciok")
        sys.stdout.flush()

        self.logger.debug(f"<<< id name {self.name} {self.version}")
        self.logger.debug(f"<<< id author {self.author}")
        self.logger.debug("<<< option name Hash type spin default 16 min 1 max 1024")
        self.logger.debug("<<< uciok")

    def handle_isready(self):
        """
        Handle 'isready' command - synchronization.

        Response:
            readyok
        """
        self.logger.info("Handling: isready")
        print("readyok")
        sys.stdout.flush()
        self.logger.debug("<<< readyok")

    def handle_ucinewgame(self):
        """Handle 'ucinewgame' command - reset for new game."""
        self.logger.info("Handling: ucinewgame - resetting board and transposition table")

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
        self.logger.info(f"Handling: position {' '.join(tokens[1:])}")

        if len(tokens) < 2:
            self.logger.warning("Position command with insufficient arguments")
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
                self.logger.debug(f"Set position from FEN: {fen}")
            except ValueError as e:
                self.logger.error(f"Invalid FEN: {e}")
                print(f"# Invalid FEN: {e}", file=sys.stderr)
                return
        else:
            self.logger.warning(f"Unknown position type: {tokens[1]}")
            return

        # Apply moves
        if move_index < len(tokens) and tokens[move_index] == "moves":
            moves_applied = []
            for move_str in tokens[move_index + 1:]:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                        moves_applied.append(move_str)
                    else:
                        self.logger.error(f"Illegal move: {move_str}")
                        print(f"# Illegal move: {move_str}", file=sys.stderr)
                        break
                except ValueError as e:
                    self.logger.error(f"Invalid move format: {move_str} - {e}")
                    print(f"# Invalid move format: {move_str} - {e}", file=sys.stderr)
                    break

            if moves_applied:
                self.logger.debug(f"Applied moves: {' '.join(moves_applied)}")

        fen = self.board.fen()
        self.logger.info(f"Position updated: {fen[:60]}{'...' if len(fen) > 60 else ''}")
        self.logger.debug(f"Full FEN: {fen}")

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
        self.logger.info(f"Handling: go {' '.join(tokens[1:])}")

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
            self.logger.debug("No depth specified, using default depth 5")

        # TODO: Implement time management
        if movetime:
            self.logger.debug(f"movetime parameter: {movetime} ms (not yet implemented)")
        if wtime and btime:
            self.logger.debug(f"Time controls: wtime={wtime} ms, btime={btime} ms (not yet implemented)")

        self.logger.info(f"Starting search thread with depth={depth}")

        # Make a copy of the board for the search thread to avoid race conditions
        board_copy = self.board.copy()

        self.stop_search = False
        self.searching = True
        self.search_thread = threading.Thread(
            target=self._search_thread,
            args=(depth, board_copy)
        )
        self.search_thread.start()

    def _search_thread(self, depth: int, board: chess.Board):
        """
        Background thread for search.

        Runs minimax search and sends result via UCI protocol.

        Args:
            depth: Search depth
            board: Copy of the board to search

        Output:
            info depth X score cp Y nodes Z
            bestmove <move>
        """
        import time
        start_time = time.time()

        try:
            fen = board.fen()
            self.logger.info(f"Search started: depth={depth}, position={fen[:50]}{'...' if len(fen) > 50 else ''}")
            self.logger.debug(f"Full FEN: {fen}")

            best_move, score, nodes_searched, pv = find_best_move(
                board,
                depth,
                self.evaluator,
                self.transposition_table,
                should_stop=lambda: self.stop_search,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)

            if self.stop_search:
                self.logger.info(f"Search stopped early: best_move={best_move.uci() if best_move else 'None'}, score={score:.2f}, nodes={nodes_searched}, time={elapsed_ms}ms")
            else:
                self.logger.info(f"Search complete: best_move={best_move.uci() if best_move else 'None'}, score={score:.2f}, nodes={nodes_searched}, time={elapsed_ms}ms")

            # Send result - always send best move found, even if stopped early
            if best_move:
                # Build complete UCI info string with all required fields
                info_parts = [
                    "info",
                    f"depth {depth}",
                    f"seldepth {depth}",
                    f"score cp {int(score)}",
                    f"nodes {nodes_searched}",
                    f"time {elapsed_ms}",
                ]

                if pv and len(pv) > 0:
                    pv_moves = " ".join([m.uci() for m in pv])
                    info_parts.append(f"pv {pv_moves}")

                info_msg = " ".join(info_parts)
                bestmove_msg = f"bestmove {best_move.uci()}"

                print(info_msg)
                print(bestmove_msg)
                sys.stdout.flush()

                self.logger.debug(f"<<< {info_msg}")
                self.logger.debug(f"<<< {bestmove_msg}")
            else:
                self.logger.error("Search returned None for best_move!")

        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Search error after {elapsed_time:.3f}s: {e}", exc_info=True)
            print(f"# Search error: {e}", file=sys.stderr)

            # Send a legal move as fallback
            legal_moves = list(board.legal_moves)
            if legal_moves:
                fallback_move = legal_moves[0].uci()
                self.logger.warning(f"Using fallback move: {fallback_move}")
                print(f"bestmove {fallback_move}")
                sys.stdout.flush()
                self.logger.debug(f"<<< bestmove {fallback_move}")
            else:
                self.logger.error("No legal moves available for fallback!")

        finally:
            self.searching = False
            self.logger.debug("Search thread finished")

    def handle_stop(self):
        """
        Handle 'stop' command - stop ongoing search.

        Sets stop_search flag and waits for search thread to finish.
        Search will return best move found so far.
        """
        self.logger.info("Handling: stop")
        self.stop_search = True

        if self.search_thread and self.search_thread.is_alive():
            self.logger.debug("Waiting for search thread to finish (timeout=5.0s)")
            self.search_thread.join(timeout=5.0)
            if self.search_thread.is_alive():
                self.logger.warning("Search thread did not finish within timeout")

    def handle_quit(self):
        """Handle 'quit' command - shutdown engine."""
        self.logger.info("Handling: quit - shutting down engine")

        # Wait for search to complete before quitting
        if self.search_thread and self.search_thread.is_alive():
            self.logger.debug("Waiting for search thread to complete before quitting")
            self.search_thread.join()

        self.logger.info("=== BarelyBlue Engine Stopped ===")
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
