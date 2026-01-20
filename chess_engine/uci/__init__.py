"""
UCI Protocol Interface

This module implements the Universal Chess Interface (UCI) protocol,
which allows the engine to communicate with chess GUIs like Arena,
En-croissant, and Lichess.

UCI is the standard protocol for chess engines. Supporting UCI makes
the engine compatible with virtually all chess software.

Protocol Flow:
    GUI → "uci"
    Engine → "id name BarelyBlue"
    Engine → "id author Your Name"
    Engine → "uciok"
    GUI → "isready"
    Engine → "readyok"
    GUI → "position startpos moves e2e4"
    GUI → "go wtime 300000 btime 300000"
    Engine → "info depth 5 score cp 25 nodes 12345"
    Engine → "bestmove e7e5"

Reference:
    UCI Protocol: https://www.chessprogramming.org/UCI
"""

from chess_engine.uci.interface import UCIEngine

__all__ = ['UCIEngine']
