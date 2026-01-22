"""
Main entry point for running BarelyBlue as a UCI engine.

Usage:
    python -m chess_engine.uci
"""

from chess_engine.uci.interface import UCIEngine

if __name__ == "__main__":
    engine = UCIEngine()
    engine.run()
