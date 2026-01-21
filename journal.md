# BarelyBlue Chess Engine - Development Journal

---

## 2026-01-21: Classical piece-square table evaluator

Implemented traditional evaluation using material counting and piece-square tables. Material values: P=100, N=320, B=330, R=500, Q=900. PSTs provide positional bonuses (e.g., central knights are stronger, edge knights weaker). This becomes the baseline evaluator for the classical engine. This was possible thanks to this amazing chess wiki page: [Simplified Evaluation Function](https://www.chessprogramming.org/Simplified_Evaluation_Function)

## 2026-01-21: Abstract evaluator interface

Created abstract base class for position evaluators. Defines evaluate() method that returns centipawns from White's perspective. Includes helper methods for terminal position detection (checkmate, draws). This interface allows swapping between classical and neural evaluators without touching search code.

## 2026-01-21: Board tensor representation

Implemented 12-channel tensor encoding for chess positions (6 piece types * 2 colors). Each channel is an 8*8 binary mask showing where pieces are located. This prepares the codebase for neural network integration while remaining unused in the classical engine. Includes bidirectional conversion (board ↔ tensor) and coordinate mapping utilities.

## 2026-01-21: Project foundation

Added project dependencies and package structure. The engine uses a modular design where evaluation functions are swappable - the search algorithm doesn't care if positions are evaluated classically or with a neural network. This abstraction lets us start with traditional piece-square tables and upgrade to ML later without rewriting the search code.
