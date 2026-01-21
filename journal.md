# BarelyBlue Chess Engine - Development Journal

---

## 2026-01-21: Project foundation

Added project dependencies and package structure. The engine uses a modular design where evaluation functions are swappable - the search algorithm doesn't care if positions are evaluated classically or with a neural network. This abstraction lets us start with traditional piece-square tables and upgrade to ML later without rewriting the search code.
