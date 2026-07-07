"""Evolve: Adaptive Priority Engine — backend package.

This package contains the engine's core logic:

- :mod:`backend.utils`     — helper utilities (code parsing, timestamps, json).
- :mod:`backend.prompts`   — standard prompt templates.
- :mod:`backend.llm`       — chat response generator + dynamic persona + citations.
- :mod:`backend.memory`    — mock vector-graph memory DB (remember/recall/improve/forget).
- :mod:`backend.context`   — context routing + temporal analysis + keyword fail-safe.
- :mod:`backend.patterns`  — analytical drift metrics + hidden pattern parsing.
"""

__all__ = ["utils", "prompts", "llm", "memory", "context", "patterns"]
