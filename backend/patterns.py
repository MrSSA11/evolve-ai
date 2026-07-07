"""Analytical drift metrics & hidden pattern parsing for Evolve.

Tracks focus-mode adjustments across the session. When the user shifts focus
modes repeatedly, a pattern is flagged: the engine prompts the AI to identify
task-switching friction and emits a **Root Cause** analysis plus a concrete
behavioral adjustment rule (consumed by the Autonomous Resolution Engine).
"""
from __future__ import annotations

import json
import re

from . import prompts, utils
from .llm import call_llm

# Switches beyond this trigger friction detection.
FRICTION_THRESHOLD = 3
# Friction score at/above this also triggers detection.
FRICTION_SCORE_TRIGGER = 0.5


def record_mode_switch(session_state, new_mode: str, reason: str = "") -> list[dict]:
    """Record a focus transition in the session (only when the mode actually changes)."""
    history = list(session_state.get("mode_history", []) or [])
    last = history[-1] if history else None
    if last != new_mode:
        history.append(new_mode)
        session_state.mode_history = history
        switches = list(session_state.get("mode_switch_log", []) or [])
        switches.append({"mode": new_mode, "timestamp": utils.now_iso(), "reason": reason})
        session_state.mode_switch_log = switches
    return list(session_state.get("mode_switch_log", []) or [])


def compute_metrics(session_state) -> dict:
    """Compute raw analytical drift metrics from the mode history."""
    history = list(session_state.get("mode_history", []) or [])
    transitions: list[tuple[str, str]] = []
    for i in range(1, len(history)):
        if history[i] != history[i - 1]:
            transitions.append((history[i - 1], history[i]))
    distinct = len(set(history))
    # Friction: normalise transitions against an arbitrary ceiling of 6.
    friction = min(1.0, len(transitions) / 6.0)
    return {
        "total_modes": len(history),
        "distinct_modes": distinct,
        "switches": len(transitions),
        "transitions": transitions,
        "switch_events": len(session_state.get("mode_switch_log", []) or []),
        "friction_score": round(friction, 2),
    }


def detect_friction(session_state) -> tuple[dict, bool]:
    """Return ``(metrics, is_friction_detected)``."""
    metrics = compute_metrics(session_state)
    detected = (
        metrics["switches"] >= FRICTION_THRESHOLD
        or metrics["friction_score"] >= FRICTION_SCORE_TRIGGER
    )
    return metrics, detected


def _parse_pattern(raw: str) -> dict | None:
    """Parse an LLM pattern-analysis response; None if it can't be parsed."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {
                "root_cause": str(data.get("root_cause", "")).strip(),
                "recommendation": str(data.get("recommendation", "")).strip(),
                "recommendation_rule": str(data.get("recommendation_rule", "")).strip(),
            }
    except (ValueError, TypeError):
        pass
    return None


def _heuristic_pattern(metrics: dict, history: list[str]) -> dict:
    """Offline fallback: derive a root cause + rule from the metrics alone."""
    transitions = metrics.get("transitions", [])
    if transitions:
        # Most common "into" mode = where the user keeps landing.
        targets = [t[1] for t in transitions]
        fav = max(set(targets), key=targets.count) if targets else "Builder"
    else:
        fav = "Builder"
    distinct = metrics.get("distinct_modes", 1)
    root = (
        f"The user oscillated between {distinct} focus modes "
        f"({metrics.get('switches', 0)} switches), suggesting reactive context "
        f"switching rather than deliberate planning. Attention keeps returning "
        f"toward {fav} work."
    )
    rec = (
        "Batch similar tasks together and finish a micro-goal before switching; "
        "let Evolve guard the boundaries for you."
    )
    rule = (
        f"When the user is in {fav} mode, gently discourage further mode switches by "
        "first confirming whether the current task is saved/complete, and surface a "
        "one-line focus anchor before responding to the new direction."
    )
    return {
        "root_cause": root,
        "recommendation": rec,
        "recommendation_rule": rule,
    }


def analyze_patterns(session_state) -> dict:
    """Full pattern analysis. Returns ``{"detected": bool, ...}``."""
    metrics, friction = detect_friction(session_state)
    if not friction:
        return {"detected": False, "metrics": metrics}

    history = list(session_state.get("mode_history", []) or [])
    trans_summary = (
        ", ".join(f"{a}->{b}" for a, b in metrics["transitions"]) or "none"
    )
    prompt = prompts.PATTERN_ANALYSIS_PROMPT.format(
        mode_history=" -> ".join(history) or "Personal",
        switches=metrics["switches"],
        transitions=trans_summary,
        friction=metrics["friction_score"],
    )

    parsed: dict | None = None
    try:
        raw = call_llm(prompt, temperature=0.4, max_tokens=320)
        parsed = _parse_pattern(raw)
    except Exception:
        parsed = None
    if not parsed or not parsed.get("root_cause"):
        parsed = _heuristic_pattern(metrics, history)

    parsed["detected"] = True
    parsed["metrics"] = metrics
    return parsed
