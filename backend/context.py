"""Context routing + temporal analysis for Evolve.

Routes the conversation into one of three FOCUS STATES:

- **Builder** — tech / coding / architecture
- **Exam**    — revision / academic
- **Personal** — casual

Primary path: an LLM call (``gemma-4-31b`` by default) decides the mode from the
chat log + time of day, returning structured reasoning.

Fail-safe path: a deterministic regex keyword scanner runs inside the ``except``
block so that, even when the API is unreachable, the engine still routes states
accurately. The chosen source ("LLM" vs "FALLBACK") is surfaced to the UI.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

from . import prompts, utils
from .llm import call_llm

VALID_MODES = ("Builder", "Exam", "Personal")

# Keyword dictionaries for the regex fail-safe scanner.
BUILDER_KEYWORDS = [
    "code", "coding", "function", "class", "method", "bug", "error", "exception",
    "stack trace", "traceback", "api", "endpoint", "database", "sql", "query",
    "python", "javascript", "typescript", "java", "react", "vue", "node",
    "server", "deploy", "deployment", "docker", "kubernetes", "git", "github",
    "algorithm", "loop", "async", "thread", "concurrency", "frontend", "backend",
    "fullstack", "framework", "library", "compile", "compiler", "refactor",
    "performance", "latency", "auth", "jwt", "oauth", "cache", "redis",
    "schema", "json", "yaml", "regex", "unit test", "ci/cd", "pipeline",
    "linux", "bash", "shell", "architecture", "microservice",
]

EXAM_KEYWORDS = [
    "exam", "exams", "study", "studying", "revise", "revision", "reviewing",
    "learn", "learning", "concept", "concepts", "definition", "formula",
    "theorem", "proof", "derive", "derivation", "equation", "essay", "thesis",
    "homework", "assignment", "coursework", "lecture", "lectures", "notes",
    "memorize", "memorise", "quiz", "test", "mock test", "chapter", "syllabus",
    "curriculum", "subject", "topic", "summarize", "summarise", "flashcard",
    "active recall", "spaced repetition", "grade", "grading", "professor",
    "tutor", "tutorial", "lab report", "citation", "bibliography",
]

PERSONAL_KEYWORDS = [
    "plan", "planning", "idea", "ideas", "recipe", "cook", "travel", "trip",
    "workout", "gym", "exercise", "movie", "film", "book", "music", "game",
    "weekend", "vacation", "holiday", "feeling", "feel", "advice", "draft",
    "email", "message", "schedule", "calendar", "budget", "money", "finance",
    "gift", "birthday", "dinner", "lunch", "restaurant", "journal", "diary",
    "habit", "goal", "goals", "resolution", "relationship", "hobby", "fun",
]


def _time_context(now: datetime | None = None) -> tuple[str, str]:
    """Return (descriptive_phrase, short_label) for the current local time."""
    now = now or datetime.now()
    h = now.hour
    if 5 <= h < 9:
        phrase, label = "early morning — fresh focus window", "early morning"
    elif 9 <= h < 12:
        phrase, label = "morning — peak analytical hours", "morning"
    elif 12 <= h < 14:
        phrase, label = "midday — post-lunch dip possible", "midday"
    elif 14 <= h < 18:
        phrase, label = "afternoon — sustained work period", "afternoon"
    elif 18 <= h < 22:
        phrase, label = "evening — winding down", "evening"
    elif 22 <= h < 24 or 0 <= h < 2:
        phrase, label = "late night — deep focus / revision prone", "late night"
    else:
        phrase, label = "very late night — low energy", "wee hours"
    return phrase, label


def _flatten(chat_history: list[dict]) -> str:
    """Flatten a chat history into a compact transcript string."""
    lines = []
    for m in chat_history or []:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _latest_user_message(chat_history: list[dict]) -> str:
    """Return the most recent user message (best signal of *current* intent).

    Falls back to the full transcript if no user message is found.
    """
    for m in reversed(chat_history or []):
        if m.get("role") == "user" and (m.get("content") or "").strip():
            return m["content"].strip()
    return _flatten(chat_history)


def _parse_routing(raw: str) -> tuple[str, str]:
    """Parse an LLM routing response into (mode, reasoning). Raises on failure."""
    raw = (raw or "").strip()
    # Strip accidental code fences.
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            mode = str(data.get("mode", "")).strip()
            reasoning = str(data.get("reasoning", "")).strip()
            mode_norm = mode.capitalize()
            if mode_norm in VALID_MODES:
                return mode_norm, reasoning or f"LLM routed to {mode_norm}."
            # case-insensitive rescue
            low = mode.lower()
            for vm in VALID_MODES:
                if vm.lower() in low:
                    return vm, reasoning or f"LLM routed to {vm}."
    except (ValueError, TypeError):
        pass

    # Last resort: scan the raw text for an explicit mode word.
    m = re.search(r"\b(Builder|Exam|Personal)\b", raw, re.IGNORECASE)
    if m:
        return m.group(1).capitalize(), raw[:200]
    raise ValueError("Unparseable routing response")


def _keyword_fallback(text: str, time_phrase: str, time_label: str) -> tuple[str, str]:
    """Deterministic keyword + temporal routing used when the LLM is unavailable."""
    low = (text or "").lower()
    scores = {
        "Builder": sum(low.count(k) for k in BUILDER_KEYWORDS),
        "Exam": sum(low.count(k) for k in EXAM_KEYWORDS),
        "Personal": sum(low.count(k) for k in PERSONAL_KEYWORDS) + 0.4,
    }

    notes = []
    # Temporal nudges.
    if time_label in {"late night", "wee hours"} and scores["Exam"] > 0:
        scores["Exam"] += 1.5
        notes.append("late-night academic activity detected")
    if time_label == "morning" and scores["Builder"] > 0:
        scores["Builder"] += 0.5
        notes.append("morning coding focus favoured")
    if time_label == "evening" and scores["Personal"] >= scores["Builder"]:
        scores["Personal"] += 0.5
        notes.append("evening leans personal")

    best = max(scores, key=scores.get)
    if max(scores.values()) <= 0.4:
        best = "Personal"
        notes.append("no strong signals — defaulting to personal")

    sig = (
        f"Keyword scan matched {best} (Builder:{scores['Builder']}, "
        f"Exam:{scores['Exam']}, Personal:{scores['Personal']})."
    )
    reasoning = sig
    if notes:
        reasoning += " " + "; ".join(notes) + f". Time context: {time_phrase}."
    return best, reasoning


def detect_context(chat_history: list[dict], time_of_day: str | None = None) -> dict:
    """Route the conversation to a focus state.

    Tries the LLM first; on ANY exception falls back to the keyword scanner.
    Returns a dict: ``{mode, reasoning, source, time_of_day, time_phrase, error?}``.
    """
    transcript = _flatten(chat_history)
    latest = _latest_user_message(chat_history)
    now = datetime.now()
    time_phrase, time_label = _time_context(now)
    tod = time_of_day or time_label

    # --- Primary path: LLM routing ----------------------------------------
    try:
        prompt = prompts.CONTEXT_ROUTING_PROMPT.format(
            history=utils.truncate(transcript, 3000) or "(empty conversation)",
            time_of_day=time_phrase,
        )
        raw = call_llm(prompt, temperature=0.1, max_tokens=200)
        mode, reasoning = _parse_routing(raw)
        if mode not in VALID_MODES:
            raise ValueError(f"invalid mode: {mode!r}")
        return {
            "mode": mode,
            "reasoning": reasoning,
            "source": "LLM",
            "time_of_day": tod,
            "time_phrase": time_phrase,
        }
    except Exception as exc:  # noqa: BLE001 — broad on purpose (fail-safe)
        # --- Fail-safe path: keyword scanner on the LATEST user message ---
        # (Scoring the full history would let an old Builder turn dominate
        # forever; the latest message is the truest signal of current intent.)
        mode, reasoning = _keyword_fallback(latest, time_phrase, time_label)
        return {
            "mode": mode,
            "reasoning": reasoning,
            "source": "FALLBACK",
            "time_of_day": tod,
            "time_phrase": time_phrase,
            "error": str(exc),
        }
