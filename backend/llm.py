"""LLM integration layer for Evolve: Adaptive Priority Engine.

Responsibilities:
- Build a lazy, fault-tolerant Google **Gen AI** client (the native SDK for
  Google's Gemma / Gemini models — *not* an OpenAI client).
- Generate chat responses with a **dynamic persona** per focus mode.
- Inject recalled memory into the system instruction and parse the citations
  the model emits back, proving context awareness.
- Degrade gracefully to a deterministic offline mock when no API key / network
  / SDK is available so the application is always demonstrable.
- **Surface the real API error** (instead of silently mocking) via
  :func:`get_last_error`, so live-but-failing setups are diagnosable.

Environment variables:
- ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` — API key (required for live calls).
  If both are set, ``GOOGLE_API_KEY`` takes precedence (matching the SDK).
- ``EVOLVE_MODEL``                       — model id (default ``gemma-4-31b-it``).
"""
from __future__ import annotations

import os
import re
from typing import Any

from . import prompts, utils


# --------------------------------------------------------------------------- #
# Diagnostics: last error from a live attempt (None == ok / never tried live)
# --------------------------------------------------------------------------- #
_last_error: str | None = None


def get_last_error() -> str | None:
    """Return the most recent live-call error, or ``None`` if the last call ok."""
    return _last_error


def _clean_err(exc: BaseException) -> str:
    """Compact single-line representation of an exception for the UI."""
    msg = str(exc).replace("\n", " ").strip()
    return msg[:400]


# --------------------------------------------------------------------------- #
# Model + client construction (lazy + fault tolerant)
# --------------------------------------------------------------------------- #
def _model_name() -> str:
    """Return the configured model id (accepts ``models/...`` or bare name)."""
    name = utils.env_or("EVOLVE_MODEL", "gemma-4-31b-it")
    return name.strip()


def _get_client():
    """Return a Google Gen AI client, or ``None`` if one cannot be built.

    Importing ``google.genai`` and constructing the client are both wrapped so
    that a missing dependency or missing key never crashes the application.
    """
    try:
        from google import genai  # type: ignore
    except Exception:
        return None

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None

    try:
        return genai.Client(api_key=key)
    except Exception:
        return None


def is_live() -> bool:
    """True when a live LLM client *could* be built (key + SDK present).

    Note: this only proves the client constructed — it does NOT prove the API
    will accept calls. Use :func:`get_last_error` to see call-level failures.
    """
    return _get_client() is not None


# --------------------------------------------------------------------------- #
# Message mapping for the Gen AI SDK (assistant -> "model" role)
# --------------------------------------------------------------------------- #
def _to_contents(messages: list[dict]) -> list:
    """Convert our chat messages into SDK ``Content`` objects.

    ``system`` messages are excluded here (they are passed separately as the
    ``system_instruction`` config field); ``assistant`` maps to ``model``.
    Consecutive same-role turns are merged so the request always alternates
    user/model (required by Gemma).
    """
    from google.genai import types  # only reached when genai is importable

    contents = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content or role == "system":
            continue
        grole = "model" if role == "assistant" else "user"
        contents.append(types.Content(role=grole, parts=[types.Part(text=content)]))

    # Merge any consecutive same-role turns into one Content.
    merged: list = []
    for c in contents:
        if merged and merged[-1].role == c.role:
            merged[-1].parts.extend(c.parts)
        else:
            merged.append(c)

    if not merged:
        merged = [types.Content(role="user", parts=[types.Part(text="(no input)")])]
    return merged


def _system_text(messages: list[dict]) -> str:
    """Concatenate every system message into one instruction block."""
    return "\n\n".join(
        (m.get("content") or "").strip()
        for m in messages
        if m.get("role") == "system"
    ).strip()


def _fold_system(contents: list, system_text: str) -> list:
    """Merge the system instruction into the FIRST user turn.

    Used as a fallback for models/endpoints that reject ``system_instruction``.
    Folding (rather than prepending a new turn) preserves strict user/model
    alternation, which Gemma requires.
    """
    if not system_text or not contents:
        return list(contents)
    try:
        from google.genai import types
    except Exception:
        return list(contents)
    first = contents[0]
    existing = "".join(getattr(p, "text", "") or "" for p in first.parts)
    merged_text = (
        f"{system_text}\n\n---\n\n(Now respond to the following user message.)\n\n{existing}"
    )
    return [types.Content(role=first.role, parts=[types.Part(text=merged_text)])] + list(contents[1:])


def _build_config(temperature: float, max_tokens: int, system_text: str,
                  use_system: bool) -> Any:
    """Build a ``GenerateContentConfig``.

    Gemma models on the Gemini API require ``top_k`` to be set, so we add a
    sensible ``top_k`` whenever the active model is a Gemma variant.
    """
    from google.genai import types  # only reached when genai is importable

    kwargs: dict[str, Any] = {
        "temperature": float(temperature),
        "max_output_tokens": int(max_tokens),
    }
    if "gemma" in _model_name().lower():
        kwargs["top_k"] = 40
    if use_system and system_text:
        kwargs["system_instruction"] = system_text
    return types.GenerateContentConfig(**kwargs)


def _finish_reason(resp) -> str:
    """Best-effort extraction of a response's finish_reason for diagnostics."""
    try:
        cands = getattr(resp, "candidates", None)
        if cands:
            return str(getattr(cands[0], "finish_reason", "UNKNOWN"))
    except Exception:
        pass
    return "UNKNOWN"


def _extract_text(resp) -> str:
    """Pull the text out of a Gen AI response, tolerating blocked/empty output."""
    try:
        text = getattr(resp, "text", None)
        if text:
            return text.strip()
    except Exception:
        pass
    try:
        parts = resp.candidates[0].content.parts
        return "".join(getattr(p, "text", "") or "" for p in parts).strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Low-level completion
# --------------------------------------------------------------------------- #
def _complete(messages: list[dict], temperature: float = 0.7, max_tokens: int = 800) -> str:
    """Run a Gen AI ``generate_content`` call, falling back to the offline mock.

    Any error from a *live* attempt is captured into ``_last_error`` so the UI
    can surface it (rather than silently swapping in the mock). Two attempts are
    made before giving up:
      1. native ``system_instruction`` config field (Gemma 4 supports it);
      2. the system text folded into the first user turn (covers any model that
         does not accept ``system_instruction``).
    """
    global _last_error

    client = _get_client()
    if client is None:
        _last_error = (
            "No API key or SDK. Set GEMINI_API_KEY (or GOOGLE_API_KEY) and "
            "install with `pip install google-genai`."
        )
        return _mock_completion(messages, reason="offline")

    try:
        system_text = _system_text(messages)
        contents = _to_contents(messages)
    except Exception as exc:
        _last_error = f"Failed to build request contents: {type(exc).__name__}: {_clean_err(exc)}"
        return _mock_completion(messages, reason="error")

    model = _model_name()
    errors: list[str] = []

    # Attempt 1: native system_instruction (Gemma 4 supports it natively).
    try:
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=_build_config(temperature, max_tokens, system_text, use_system=True),
        )
        text = _extract_text(resp)
        if text:
            _last_error = None
            return text
        errors.append(f"empty response (finish_reason={_finish_reason(resp)})")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {_clean_err(exc)}")

    # Attempt 2: fold system instruction into the first user turn.
    try:
        folded = _fold_system(contents, system_text)
        resp = client.models.generate_content(
            model=model,
            contents=folded,
            config=_build_config(temperature, max_tokens, "", use_system=False),
        )
        text = _extract_text(resp)
        if text:
            _last_error = None
            return text
        errors.append(f"empty response (finish_reason={_finish_reason(resp)})")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {_clean_err(exc)}")

    _last_error = "  ||  ".join(errors)
    return _mock_completion(messages, reason="error")


def call_llm(prompt: str, system: str | None = None,
             temperature: float = 0.7, max_tokens: int = 400) -> str:
    """Convenience single-turn call used by the routing & analysis modules."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _complete(messages, temperature=temperature, max_tokens=max_tokens)


# --------------------------------------------------------------------------- #
# Dynamic persona assembly
# --------------------------------------------------------------------------- #
def build_persona(mode: str, extra_rules: list[str] | None = None) -> str:
    """Assemble the system prompt: persona + citation policy + injected rules.

    ``extra_rules`` are the behavioral adjustments appended by the Autonomous
    Resolution Engine (the "Auto-Adjustment Protocol").
    """
    parts = [prompts.persona_for(mode), "", prompts.CITATION_INSTRUCTION]
    if extra_rules:
        parts.append("")
        parts.append("ADDITIONAL BEHAVIORAL ADJUSTMENTS (auto-applied):")
        for rule in extra_rules:
            rule = (rule or "").strip()
            if rule:
                parts.append(f"- {rule}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Memory cluster summariser (used by memory.synthesize_memory via lazy import)
# --------------------------------------------------------------------------- #
def summarize_cluster(contents: list[str], mode: str) -> str:
    """Produce a dense Knowledge Cluster summary from raw frame contents.

    Returns an empty string when no live LLM is available so the caller
    (:func:`backend.memory.MemoryStore._summarize`) can fall back to a
    deterministic heuristic summary instead of mock text.
    """
    if not contents:
        return ""
    if not is_live():
        return ""
    joined = "\n".join(f"- {utils.truncate(c, 200)}" for c in contents[:14])
    prompt = prompts.MEMORY_SYNTHESIS_PROMPT.format(
        mode=mode, count=len(contents), frames=joined
    )
    out = (call_llm(prompt, temperature=0.3, max_tokens=180) or "").strip()
    # Reject obvious mock output so the heuristic path can take over.
    if "Offline mock response" in out or "Live API call failed" in out:
        return ""
    return out


# --------------------------------------------------------------------------- #
# Main chat response generator
# --------------------------------------------------------------------------- #
def generate_response(user_input: str,
                      chat_history: list[dict],
                      mode: str,
                      memory_context: dict | None,
                      extra_rules: list[str] | None = None) -> tuple[str, list[int]]:
    """Generate an assistant reply and the list of cited memory ids.

    Returns ``(response_text, citation_ids)``.
    """
    persona = build_persona(mode, extra_rules)

    memory_text = ""
    if memory_context:
        memory_text = memory_context.get("context_text", "") or ""

    system = persona
    if memory_text:
        system += "\n\nRECALLED MEMORY — cite these where relevant:\n" + memory_text

    # Build the conversation. Keep the last few turns to stay token-friendly.
    messages: list[dict] = [{"role": "system", "content": system}]
    recent = chat_history[-8:] if chat_history else []
    for turn in recent:
        role = turn.get("role")
        content = turn.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_input})

    response = _complete(messages, temperature=0.7, max_tokens=900)
    citations = utils.parse_citations(response)
    return response, citations


# --------------------------------------------------------------------------- #
# Offline / fallback mock completion (keeps the app functional without a key)
# --------------------------------------------------------------------------- #
def _detect_mode(system_msg: str) -> str:
    low = system_msg.lower()
    if "builder focus mode" in low or "**builder**" in low:
        return "Builder"
    if "exam focus mode" in low or "**exam**" in low:
        return "Exam"
    return "Personal"


def _mock_completion(messages: list[dict], reason: str = "offline") -> str:
    """Deterministic, context-aware mock used when no live LLM is reachable.

    ``reason`` controls the trailing note so the user understands *why* the
    reply is synthetic: ``"offline"`` (no key/SDK) vs ``"error"`` (live call
    failed — details surfaced separately via :func:`get_last_error`).

    It still honours the citation contract by referencing any [Frame #N]
    markers present in the recalled-memory section of the system prompt.
    """
    system_msg = ""
    user_msg = ""
    for m in messages:
        role = m.get("role")
        if role == "system":
            system_msg += "\n" + (m.get("content", "") or "")
        elif role == "user":
            user_msg = m.get("content", "") or ""

    mode = _detect_mode(system_msg)

    # Only harvest citation markers from the RECALLED MEMORY section, never from
    # the citation-policy instruction (whose examples contain literal markers).
    recalled = ""
    marker = "RECALLED MEMORY"
    idx = system_msg.find(marker)
    if idx != -1:
        recalled = system_msg[idx:]

    frame_ids: list[str] = []
    for fid in re.findall(r"\[Frame #(\d+)\]", recalled):
        if fid not in frame_ids:
            frame_ids.append(fid)
    cluster_ids: list[str] = []
    for cid in re.findall(r"\[Cluster #(\d+)\]", recalled):
        if cid not in cluster_ids:
            cluster_ids.append(cid)

    intro = {
        "Builder": "Here's a focused, production-minded take:",
        "Exam": "Let's structure this for solid revision:",
        "Personal": "Happy to help with that!",
    }[mode]

    body = {
        "Builder": (
            "I'd approach this with clear separation of concerns, guard every "
            "external boundary with error handling, and keep the hot path lean. "
            "Prefer small, testable units and document the trade-offs explicitly. "
            "Want me to sketch the implementation?"
        ),
        "Exam": (
            "Key moves: (1) define the concept crisply, (2) walk through a worked "
            "example step-by-step, then (3) turn it into an active-recall question "
            "so you can self-test. Watch the common edge cases examiners love."
        ),
        "Personal": (
            "Here are a couple of practical, low-friction options that should fit "
            "your situation nicely — pick the one that feels right and we can "
            "refine it together."
        ),
    }[mode]

    cites: list[str] = []
    for fid in frame_ids[:3]:
        cites.append(f"[Frame #{fid}]")
    for cid in cluster_ids[:1]:
        cites.append(f"[Cluster #{cid}]")
    cite_line = ""
    if cites:
        cite_line = "\n\n_Relevant memory: " + ", ".join(cites) + "_"

    # When offline and the user asks for code in Builder mode, emit a small
    # fenced block so the Dynamic Workspace Tooling (Save Snippet) is exercised.
    code_block = ""
    if mode == "Builder" and re.search(
        r"\b(code|snippet|example|implement|function|script|sample|write a|show me|program|boilerplate)\b",
        user_msg.lower(),
    ):
        code_block = (
            "\n\nHere's a minimal starting point you can adapt:\n\n"
            "```python\n"
            "def evolve_handler(payload: dict) -> dict:\n"
            '    """Minimal scaffold — wire your real logic in here."""\n'
            "    if not payload:\n"
            '        raise ValueError("payload must not be empty")\n'
            '    return {"status": "ok", "echo": payload}\n'
            "```\n"
        )

    if reason == "error":
        note = (
            "\n\n_⚠️ Live Gemma call failed — showing a fallback reply. See the "
            "red error banner above for the exact API error._"
        )
    else:
        note = (
            "\n\n_⚠️ Offline mock response — no `GEMINI_API_KEY` / `GOOGLE_API_KEY` "
            "configured (or the `google-genai` package is not installed). Set the key "
            "and install `google-genai` to enable live Gemma output._"
        )

    snippet = utils.truncate(user_msg, 160)
    return f"**{intro}**\n\nRegarding _{snippet}_ — {body}{code_block}{cite_line}{note}"
