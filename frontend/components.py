"""Rich dashboard components for Evolve.

Rendering strategy (important):
- Presentational HTML blocks (header, status banner, memory timeline, knowledge
  clusters, cited sources, pattern card) are rendered via
  ``streamlit.components.v1.html``. This injects a real iframe and **bypasses
  Streamlit's markdown HTML sanitizer**, so the markup always renders correctly
  (instead of leaking as raw ``<div>`` text, which happens with
  ``st.markdown(unsafe_allow_html=True)`` in some environments / Streamlit
  versions).
- The analytics KPI matrix uses native ``st.columns`` + ``st.metric`` so cards
  sit side-by-side on desktop and stack natively on mobile (matching the
  ``flex: 1 1 240px`` requirement without fragile CSS overrides on viewport
  divs).
- Interactive pieces that need Streamlit state (the Graphviz chart, the
  Save-Snippet buttons + code blocks) stay native.
- Simple messages use ``st.caption`` — **zero** ``unsafe_allow_html`` usage for
  content, so nothing can leak as raw text.

Every HTML block carries its own inline ``<style>`` (iframes are isolated from
the app's injected theme CSS), so styling is self-contained and consistent.
"""
from __future__ import annotations

import math
import os
import streamlit as st
import streamlit.components.v1 as components

from backend import utils

# Mode -> accent colour.
MODE_COLORS = {
    "Builder": "#2F81F7",
    "Exam": "#A371F7",
    "Personal": "#238636",
}


def _mode_color(mode: str) -> str:
    return MODE_COLORS.get(mode, "#8A99AD")


def _dot_escape(s: str) -> str:
    """Escape a string for safe embedding inside a DOT label."""
    return (s or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _html_escape(s: str) -> str:
    """Escape a string for safe embedding inside HTML text."""
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# Self-contained stylesheet for every component iframe.
COMPONENT_CSS = """
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0; background: #0E1117;
  color: #E6EDF3;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 14px; line-height: 1.45;
}
.ev-muted { color: #8A99AD; }
b, strong { color: #E6EDF3; font-weight: 700; }
.ev-card {
  background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px; padding: 14px 16px;
}
.ev-pill {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  font-size: 12px; font-weight: 600; letter-spacing: 0.02em;
  border: 1px solid rgba(255,255,255,0.06); color: #E6EDF3;
  background: rgba(255,255,255,0.02); white-space: nowrap;
}
.ev-timeline-item {
  position: relative; padding: 4px 0 12px 18px;
  border-left: 2px solid rgba(255,255,255,0.06); margin-left: 6px;
}
.ev-timeline-item:last-child { padding-bottom: 2px; }
.ev-timeline-item::before {
  content: ""; position: absolute; left: -6px; top: 9px; width: 10px; height: 10px;
  border-radius: 50%; background: #2F81F7;
  box-shadow: 0 0 0 3px rgba(47,129,247,0.15);
}
"""


def render_html(inner_html: str, height: int) -> None:
    """Render an HTML block inside a guaranteed-rendering iframe.

    Uses ``streamlit.components.v1.html`` which bypasses the markdown sanitizer.
    """
    height = max(int(height or 40), 24)
    doc = f"<style>{COMPONENT_CSS}</style>{inner_html}"
    components.html(doc, height=height, scrolling=False)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
def render_header() -> None:
    inner = """
    <div class="ev-card" style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
      <div style="font-size:2.1rem;line-height:1;">🧬</div>
      <div style="flex:1 1 320px;min-width:0;">
        <div style="font-size:1.55rem;font-weight:800;letter-spacing:-0.02em;">Evolve</div>
        <div class="ev-muted" style="font-size:0.86rem;margin-top:2px;">
          Adaptive Priority Engine · context-aware consultation powered by a
          vector-graph memory (Cognee-style <b>remember · recall · improve · forget</b>)
        </div>
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;">
        <span class="ev-pill" style="border-color:#2F81F7;color:#2F81F7;">Builder</span>
        <span class="ev-pill" style="border-color:#A371F7;color:#A371F7;">Exam</span>
        <span class="ev-pill" style="border-color:#238636;color:#238636;">Personal</span>
      </div>
    </div>
    """
    render_html(inner, height=84)


# --------------------------------------------------------------------------- #
# Status banner
# --------------------------------------------------------------------------- #
def render_status_banner(routing: dict) -> None:
    """Top-of-chat banner showing the routed focus mode + reasoning."""
    mode = routing.get("mode", "Personal")
    color = _mode_color(mode)
    reasoning = _html_escape(routing.get("reasoning", "") or "No routing detail available.")
    source = str(routing.get("source", "LLM")).upper()
    tod = _html_escape(routing.get("time_of_day", "") or "")
    time_phrase = _html_escape(routing.get("time_phrase", "") or "")

    source_color = "#238636" if source == "LLM" else "#D29922"
    source_label = "LLM routing" if source == "LLM" else "Keyword fail-safe"

    tod_pill = (
        f'<span class="ev-pill ev-muted">🕒 {_html_escape(tod)}</span>' if tod else ""
    )
    time_line = (
        f'<div class="ev-muted" style="margin-top:4px;font-size:0.72rem;">Temporal context: {time_phrase}</div>'
        if time_phrase else ""
    )

    inner = f"""
    <div class="ev-card">
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        <span class="ev-pill" style="background:{color}26;border-color:{color};color:#fff;">🎯 {mode} Focus</span>
        <span class="ev-pill" style="border-color:{source_color};color:{source_color};">⚙ {source_label}</span>
        {tod_pill}
      </div>
      <div class="ev-muted" style="margin-top:8px;font-size:0.86rem;">{reasoning}</div>
      {time_line}
    </div>
    """
    # Estimate height from reasoning length (~64 chars/line at dashboard width).
    lines = max(1, math.ceil(len(reasoning) / 64))
    height = 40 + 28 + lines * 20 + (16 if time_line else 0) + 12
    render_html(inner, height=height)


# --------------------------------------------------------------------------- #
# Pattern card (Hidden Pattern Detected + Root Cause)
# --------------------------------------------------------------------------- #
def render_pattern_card(pattern: dict) -> None:
    """Render the hidden-pattern detection card via an iframe (no raw HTML)."""
    metrics = pattern.get("metrics", {}) or {}
    friction = float(metrics.get("friction_score", 0.0) or 0.0)
    root_cause = _html_escape(pattern.get("root_cause", "") or "")
    recommendation = _html_escape(pattern.get("recommendation", "") or "")
    switches = int(metrics.get("switches", 0) or 0)

    inner = f"""
    <div class="ev-card" style="border-color:#D2992255;background:rgba(210,153,34,0.05);">
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <span class="ev-pill" style="background:#D2992222;border-color:#D29922;color:#D29922;">
          ⚡ Hidden Pattern Detected
        </span>
        <span class="ev-pill ev-muted">Friction {friction:.2f}</span>
        <span class="ev-pill ev-muted">{switches} switches</span>
      </div>
      <div style="margin-top:8px;">
        <b>Root Cause:</b>
        <span>{root_cause}</span>
      </div>
      <div class="ev-muted" style="margin-top:4px;font-size:0.85rem;">
        💡 {recommendation}
      </div>
    </div>
    """
    # Height: header row (~34) + root cause lines + recommendation lines + padding
    rc_lines = max(1, math.ceil(len(root_cause) / 70))
    rec_lines = max(1, math.ceil(len(recommendation) / 70))
    height = 34 + rc_lines * 20 + rec_lines * 20 + 40
    render_html(inner, height=height)


# --------------------------------------------------------------------------- #
# Analytics matrix (native Streamlit columns + metrics — responsive & robust)
# --------------------------------------------------------------------------- #
def render_analytics_cards(stats: dict, session_state=None) -> None:
    """Responsive KPI row.

    Uses native ``st.columns`` so cards split side-by-side on desktop and stack
    automatically on mobile — the same result as the requested
    ``flex: 1 1 240px`` matrix, but without fragile CSS overrides and without
    relying on ``unsafe_allow_html`` rendering.
    """
    current = "Personal"
    friction = 0.0
    switches = 0
    if session_state is not None:
        try:
            current = session_state.get("current_mode", "Personal")
        except Exception:
            current = "Personal"
        try:
            pattern = session_state.get("last_pattern") or {}
            metrics = pattern.get("metrics", {}) or {}
            friction = float(metrics.get("friction_score", 0.0) or 0.0)
            switches = int(metrics.get("switches", 0) or 0)
        except Exception:
            pass

    cards = [
        ("Memory Frames", f"🧠 {stats.get('frames', 0)}"),
        ("Knowledge Clusters", f"🧩 {stats.get('clusters', 0)}"),
        ("Active Focus", f"🎯 {current}"),
        ("Friction Score", f"⚡ {friction:.2f}"),
        ("Mode Switches", f"🔀 {switches}"),
    ]

    cols = st.columns(len(cards))
    for col, (label, value) in zip(cols, cards):
        with col:
            st.metric(label=label, value=value)


# --------------------------------------------------------------------------- #
# Cited sources
# --------------------------------------------------------------------------- #
def render_memory_sources(recall_ctx: dict | None, citations: list[int] | None) -> None:
    """Show recalled frames, highlighting the ones the assistant actually cited."""
    if not recall_ctx:
        st.caption("No active recall yet — send a message to pull memory frames.")
        return
    frames = recall_ctx.get("frames", []) or []
    if not frames:
        st.caption("No frames recalled for the last query.")
        return

    cited = set(citations or [])
    rows = []
    for f in frames:
        fid = f.get("frame_id")
        is_cited = fid in cited
        color = "#238636" if is_cited else "#8A99AD"
        tag = "✓ cited in reply" if is_cited else f"relevance {f.get('score', 0):.2f}"
        border = "border-color:#23863655;background:rgba(35,134,54,0.05);" if is_cited else ""
        content = _html_escape(utils.truncate(f.get("content", ""), 120))
        rows.append(
            f"""
            <div class="ev-card" style="padding:8px 10px;margin-bottom:6px;{border}">
              <div style="display:flex;justify-content:space-between;gap:6px;align-items:center;flex-wrap:wrap;">
                <span class="ev-pill" style="background:{color}22;border-color:{color};">[Frame #{fid}]</span>
                <span class="ev-muted" style="font-size:0.7rem;">{tag}</span>
              </div>
              <div style="font-size:0.8rem;margin-top:4px;">{content}</div>
            </div>
            """
        )
    inner = "".join(rows)
    height = len(frames) * 74 + 8
    render_html(inner, height=height)


# --------------------------------------------------------------------------- #
# Memory timeline
# --------------------------------------------------------------------------- #
def render_memory_timeline(frames: list[dict], limit: int = 14) -> None:
    """Vertical timeline of the most recent Memory Frames."""
    if not frames:
        st.caption("No memory frames yet.")
        return

    recent = frames[-limit:][::-1]
    items = []
    for f in recent:
        color = _mode_color(f.get("mode", "Personal"))
        cite = f"[Frame #{f.get('frame_id')}]"
        rel = _html_escape(utils.relative_day_phrase(f.get("timestamp")))
        content = _html_escape(utils.truncate(f.get("content", ""), 160))
        items.append(
            f"""
            <div class="ev-timeline-item">
              <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                <span class="ev-pill" style="background:{color}22;border-color:{color};">{cite}</span>
                <span class="ev-pill ev-muted">{_html_escape(f.get('mode', ''))}</span>
                <span class="ev-muted" style="font-size:0.72rem;">{rel}</span>
              </div>
              <div style="margin-top:4px;font-size:0.84rem;">{content}</div>
            </div>
            """
        )
    inner = '<div style="margin-top:2px;">' + "".join(items) + "</div>"
    height = len(recent) * 62 + 8
    render_html(inner, height=height)


# --------------------------------------------------------------------------- #
# Knowledge clusters
# --------------------------------------------------------------------------- #
def render_knowledge_clusters(clusters: list[dict]) -> None:
    """Render consolidated Knowledge Cluster nodes."""
    if not clusters:
        st.caption("No clusters yet — run **Improve** in the sidebar to synthesize.")
        return

    blocks = []
    for cl in clusters:
        color = _mode_color(cl.get("mode", ""))
        kw = _html_escape(" · ".join(cl.get("keywords", [])[:6]))
        src = _html_escape(", ".join(f"#{i}" for i in cl.get("source_frames", [])[:8]))
        summary = _html_escape(utils.truncate(cl.get("summary", ""), 320))
        blocks.append(
            f"""
            <div class="ev-card" style="margin-bottom:8px;">
              <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                <span class="ev-pill" style="background:{color}22;border-color:{color};">🧩 [Cluster #{cl.get('cluster_id')}]</span>
                <span class="ev-pill ev-muted">{_html_escape(cl.get('mode', ''))}</span>
                <span class="ev-pill ev-muted">{cl.get('frame_count', 0)} frames</span>
              </div>
              <div style="margin-top:8px;font-size:0.88rem;line-height:1.45;">{summary}</div>
              <div class="ev-muted" style="margin-top:6px;font-size:0.72rem;">keywords: {kw or '—'} · sources: {src or '—'}</div>
            </div>
            """
        )
    inner = "".join(blocks)
    height = len(clusters) * 150 + 8
    render_html(inner, height=height)


# --------------------------------------------------------------------------- #
# Graphviz lifecycle graph (native Streamlit widget)
# --------------------------------------------------------------------------- #
def render_lifecycle_graph(current_mode: str, frames: list[dict],
                           clusters: list[dict] | None = None) -> None:
    """Visual lifecycle: User -> Active Focus Mode -> Memory Frame / Cluster nodes."""
    mode_color = _mode_color(current_mode)

    lines = [
        "digraph Evolve {",
        "  rankdir=LR;",
        '  graph [bgcolor="#0E1117", fontname="Helvetica"];',
        '  node [fontname="Helvetica", fontsize=10, style="filled", penwidth=0, '
        'shape=box, margin="0.14,0.08"];',
        '  edge [color="#8A99AD", penwidth=1.3, arrowsize=0.7];',
        '  "user" [label="👤 User", fillcolor="#1F6FEB", fontcolor="white", penwidth=1];',
        f'  "mode" [label="🎯 {current_mode} Mode", fillcolor="{mode_color}", '
        f'fontcolor="white", penwidth=1];',
        '  "user" -> "mode" [penwidth=1.6, color="#E6EDF3"];',
    ]

    shown = frames[-6:][::-1] if frames else []
    for f in shown:
        fid = f.get("frame_id")
        snippet = _dot_escape(utils.truncate(f.get("content", ""), 26))
        c = _mode_color(f.get("mode", ""))
        lines.append(
            f'  "f{fid}" [label="Frame #{fid}\\n{snippet}", fillcolor="{c}33", '
            f'fontcolor="#E6EDF3", color="{c}"];'
        )
        lines.append(f'  "mode" -> "f{fid}";')

    if clusters:
        for cl in clusters[:3]:
            cid = cl.get("cluster_id")
            snippet = _dot_escape(utils.truncate(cl.get("summary", ""), 22))
            lines.append(
                f'  "c{cid}" [label="Cluster #{cid}\\n{snippet}", '
                f'fillcolor="#A371F733", fontcolor="#E6EDF3", shape=ellipse, color="#A371F7"];'
            )
            lines.append(f'  "mode" -> "c{cid}" [style=dashed, color="#A371F7"];')

    lines.append("}")
    dot = "\n".join(lines)

    try:
        st.graphviz_chart(dot, use_container_width=True)
    except Exception as exc:  # graceful fallback if the renderer is unavailable
        st.code(dot, language="dot")
        st.caption(f"Graphviz render unavailable — showing DOT source ({exc}).")


# --------------------------------------------------------------------------- #
# Code workspace tooling (native buttons + code blocks)
# --------------------------------------------------------------------------- #
def render_code_workspace(blocks: list[dict]) -> None:
    """Render an execution widget + '💾 Save Snippet' for each code block."""
    if not blocks:
        st.caption("No code blocks in the last response.")
        return

    for b in blocks:
        lang = b.get("language", "text")
        code = b.get("code", "")
        idx = b.get("index", 0)
        st.code(code, language=lang)
        if st.button("💾 Save Snippet", key=f"save_snippet_{idx}", use_container_width=False):
            try:
                path = utils.save_snippet(code, lang, idx)
                st.toast(f"Saved → {os.path.basename(path)}", icon="💾")
            except Exception as exc:
                st.error(f"Could not save snippet: {exc}")

    files = utils.list_workspace_files()
    if files:
        st.caption("Workspace files:")
        for fl in files[-12:][::-1]:
            st.caption(f"• {fl['name']} ({fl['size']} B)")
