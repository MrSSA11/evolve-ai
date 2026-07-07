"""Rich dashboard components for Evolve.

All renderers emit inline-styled HTML (no external assets) so they display
correctly in the sandboxed preview and in a real browser. The analytics matrix
uses a standard CSS Flexbox with ``flex: 1 1 240px`` so cards sit side-by-side
on desktop and stack automatically on mobile.
"""
from __future__ import annotations

import os
import streamlit as st

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


# --------------------------------------------------------------------------- #
# Header & status banner
# --------------------------------------------------------------------------- #
def render_header() -> None:
    st.markdown(
        """
        <div class="ev-card" style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
          <div style="font-size:2.1rem;line-height:1;">🧬</div>
          <div style="flex:1 1 320px;">
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
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(routing: dict) -> None:
    """Top-of-chat banner showing the routed focus mode + reasoning."""
    mode = routing.get("mode", "Personal")
    color = _mode_color(mode)
    reasoning = routing.get("reasoning", "") or "No routing detail available."
    source = routing.get("source", "LLM")
    tod = routing.get("time_of_day", "")
    time_phrase = routing.get("time_phrase", "")

    source_color = "#238636" if str(source).upper() == "LLM" else "#D29922"
    source_label = "LLM routing" if str(source).upper() == "LLM" else "Keyword fail-safe"

    st.markdown(
        f"""
        <div class="ev-banner">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <span class="ev-pill" style="background:{color}26;border-color:{color};color:#fff;">
              🎯 {mode} Focus
            </span>
            <span class="ev-pill" style="border-color:{source_color};color:{source_color};">
              ⚙ {source_label}
            </span>
            {f'<span class="ev-pill ev-muted">🕒 {tod}</span>' if tod else ''}
          </div>
          <div class="ev-muted" style="margin-top:8px;font-size:0.86rem;line-height:1.45;">
            {reasoning}
          </div>
          {f'<div class="ev-muted" style="margin-top:4px;font-size:0.72rem;">Temporal context: {time_phrase}</div>' if time_phrase else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Analytics matrix (responsive flexbox)
# --------------------------------------------------------------------------- #
def render_analytics_cards(stats: dict, session_state=None) -> None:
    """Responsive row of KPI cards using ``flex: 1 1 240px``."""
    current = "Personal"
    friction = 0.0
    switches = 0
    if session_state is not None:
        current = session_state.get("current_mode", "Personal")
        pattern = session_state.get("last_pattern") or {}
        metrics = pattern.get("metrics", {}) or {}
        friction = float(metrics.get("friction_score", 0.0) or 0.0)
        switches = int(metrics.get("switches", 0) or 0)

    cards = [
        ("🧠 Memory Frames", str(stats.get("frames", 0)), "#2F81F7"),
        ("🧩 Knowledge Clusters", str(stats.get("clusters", 0)), "#A371F7"),
        ("🎯 Active Focus", current, _mode_color(current)),
        ("⚡ Friction Score", f"{friction:.2f}", "#D29922"),
        ("🔀 Mode Switches", str(switches), "#F85149"),
    ]

    cells = "".join(
        f"""
        <div class="ev-stat ev-cell">
          <div class="ev-stat-value" style="color:{color};">{value}</div>
          <div class="ev-stat-label">{label}</div>
        </div>
        """
        for label, value, color in cards
    )
    st.markdown(f'<div class="ev-grid">{cells}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cited sources
# --------------------------------------------------------------------------- #
def render_memory_sources(recall_ctx: dict | None, citations: list[int] | None) -> None:
    """Show recalled frames, highlighting the ones the assistant actually cited."""
    if not recall_ctx:
        st.markdown(
            '<div class="ev-muted" style="font-size:0.85rem;">No active recall yet — '
            "send a message to pull memory frames.</div>",
            unsafe_allow_html=True,
        )
        return
    frames = recall_ctx.get("frames", []) or []
    if not frames:
        st.markdown(
            '<div class="ev-muted" style="font-size:0.85rem;">No frames recalled for the last query.</div>',
            unsafe_allow_html=True,
        )
        return

    cited = set(citations or [])
    rows = []
    for f in frames:
        fid = f.get("frame_id")
        is_cited = fid in cited
        color = "#238636" if is_cited else "#8A99AD"
        tag = "✓ cited in reply" if is_cited else f"relevance {f.get('score', 0):.2f}"
        border = "border-color:#23863666;background:rgba(35,134,54,0.05);" if is_cited else ""
        rows.append(
            f"""
            <div class="ev-card" style="padding:8px 10px;margin-bottom:6px;{border}">
              <div style="display:flex;justify-content:space-between;gap:6px;align-items:center;flex-wrap:wrap;">
                <span class="ev-pill" style="background:{color}22;border-color:{color};">[Frame #{fid}]</span>
                <span class="ev-muted" style="font-size:0.7rem;">{tag}</span>
              </div>
              <div style="font-size:0.8rem;margin-top:4px;color:var(--ev-text);">
                {utils.truncate(f.get('content', ''), 120)}
              </div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Memory timeline
# --------------------------------------------------------------------------- #
def render_memory_timeline(frames: list[dict], limit: int = 14) -> None:
    """Vertical timeline of the most recent Memory Frames."""
    if not frames:
        st.markdown(
            '<div class="ev-muted" style="font-size:0.85rem;">No memory frames yet.</div>',
            unsafe_allow_html=True,
        )
        return

    recent = frames[-limit:][::-1]
    items = []
    for f in recent:
        color = _mode_color(f.get("mode", "Personal"))
        cite = f"[Frame #{f.get('frame_id')}]"
        items.append(
            f"""
            <div class="ev-timeline-item">
              <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                <span class="ev-pill" style="background:{color}22;border-color:{color};">{cite}</span>
                <span class="ev-pill ev-muted">{f.get('mode', '')}</span>
                <span class="ev-muted" style="font-size:0.72rem;">
                  {utils.relative_day_phrase(f.get('timestamp'))}
                </span>
              </div>
              <div style="margin-top:4px;font-size:0.84rem;color:var(--ev-text);">
                {utils.truncate(f.get('content', ''), 160)}
              </div>
            </div>
            """
        )
    st.markdown('<div style="margin-top:4px;">' + "".join(items) + "</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Knowledge clusters
# --------------------------------------------------------------------------- #
def render_knowledge_clusters(clusters: list[dict]) -> None:
    """Render consolidated Knowledge Cluster nodes."""
    if not clusters:
        st.markdown(
            '<div class="ev-muted" style="font-size:0.85rem;">No clusters yet — '
            "run <b>Improve</b> in the sidebar to synthesize.</div>",
            unsafe_allow_html=True,
        )
        return

    for cl in clusters:
        color = _mode_color(cl.get("mode", ""))
        kw = " · ".join(cl.get("keywords", [])[:6])
        src = ", ".join(f"#{i}" for i in cl.get("source_frames", [])[:8])
        st.markdown(
            f"""
            <div class="ev-card">
              <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                <span class="ev-pill" style="background:{color}22;border-color:{color};">
                  🧩 [Cluster #{cl.get('cluster_id')}]
                </span>
                <span class="ev-pill ev-muted">{cl.get('mode', '')}</span>
                <span class="ev-pill ev-muted">{cl.get('frame_count', 0)} frames</span>
              </div>
              <div style="margin-top:8px;font-size:0.88rem;line-height:1.45;color:var(--ev-text);">
                {utils.truncate(cl.get('summary', ''), 320)}
              </div>
              <div class="ev-muted" style="margin-top:6px;font-size:0.72rem;">
                keywords: {kw or '—'} · sources: {src or '—'}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Graphviz lifecycle graph
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
# Code workspace tooling
# --------------------------------------------------------------------------- #
def render_code_workspace(blocks: list[dict]) -> None:
    """Render an execution widget + '💾 Save Snippet' for each code block."""
    if not blocks:
        st.markdown(
            '<div class="ev-muted" style="font-size:0.85rem;">'
            "No code blocks in the last response.</div>",
            unsafe_allow_html=True,
        )
        return

    for b in blocks:
        lang = b.get("language", "text")
        code = b.get("code", "")
        idx = b.get("index", 0)
        st.markdown(
            f'<div class="ev-pill ev-muted" style="margin-bottom:4px;">📄 {lang}</div>',
            unsafe_allow_html=True,
        )
        st.code(code, language=lang)
        if st.button("💾 Save Snippet", key=f"save_snippet_{idx}", use_container_width=False):
            try:
                path = utils.save_snippet(code, lang, idx)
                st.toast(f"Saved → {os.path.basename(path)}", icon="💾")
            except Exception as exc:
                st.error(f"Could not save snippet: {exc}")

    files = utils.list_workspace_files()
    if files:
        st.markdown(
            '<div class="ev-muted" style="font-size:0.74rem;margin-top:8px;">'
            "Workspace files:</div>",
            unsafe_allow_html=True,
        )
        for fl in files[-12:][::-1]:
            st.markdown(
                f'<div class="ev-muted" style="font-size:0.74rem;">'
                f"• {fl['name']} <span style='opacity:0.6;'>({fl['size']} B)</span></div>",
                unsafe_allow_html=True,
            )
