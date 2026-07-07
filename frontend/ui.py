"""Page layout, global dark-theme CSS injection, and sidebar hooks.

Implements the Evolve unified dark theme via CSS injection (no external assets
are loaded — the in-app sandboxed preview degrades gracefully) and caps the
main ``block-container`` at 1350px, centered, so widescreen desktop monitors
avoid the left-heavy look while mobile stacks natively.
"""
from __future__ import annotations

import streamlit as st


# --------------------------------------------------------------------------- #
# Unified dark theme stylesheet
# --------------------------------------------------------------------------- #
THEME_CSS = """
:root {
  --ev-bg: #0E1117;
  --ev-card: rgba(255,255,255,0.02);
  --ev-card-strong: rgba(255,255,255,0.035);
  --ev-border: rgba(255,255,255,0.06);
  --ev-border-strong: rgba(255,255,255,0.10);
  --ev-text: #E6EDF3;
  --ev-muted: #8A99AD;
  --ev-accent: #2F81F7;
  --ev-green: #238636;
  --ev-amber: #D29922;
  --ev-purple: #A371F7;
  --ev-red: #F85149;
}

/* ---- Base background ---- */
html, body, [data-testid="stAppViewContainer"], .stApp, .main,
section[data-testid="stMain"], [data-testid="stMain"] {
  background-color: var(--ev-bg) !important;
  color: var(--ev-text) !important;
}

/* ---- Desktop centering: cap width + center, mobile stacks natively ---- */
.block-container,
section[data-testid="stMain"] > div[data-testid="stVerticalBlock"],
div[data-testid="stVerticalBlockBorderWrapper"] > div {
  max-width: 1350px !important;
  margin: 0 auto !important;
  padding: 1.4rem 1.6rem 6.5rem !important;
}

/* ---- Typography ---- */
h1, h2, h3, h4, h5, h6, p, span, li, label {
  color: var(--ev-text) !important;
}
.ev-muted { color: var(--ev-muted) !important; }
a { color: var(--ev-accent) !important; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"],
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
  background-color: #0B0E14 !important;
  border-right: 1px solid var(--ev-border) !important;
}
section[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown { color: var(--ev-text) !important; }

/* ---- Cards & containers ---- */
.ev-card {
  background: var(--ev-card) !important;
  border: 1px solid var(--ev-border) !important;
  border-radius: 12px !important;
  padding: 14px 16px !important;
  margin-bottom: 10px !important;
}

/* ---- Flexbox matrix (analytics cards) ---- */
.ev-grid {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 12px !important;
  width: 100% !important;
}
.ev-grid > .ev-cell {
  flex: 1 1 240px !important;
  min-width: 200px !important;
}
.ev-stat {
  background: var(--ev-card);
  border: 1px solid var(--ev-border);
  border-radius: 12px;
  padding: 14px 16px;
  box-sizing: border-box;
}
.ev-stat .ev-stat-value {
  font-size: 1.85rem; font-weight: 800; line-height: 1.05; color: var(--ev-text);
}
.ev-stat .ev-stat-label {
  font-size: 0.74rem; letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--ev-muted); margin-top: 5px;
}

/* ---- Status banner ---- */
.ev-banner {
  border-radius: 12px; padding: 12px 16px;
  border: 1px solid var(--ev-border); background: var(--ev-card);
}

/* ---- Pills / badges ---- */
.ev-pill {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
  border: 1px solid var(--ev-border); color: var(--ev-text);
  white-space: nowrap;
}

/* ---- Memory timeline ---- */
.ev-timeline-item {
  position: relative; padding: 6px 0 10px 18px;
  border-left: 2px solid var(--ev-border); margin-left: 6px;
}
.ev-timeline-item::before {
  content: ""; position: absolute; left: -6px; top: 9px; width: 10px; height: 10px;
  border-radius: 50%; background: var(--ev-accent);
  box-shadow: 0 0 0 3px rgba(47,129,247,0.15);
}

/* ---- Chat bubbles ---- */
[data-testid="stChatMessage"] {
  background: var(--ev-card) !important;
  border: 1px solid var(--ev-border) !important;
  border-radius: 12px !important;
  padding: 12px 14px !important;
}
[data-testid="stChatMessageAvatarUser"] { background-color: #21262d !important; }
[data-testid="stChatMessageAvatarAssistant"] { background-color: #1f2937 !important; }

/* ---- Chat input ---- */
[data-testid="stChatInput"], div[data-testid="stChatInput"] textarea {
  background-color: #0D1117 !important;
}
div[data-testid="stChatInput"] textarea {
  color: var(--ev-text) !important;
  border: 1px solid var(--ev-border) !important;
}

/* ---- Code blocks ---- */
.stCodeBlock, pre, code, [data-testid="stCodeBlock"] {
  background: #010409 !important;
  border: 1px solid var(--ev-border) !important;
  border-radius: 10px !important;
}
code { color: #c9d1d9 !important; }

/* ---- Buttons ---- */
.stButton > button, .stDownloadButton > button {
  background-color: #21262d !important;
  border: 1px solid var(--ev-border-strong) !important;
  color: var(--ev-text) !important;
  border-radius: 8px !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  border-color: var(--ev-accent) !important;
  background-color: #30363d !important;
  color: #ffffff !important;
}

/* ---- Metrics ---- */
[data-testid="stMetric"] {
  background: var(--ev-card); border: 1px solid var(--ev-border);
  border-radius: 10px; padding: 8px 12px;
}
[data-testid="stMetricValue"] { color: var(--ev-text) !important; }
[data-testid="stMetricDelta"] { color: var(--ev-green) !important; }

/* ---- Graphviz (dark) ---- */
[data-testid="stGraphvizChart"], [data-testid="stGraphvizChart"] svg {
  background: transparent !important;
}

/* ---- Dividers & scrollbar ---- */
hr { border-color: var(--ev-border) !important; background: var(--ev-border) !important; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--ev-bg); }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* ---- Selectbox / inputs ---- */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
  background-color: #0D1117 !important;
  border-color: var(--ev-border) !important;
}

/* ---- Expander ---- */
[data-testid="stExpander"] {
  background: var(--ev-card) !important;
  border: 1px solid var(--ev-border) !important;
  border-radius: 12px !important;
}
"""


def render_page_config() -> None:
    """Configure the Streamlit page (must be the first Streamlit call)."""
    st.set_page_config(
        page_title="Evolve: Adaptive Priority Engine",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                "Evolve: Adaptive Priority Engine — a context-aware AI consultation "
                "engine with a vector-graph memory (remember / recall / improve / forget)."
            ),
        },
    )


def inject_theme_css() -> None:
    """Inject the unified dark-theme stylesheet + desktop centering."""
    st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)
