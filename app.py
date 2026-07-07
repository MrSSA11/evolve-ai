"""Evolve: Adaptive Priority Engine — core execution file.

Coordinates layout, columns, session state and rendering, wiring together:

- backend.context  — context routing + temporal fail-safe
- backend.llm      — chat generation + dynamic persona + citation parsing
- backend.memory   — mock vector-graph DB (remember / recall / improve / forget)
- backend.patterns — analytical drift + hidden pattern parsing
- backend.utils    — helpers
- frontend.ui      — page config + unified dark theme + desktop centering
- frontend.components — rich dashboard components

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os
import streamlit as st

from frontend import ui, components
from backend import context, llm, memory, patterns, utils

# Shared, persistent memory-graph singleton.
MEM = memory.store


# --------------------------------------------------------------------------- #
# Session-state initialisation
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {
        "messages": [],            # chat history: [{"role","content"}]
        "mode_history": ["Personal"],
        "mode_switch_log": [],
        "current_mode": "Personal",
        "last_routing": None,      # last context-routing decision
        "last_pattern": None,      # last pattern analysis
        "_pattern_mh_len": -1,     # cache key for pattern re-analysis
        "persona_rules": [],       # injected behavioral adjustment rules
        "last_recall": None,       # last recall_memory() result
        "last_citations": [],      # frame ids cited in the last reply
        "nuke_confirm": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------------------------------------------------------- #
# Sidebar hooks
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🧬 Evolve")
        st.caption("Adaptive Priority Engine")
        st.divider()

        stats = MEM.get_stats()
        st.markdown(f"**Frames:** {stats['frames']}  ·  **Clusters:** {stats['clusters']}")
        by_mode = stats.get("by_mode", {}) or {}
        if by_mode:
            st.caption("By mode: " + ", ".join(f"{m} {n}" for m, n in by_mode.items()))
        st.caption(f"Store: `{os.path.basename(stats.get('base_dir', ''))}/`")

        st.divider()

        # ---- Memory pipeline controls ----
        st.markdown("**Memory Pipeline**")

        if st.button("🧠 Improve — Synthesize Clusters", use_container_width=True):
            with st.spinner("Consolidating memory into knowledge clusters…"):
                res = MEM.synthesize_memory()
            st.toast(res.get("message", "Done"), icon="🧠")
            # Re-evaluate patterns after consolidation.
            st.session_state["_pattern_mh_len"] = -1
            st.rerun()

        st.markdown("**Forget — Granular**")
        frames = MEM.get_all_frames()
        if frames:
            recent = frames[-30:][::-1]
            options = {
                f"[Frame #{f['frame_id']}] {utils.truncate(f['content'], 40)}": f["frame_id"]
                for f in recent
            }
            choice = st.selectbox("Select a frame to delete", list(options.keys()))
            if st.button("🗑️ Delete Frame", use_container_width=True):
                res = MEM.delete_specific_memory(options[choice])
                st.toast(res.get("message", "Deleted"), icon="🗑️")
                st.session_state["_pattern_mh_len"] = -1
                st.rerun()
        else:
            st.caption("No frames to delete.")

        st.markdown("**Forget — Nuke Protocol**")
        st.session_state.nuke_confirm = st.checkbox(
            "I understand this wipes ALL memory",
            value=bool(st.session_state.get("nuke_confirm", False)),
        )
        if st.button(
            "☢️ Nuke Database",
            use_container_width=True,
            disabled=not st.session_state.nuke_confirm,
        ):
            res = MEM.forget_memory()
            st.session_state.messages = []
            st.session_state.mode_history = ["Personal"]
            st.session_state.mode_switch_log = []
            st.session_state.current_mode = "Personal"
            st.session_state.last_routing = None
            st.session_state.last_pattern = None
            st.session_state.last_recall = None
            st.session_state.last_citations = []
            st.session_state["_pattern_mh_len"] = -1
            st.toast(res.get("message", "Database nuked"), icon="☢️")
            st.rerun()

        st.divider()
        st.markdown("**Chat**")
        if st.button("🧹 Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_recall = None
            st.session_state.last_citations = []
            st.rerun()

        st.divider()
        st.caption(f"LLM model: `{utils.env_or('EVOLVE_MODEL', 'gemma-4-31b-it')}`")
        st.caption(f"LLM live: {'✅ yes' if llm.is_live() else '⚪ offline mock'}")
        if not llm.is_live():
            st.caption("Set `GEMINI_API_KEY` (and `pip install google-genai`) to go live.")
        st.caption(f"Workspace files: {len(utils.list_workspace_files())}")

        with st.expander("How Evolve works"):
            st.markdown(
                """
                1. **Context routing** classifies each turn into Builder / Exam /
                   Personal (LLM, with a keyword fail-safe).
                2. **remember()** stores every message as an indexed *Memory Frame*.
                3. **recall()** pulls the most relevant frames (cited as `[Frame #N]`).
                4. The **LLM** answers, citing the frames it actually used.
                5. **improve()** consolidates frames into *Knowledge Clusters*.
                6. **forget()** prunes a frame or wipes the whole DB (Nuke Protocol).
                7. **Patterns** watch for task-switching friction and offer an
                   *Auto-Adjustment Protocol* that injects a rule into the persona.
                """
            )


# --------------------------------------------------------------------------- #
# Pattern panel (hidden pattern detection + autonomous resolution)
# --------------------------------------------------------------------------- #
def render_pattern_section() -> None:
    # Only re-analyse when the mode history actually changed (cheap caching).
    mh_len = len(st.session_state.mode_history)
    if st.session_state.get("_pattern_mh_len") != mh_len:
        st.session_state.last_pattern = patterns.analyze_patterns(st.session_state)
        st.session_state["_pattern_mh_len"] = mh_len

    pattern = st.session_state.last_pattern or {"detected": False}
    if not pattern.get("detected"):
        return

    metrics = pattern.get("metrics", {}) or {}
    friction = float(metrics.get("friction_score", 0.0) or 0.0)

    st.markdown(
        f"""
        <div class="ev-card" style="border-color:#D2992255;background:rgba(210,153,34,0.05);">
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
            <span class="ev-pill" style="background:#D2992222;border-color:#D29922;color:#D29922;">
              ⚡ Hidden Pattern Detected
            </span>
            <span class="ev-pill ev-muted">Friction {friction:.2f}</span>
            <span class="ev-pill ev-muted">{metrics.get('switches', 0)} switches</span>
          </div>
          <div style="margin-top:8px;">
            <b>Root Cause:</b>
            <span>{pattern.get('root_cause', '')}</span>
          </div>
          <div class="ev-muted" style="margin-top:4px;font-size:0.85rem;">
            💡 {pattern.get('recommendation', '')}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("⚡ Execute Auto-Adjustment Protocol"):
        rule = (
            pattern.get("recommendation_rule")
            or pattern.get("recommendation")
            or "Stay focused and confirm before switching contexts."
        )
        st.session_state.persona_rules.append(rule)
        # Mark resolved until the next mode change so we don't nag.
        st.session_state.last_pattern = {"detected": False, "resolved": True}
        st.toast("Auto-adjustment rule injected into the active persona.", icon="⚡")
        st.rerun()


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def handle_input(user_input: str) -> None:
    # 1) Context routing (with temporal + keyword fail-safe).
    ephemeral = list(st.session_state.messages) + [{"role": "user", "content": user_input}]
    routing = context.detect_context(ephemeral)
    st.session_state.current_mode = routing["mode"]
    patterns.record_mode_switch(st.session_state, routing["mode"], reason=routing.get("reasoning", ""))
    st.session_state.last_routing = routing

    # 2) remember() — persist the user turn.
    MEM.remember_message(user_input, routing["mode"], metadata={"role": "user"})

    # 3) recall() — retrieve relevant memory frames + clusters.
    mem_ctx = MEM.recall_memory(user_input, top_k=5)
    st.session_state.last_recall = mem_ctx

    # 4) Generate with dynamic persona + cited memory.
    response, citations = llm.generate_response(
        user_input=user_input,
        chat_history=st.session_state.messages,
        mode=routing["mode"],
        memory_context=mem_ctx,
        extra_rules=st.session_state.persona_rules,
    )
    st.session_state.last_citations = citations

    # 5) remember() the assistant reply too.
    MEM.remember_message(response, routing["mode"], metadata={"role": "assistant"})

    # 6) Append to chat history.
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ui.render_page_config()
    ui.inject_theme_css()
    init_state()
    render_sidebar()

    components.render_header()

    # Status banner (routing reasoning).
    if st.session_state.last_routing:
        components.render_status_banner(st.session_state.last_routing)

    # Autonomous Resolution Engine / hidden patterns.
    render_pattern_section()

    # Analytics KPI matrix (full-width, responsive).
    components.render_analytics_cards(MEM.get_stats(), st.session_state)

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    # Two-column layout: chat (left, wider) + live dashboard (right).
    chat_col, dash_col = st.columns([2.1, 1.0], gap="large")

    with chat_col:
        st.markdown("#### 💬 Consultation")
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                avatar = "🧬" if msg["role"] == "assistant" else "👤"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

    with dash_col:
        st.markdown("##### 🗺️ Lifecycle Graph")
        components.render_lifecycle_graph(
            st.session_state.current_mode,
            MEM.get_all_frames(),
            MEM.get_all_clusters(),
        )

        st.markdown("##### 📌 Cited Sources")
        components.render_memory_sources(
            st.session_state.last_recall, st.session_state.last_citations
        )

        st.markdown("##### 🕘 Memory Timeline")
        components.render_memory_timeline(MEM.get_all_frames())

        st.markdown("##### 🧩 Knowledge Clusters")
        components.render_knowledge_clusters(MEM.get_all_clusters())

        st.markdown("##### 💾 Code Workspace")
        last_assistant = next(
            (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
            "",
        )
        components.render_code_workspace(utils.extract_code_blocks(last_assistant))

    # Chat input (pinned to the bottom of the viewport).
    user_input = st.chat_input("Ask Evolve anything — it adapts to your focus…")
    if user_input:
        handle_input(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
