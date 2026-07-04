import streamlit as st
from backend.llm import ask_llm, get_active_model

st.set_page_config(page_title="Evolve", page_icon="🧠")

st.title("🧠 Evolve")
st.caption("Your AI should grow with you, not reset every conversation.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.success(get_active_model())

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Talk to Evolve")

if prompt:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    answer = ask_llm(prompt)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )

    st.rerun()
