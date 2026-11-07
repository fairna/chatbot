"""Minimal chat UI. Run: streamlit run app.py"""

import html
import streamlit as st
import traceback
from rag import ask_stream

st.set_page_config(
    page_title="Think Like Lee Kuan Yew",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Source+Serif+4:wght@400;600&display=swap');
:root{--bg:#FAFAF8;--ink:#1A1A1A;--muted:#6B6B66;--line:#E7E5DF;--card:#FFFFFF;--user:#EFEDE7;--accent:#B3261E;}
html, body, .stApp, [data-testid="stAppViewContainer"]{background:var(--bg);color:var(--ink);font-family:'Inter',system-ui,sans-serif;}
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"]{display:none !important;}
.block-container{max-width:720px;padding:2.5rem 1.25rem 8rem;}

h1.lky-title{font-family:'Source Serif 4',Georgia,serif;font-weight:600;font-size:1.9rem;letter-spacing:-0.01em;margin:0;padding:0;}
.lky-sub{color:var(--muted);font-size:.85rem;margin:.3rem 0 0;line-height:1.5;}

.hero{margin:3.5rem 0 1.5rem;}
.hero-q{font-family:'Source Serif 4',Georgia,serif;font-size:1.6rem;margin:0 0 .5rem;}
.hero-s{color:var(--muted);font-size:.95rem;line-height:1.6;margin:0;max-width:34rem;}

div.stButton > button{width:100%;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:12px;
  padding:.8rem 1rem;font-size:.88rem;font-weight:400;justify-content:flex-start;text-align:left;box-shadow:none;transition:border-color .15s;}
div.stButton > button p{text-align:left;}
div.stButton > button:hover{border-color:var(--accent);color:var(--ink);background:var(--card);}
div.stButton > button:focus:not(:active){border-color:var(--accent);color:var(--ink);}
.st-key-newchat button{width:auto;padding:.3rem .75rem;font-size:.8rem;color:var(--muted);float:right;}

[data-testid="stChatMessage"]{background:transparent;border:none;padding:.75rem 0;gap:0;}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"]{display:none;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){flex-direction:row-reverse;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{
  flex:0 1 auto;max-width:85%;background:var(--user);border:1px solid transparent;border-radius:16px 16px 4px 16px;padding:.65rem 1rem;font-size:.95rem;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"]{
  flex:0 1 auto;max-width:92%;background:var(--card);border:1px solid var(--line);border-radius:16px 16px 16px 4px;padding:.85rem 1.1rem;
  font-family:'Source Serif 4',Georgia,serif;font-size:1rem;line-height:1.7;}
.who{font-family:'Inter',system-ui,sans-serif;font-size:.72rem;letter-spacing:.02em;color:var(--muted);margin-bottom:.4rem;}
.who::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);margin-right:.45rem;vertical-align:1px;}

[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:var(--bg);margin-top:.75rem;}
[data-testid="stExpander"] summary{font-family:'Inter',system-ui,sans-serif;font-size:.8rem;color:var(--muted);}
.src{font-family:'Inter',system-ui,sans-serif;padding:.5rem 0;border-top:1px solid var(--line);}
.src:first-child{border-top:none;}
.src-h{font-size:.78rem;font-weight:500;}
.src-t{font-size:.78rem;color:var(--muted);line-height:1.55;margin-top:.15rem;}

[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"]{background:var(--bg);}
[data-testid="stChatInput"]{background:var(--card);border:1px solid var(--line);border-radius:14px;}
[data-testid="stChatInput"]:focus-within{border-color:var(--accent);box-shadow:none;}
[data-testid="stChatInput"] textarea{font-family:'Inter',system-ui,sans-serif;font-size:.95rem;}
</style>
""",
    unsafe_allow_html=True,
)

SUGGESTIONS = [
    "How do I attract investors to a country with no natural resources?",
    "What makes a leader worth following?",
    "How should a small country deal with bigger powers?",
    "How would you approach AI?",
]
WHO = '<div class="who">Lee Kuan Yew · AI simulation</div>'

if "history" not in st.session_state:
    st.session_state.history = []


def set_pending(q):
    st.session_state.pending = q


def new_chat():
    st.session_state.history = []


def render_sources(hits):
    with st.expander(f"Sources ({len(hits)})"):
        for h in hits:
            label = f"{h['source']} · {h['type']}"
            if h["year"]:
                label += f" · {h['year']}"
            label += f" · p.{h['page']}"
            if "score" in h:
                label += f" · relevance {h['score']:.2f}"
            st.markdown(
                f'<div class="src"><div class="src-h">{html.escape(label)}</div>'
                f'<div class="src-t">{html.escape(h["text"][:320])}…</div></div>',
                unsafe_allow_html=True,
            )


def render_msg(m):
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            st.markdown(WHO, unsafe_allow_html=True)
        st.markdown(m["content"])
        if m.get("hits"):
            render_sources(m["hits"])


left, right = st.columns([5, 1])
with left:
    st.markdown(
        '<h1 class="lky-title">Think Like Lee Kuan Yew</h1>'
        '<p class="lky-sub">AI simulation based on 1964–1968 speeches and book excerpts. '
        "Not his actual words.</p>",
        unsafe_allow_html=True,
    )
with right:
    if st.session_state.history:
        st.button("New chat", key="newchat", on_click=new_chat)

pending = st.session_state.pop("pending", None)
typed = st.chat_input("Ask anything: leadership, geopolitics, economy, life")
q = pending or typed
prior = list(st.session_state.history)
if q:
    st.session_state.history.append({"role": "user", "content": q})

if not st.session_state.history:
    st.markdown(
        '<div class="hero"><p class="hero-q">Ask how he would think.</p>'
        '<p class="hero-s">Answers are grounded in the documents and labeled '
        "when they are extrapolation.</p></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        cols[i % 2].button(s, key=f"sug{i}", on_click=set_pending, args=(s,))
else:
    for m in st.session_state.history:
        render_msg(m)

if q:
    with st.chat_message("assistant"):
        st.markdown(WHO, unsafe_allow_html=True)
        try:
            with st.spinner("Searching the answer..."):
                stream, hits = ask_stream(q, prior)  # rewrite + retrieval only
            answer = st.write_stream(stream)  # answer appears as it is written
        except Exception as e:
            traceback.print_exc()
            st.session_state.history.pop()
            st.error(f"Couldn’t get an answer: {e}")
            st.stop()
        render_sources(hits)
    st.session_state.history.append(
        {"role": "assistant", "content": answer, "hits": hits}
    )
