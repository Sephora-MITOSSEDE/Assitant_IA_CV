from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import uuid
import base64
import streamlit as st
from dotenv import load_dotenv

from src.generation import generer_reponse



# ----------------------------
# Config
# ----------------------------
RACINE_PROJET = Path(__file__).resolve().parent
load_dotenv(RACINE_PROJET / ".env")

st.set_page_config(
    page_title="Assistant CV — SephBot",
    page_icon="💬",
    layout="centered",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 900px; }
      .small-muted { color: rgba(255,255,255,0.65); font-size: 0.92rem; }
      .card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(255,255,255,0.03);
      }
      .pill {
        display:inline-block; padding: .25rem .6rem; border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.06);
        font-size: .85rem; margin-right: .35rem;
      }
      .sidebar-title {
        font-weight: 700;
        margin: .2rem 0 .6rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# State init
# ----------------------------
if "chats" not in st.session_state:
    # chats: {chat_id: {"title": str, "messages": list[dict]}}
    st.session_state["chats"] = {}

if "current_chat_id" not in st.session_state:
    st.session_state["current_chat_id"] = None


def _new_chat() -> str:
    """Crée un nouveau chat vide et le rend actif."""
    chat_id = str(uuid.uuid4())[:8]
    st.session_state["chats"][chat_id] = {"title": "Nouvelle conversation", "messages": []}
    st.session_state["current_chat_id"] = chat_id
    return chat_id


def _get_current_chat() -> Dict[str, Any]:
    """Retourne le chat actif (créé s'il n'existe pas)."""
    if (
        st.session_state["current_chat_id"] is None
        or st.session_state["current_chat_id"] not in st.session_state["chats"]
    ):
        _new_chat()
    return st.session_state["chats"][st.session_state["current_chat_id"]]



# ----------------------------
# Sidebar: liste des conversations (style ChatGPT)
# ----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">Conversations</div>', unsafe_allow_html=True)

    if st.button("➕ Nouvelle conversation", use_container_width=True):
        _new_chat()
        st.rerun()

    st.divider()

    # Liste des anciens chats (du plus récent au plus ancien)
    items = list(st.session_state["chats"].items())
    items.reverse()

    for chat_id, chat in items:
        title = chat.get("title", "Conversation")
        label = title if len(title) <= 38 else title[:38] + "…"

        is_active = (chat_id == st.session_state["current_chat_id"])
        btn_label = ("✅ " if is_active else "") + label

        if st.button(btn_label, key=f"chat_{chat_id}", use_container_width=True):
            st.session_state["current_chat_id"] = chat_id
            st.rerun()


# ----------------------------
# Header
# ----------------------------

logo_path = RACINE_PROJET / "data" / "logo_chat.jpg"
with open(logo_path, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()


st.markdown(
    f"""
<style>
.seph-header {{
    text-align: center;
}}

.seph-title-row {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin-bottom: 0.35rem;
}}

.seph-logo {{
    width: 64px;
    height: 64px;
    border-radius: 50%;
    object-fit: cover;
}}

.seph-title {{
    margin: 0;
    font-size: 2.7rem;
    font-weight: 800;
    color: #7ec8ff;
    line-height: 1;
}}

.seph-subtitle {{
    color: #c8d1dc;
    margin-top: 0.2rem;
    font-size: 1.05rem;
}}

.seph-desc {{
    margin-top: 0.75rem;
    color: #aab4c0;
    font-size: 0.97rem;
}}

.seph-links {{
    margin-top: 1rem;
    display: flex;
    justify-content: center;
    gap: 26px;
    flex-wrap: wrap;
}}

.seph-links a {{
    text-decoration: none !important;
    color: #7ec8ff !important;
    font-weight: 600;
}}

.seph-links a:hover {{
    text-decoration: none !important;
    color: #a9dcff !important;
}}
</style>

<div class="card seph-header">
<div class="seph-title-row">
<img class="seph-logo" src="data:image/jpeg;base64,{logo_base64}" />
<h1 class="seph-title">SephBot</h1>
</div>
<div class="seph-subtitle">Assistant IA du CV de Séphora</div>
<div class="seph-desc">Interrogez SephBot pour découvrir le parcours, les projets et les compétences de Séphora MITOSSEDE.</div>
<div class="seph-links">
<a href="https://github.com/Sephora-MITOSSEDE" target="_blank">GitHub</a>
<a href="https://www.linkedin.com/in/sephora-mitossede-a4b62a346/" target="_blank">LinkedIn</a>
<a href="https://sephora-mitossede.github.io/sephora-M.github.io/" target="_blank">Portfolio</a>
<a href="mailto:mitossedes@gmail.com">Email</a>
</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# ----------------------------
# Affiche les messages du chat courant
# ----------------------------
chat = _get_current_chat()
messages = chat["messages"]

for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])



# ----------------------------
# Input
# ----------------------------
question = st.chat_input("Écris ta question…")

if question:
    # Ajoute message user
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Fix titre du chat = première question
    if chat["title"] == "Nouvelle conversation":
        chat["title"] = question

    # Retrieval + génération
    with st.chat_message("assistant"):
        with st.spinner("Réflexion…"):
            


            reponse = generer_reponse(question, historique=messages[:-1])

        st.markdown(reponse)

    

    # Ajoute message assistant
    messages.append({"role": "assistant", "content": reponse})