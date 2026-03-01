from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import uuid

import streamlit as st
from dotenv import load_dotenv

from src.generation import generer_reponse
from src.recherche import rechercher, ResultatRecherche


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


def construire_contexte(passages: List[ResultatRecherche]) -> str:
    blocs = []
    for p in passages:
        blocs.append(
            f"Source: {p.source}\n"
            f"Section: {p.header1} > {p.header2}\n"
            f"Contenu:\n{p.contenu}"
        )
    return "\n\n---\n\n".join(blocs)


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
st.markdown(
    """
    <div class="card">
      <h2 style="margin:0;">Assistant CV — SephBot</h2>
      <div class="small-muted" style="margin-top:.35rem;">
        Pose moi des questions sur le parcours, les expériences, les projets et les compétences de Séphora.
      </div>
      <div style="margin-top:.7rem;">
        <span class="pill">RAG hybride</span>
        <span class="pill">FAISS + BM25</span>
        <span class="pill">Rerank cross-encoder</span>
        <span class="pill">GPT-4o-mini</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


# ----------------------------
# Afficher les messages du chat courant
# ----------------------------
chat = _get_current_chat()
messages = chat["messages"]

for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Voir les sources", expanded=False):
                for i, s in enumerate(m["sources"], start=1):
                    st.markdown(f"**[{i}] {s['source']} — {s['section']}**")
                    st.write(s["contenu"])


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
            passages = rechercher(
                question,
                k_final=4,
                candidates_dense=40,
                candidates_sparse=40,
                rrf_top_n=30,
                rerank_top_n=12,
            )

            contexte = construire_contexte(passages) if passages else ""


            reponse = generer_reponse(question)

        st.markdown(reponse)

        sources = [
            {
                "source": p.source,
                "section": f"{p.header1} > {p.header2}",
                "contenu": p.contenu[:700] + ("…" if len(p.contenu) > 700 else ""),
            }
            for p in passages
        ]

        if sources:
            with st.expander("Voir les sources", expanded=False):
                for i, s in enumerate(sources, start=1):
                    st.markdown(f"**[{i}] {s['source']} — {s['section']}**")
                    st.write(s["contenu"])

    # Ajoute message assistant
    messages.append({"role": "assistant", "content": reponse, "sources": sources})