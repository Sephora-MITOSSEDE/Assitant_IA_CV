from __future__ import annotations

from typing import List
from pathlib import Path
import os
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.recherche import rechercher, ResultatRecherche

# charge .env en local
RACINE_PROJET = Path(__file__).resolve().parents[1]
load_dotenv(RACINE_PROJET / ".env")

MODELE_LLM = "gpt-4o-mini"


def _get_openai_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


OPENAI_API_KEY = _get_openai_key()

def question_est_vague(question: str) -> bool:
    q = question.lower().strip()

    expressions_vagues = [
        "elle y",
        "il y",
        "y fait",
        "là-bas",
        "depuis quand",
        "dans ce poste",
        "dans cette expérience",
        "sur ce projet",
        "ce projet",
        "cette expérience",
        "cet emploi",
        "ce poste",
        "et là",
        "et ensuite",
    ]

    questions_tres_courtes = {
        "et en alternance ?",
        "et à l'insee ?",
        "et là-bas ?",
        "et ensuite ?",
        "depuis quand ?",
        "elle y fait quoi ?",
        "qu'est-ce qu'elle y fait ?",
        "qu’y fait-elle ?",
    }

    if q in questions_tres_courtes:
        return True

    return any(expr in q for expr in expressions_vagues)


def construire_requete_recherche(question: str, historique: list[dict] | None = None) -> str:
    if not historique or not question_est_vague(question):
        return question

    derniers_messages = historique[-4:]
    morceaux = []

    for msg in derniers_messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            content = content.strip()
            if content:
                morceaux.append(content)

    contexte_recent = " ".join(morceaux)

    return f"{contexte_recent} {question}"


def construire_contexte(passages: List[ResultatRecherche]) -> str:
    blocs = []
    for p in passages:
        blocs.append(
            f"Source: {p.source}\n"
            f"Section: {p.header1} > {p.header2}\n"
            f"Contenu:\n{p.contenu}"
        )
    return "\n\n---\n\n".join(blocs)


@st.cache_resource
def charger_llm():
    return ChatOpenAI(
        model=MODELE_LLM,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )


def generer_reponse(question: str, historique: list[dict] | None = None) -> str:
    question_clean = question.strip().lower()

    salutations = {"bonjour", "salut", "hello", "bonsoir", "coucou"}
    if question_clean in salutations:
        return (
            "Bonjour, je suis SephBot, l’assistant IA de Séphora MITOSSEDE. "
            "Je peux répondre à vos questions sur son parcours, ses compétences, "
            "ses projets et ses expériences."
        )

    requete_recherche = construire_requete_recherche(question, historique)

    passages = rechercher(
        requete_recherche,
        k_final=6,
        candidates_dense=15,
        candidates_sparse=15,
        rrf_top_n=20,
        rerank_top_n=0,  # Reranking désactivé pour améliorer les performances et réduire le temps de démarrage.
    )



    if not passages:
        return "Je ne trouve pas d'information pertinente dans les documents."

    contexte = construire_contexte(passages)

    if not OPENAI_API_KEY:
        return (
            "Clé OpenAI manquante : définis OPENAI_API_KEY dans .env (local) "
            "ou dans Secrets (Streamlit Cloud)."
        )

    llm = charger_llm()

    system_prompt = (
        "Tu es un assistant professionnel qui répond au sujet de Séphora MITOSSEDE.\n"
        "Tu parles d'elle à la troisième personne (\"Séphora\", \"elle\").\n"
        "\n"
        "RÈGLES STRICTES :\n"
        "1) Tu réponds UNIQUEMENT à partir du CONTEXTE fourni.\n"
        "2) Si une info n'est pas dans le contexte, tu dis : "
        "\"Je n'ai pas cette information dans mes documents.\".\n"
        "3) Tu n'inventes rien (pas d'entreprise, pas de dates, pas de responsabilités).\n"
        "4) Si la question porte sur des dates/périodes, tu restitues les périodes explicitement.\n"
        "5) Si plusieurs éléments existent (formations, expériences, projets), tu les classes par ordre chronologique si il y a des dates:\n"
        "   - par défaut : du plus récent au plus ancien.\n"
        "   - pour un parcours académique complet : du plus ancien au plus récent.\n"
        "6) Tu n'utilises \"actuellement\" / \"en ce moment\" QUE si le contexte contient \"en cours\".\n"
        "7) Réponds directement à la question, sans formule d’introduction inutile.\n"
        "8) N’écris pas \"Bonjour\", \"Avec plaisir\", \"Bien sûr\" ou toute autre formule de politesse, "
        "sauf si l’utilisateur envoie uniquement une salutation.\n"
        "9) Ne te présentes que si l’utilisateur salue ou demande explicitement qui tu es.\n"
        "10) Ne reformule pas la question. Donne d’abord l’information utile.\n"
        "11) Style : clair, concis, professionnel, orienté recruteur.\n"
        "12) Réponse courte : privilégie un paragraphe fluide de 3 à 6 phrases. Utilise une liste uniquement si la question demande explicitement un détail ou une chronologie.\n"
        "13) Si l'historique récent éclaire une référence comme \"elle\", \"son\", "
        "\"cette expérience\", \"ce projet\", tu peux t'en servir pour comprendre la question, "
        "mais jamais pour ajouter des faits absents du CONTEXTE."
        "14) Pour les questions sur le parcours, la formation ou les expériences, "
        "reconstruis une chronologie cohérente à partir des dates présentes dans le contexte.\n"
        "15) Ne jamais présenter une ancienne formation comme actuelle si une formation plus récente existe dans le contexte.\n"
        "16) Si plusieurs formations existent, la plus récente correspond à la situation académique actuelle.\n"
        "17) Pour les questions sur le parcours académique ou professionnel, commence par une courte synthèse avant de détailler les formations ou expériences.\n"
    )

    # On garde seulement quelques messages récents
    historique_messages = []
    if historique:
        derniers_messages = historique[-4:]  # 3 échanges max environ
        for msg in derniers_messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if not content:
                continue

            if role == "user":
                historique_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                historique_messages.append(AIMessage(content=content))

    human_prompt = (
        f"QUESTION ACTUELLE :\n{question}\n\n"
        f"CONTEXTE :\n{contexte}\n\n"
        "INSTRUCTIONS DE RÉPONSE :\n"
        "- Réponds en français.\n"
        "- Va directement à l'information demandée.\n"
        "- Si la question demande un résumé, fais un paragraphe court.\n"
        "- Réponds toujours avec un paragraphe court et directe à moins que cela ne nécéssite une liste ou un résumé plus structuré.\n"
        "- N'ajoute aucune information absente du contexte.\n"
    )

    messages_llm = [SystemMessage(content=system_prompt)]
    messages_llm.extend(historique_messages)
    messages_llm.append(HumanMessage(content=human_prompt))

    resp = llm.invoke(messages_llm)
    return resp.content


if __name__ == "__main__":
    tests = [
        "Est-elle en alternance?",
        "Quels sont ses projets principaux ?",
        "Quelle est son parcours académique?"
    ]
    for q in tests:
        print("\n=== QUESTION ===")
        print(q)
        print("\n=== RÉPONSE ===")
        print(generer_reponse(q))
