from __future__ import annotations

from typing import List
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.recherche import rechercher, ResultatRecherche


# Chargement de .env
RACINE_PROJET = Path(__file__).resolve().parents[1]
load_dotenv(RACINE_PROJET / ".env")

MODELE_LLM = "gpt-4o-mini"


def construire_contexte(passages: List[ResultatRecherche]) -> str:
    blocs = []
    for p in passages:
        blocs.append(
            f"Source: {p.source}\n"
            f"Section: {p.header1} > {p.header2}\n"
            f"Contenu:\n{p.contenu}"
        )
    return "\n\n---\n\n".join(blocs)


def generer_reponse(question: str) -> str:
    passages = rechercher(
    question,
    k_final=4,
    candidates_dense=40,
    candidates_sparse=40,
    rrf_top_n=30,
    rerank_top_n=12,
)

    if not passages:
        return "Je ne trouve pas d'information pertinente dans les documents."

    contexte = construire_contexte(passages)

    llm = ChatOpenAI(model=MODELE_LLM, temperature=0)

    system_prompt = (
        "Tu es un assistant professionnel qui répond au sujet de Séphora MITOSSEDE.\n"
        "Tu parles d'elle à la troisième personne (\"Séphora\", \"elle\").\n"
        "\n"
        "RÈGLES STRICTES :\n"
        "1) Tu réponds UNIQUEMENT à partir du CONTEXTE fourni.\n"
        "2) Si une info n'est pas dans le contexte, tu dis : \"Je n'ai pas cette information dans mes documents.\".\n"
        "3) Tu n'inventes rien (pas d'entreprise, pas de dates, pas de responsabilités).\n"
        "4) Si la question porte sur des dates/périodes, tu restitues les périodes explicitement.\n"
        "5) Si plusieurs éléments existent (formations, expériences, projets), tu les classes par ordre chronologique si il y a des dates:\n"
        "   - par défaut : du plus récent au plus ancien.\n"
        "   - pour un parcours académique complet : du plus ancien au plus récent.\n"
        "6) Tu n'utilises \"actuellement\" / \"en ce moment\" QUE si le contexte contient \"en cours\".\n"
        "7) Style : clair, concis, professionnel (4 à 8 phrases max)."
    )

    human_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXTE:\n{contexte}\n\n"
        "INSTRUCTIONS DE RÉPONSE :\n"
        "- Réponds en français.\n"
        "- Si la question demande un résumé, fais un paragraphe.\n"
        "- Termine par 1 phrase courte si besoin (ex: impact / objectif), sans inventer.\n"
    )

    resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    return resp.content


if __name__ == "__main__":
    tests = [
    "Quelle est son parcours académique ?",
    "Qu’a-t-elle fait à l’INSEE ?",
    "Quels sont ses projets principaux ?",
]
    for q in tests:
        print("\n=== QUESTION ===")
        print(q)
        print("\n=== RÉPONSE ===")
        print(generer_reponse(q))