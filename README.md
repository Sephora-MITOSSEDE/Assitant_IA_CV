# SephBot — Assistant IA CV (RAG Hybride)

Application déployée :  
[Accéder à l'application](https://assistant-ia-sephbot.streamlit.app/)

## Objectif

SephBot est un assistant IA permettant d’interroger dynamiquement le parcours académique et professionnel de Séphora MITOSSEDE à partir de documents structurés en Markdown.

Le projet met en œuvre une architecture RAG (Retrieval-Augmented Generation) complète afin de produire des réponses fiables, contextualisées et sans hallucination.

---

## Architecture

Le système repose sur :

* Indexation vectorielle avec FAISS
* Recherche lexicale BM25
* Fusion hybride via Reciprocal Rank Fusion (RRF)
* Reranking par cross-encoder
* Génération contrôlée avec GPT-4o-mini

Les documents sont découpés par niveaux de titres (#, ##) et enrichis en métadonnées (source, section, identifiant de chunk).

---

## Fonctionnement

1. La question est envoyée au moteur de recherche hybride.
2. Les passages les plus pertinents sont sélectionnés.
3. Un reranking neuronal affine l’ordre de pertinence.
4. Le LLM génère une réponse strictement basée sur le contexte récupéré.
5. Les sources sont affichées dans l’interface.

Le prompt impose des règles strictes :

* Aucune invention
* Respect des dates
* Classement chronologique lorsque nécessaire
* Réponse professionnelle et concise

---

## Technologies utilisées

* Python
* Streamlit
* LangChain
* FAISS
* Sentence-Transformers
* Cross-Encoder
* Rank-BM25
* OpenAI API

---

## Compétences démontrées

* Conception d’une architecture RAG complète
* Hybrid retrieval dense + sparse
* Reranking neuronal
* Prompt engineering contraint
* Déploiement d’une application IA en production

---


