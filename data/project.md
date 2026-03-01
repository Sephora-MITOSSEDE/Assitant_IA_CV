---
type: projet
candidat: Séphora MITOSSEDE
competences_cles: Machine Learning, NLP, Déploiement API
---

# Projets Techniques

## Système de recommandation culinaire (NLP)
**Pitch :** Application conversationnelle suggérant des recettes via similarité sémantique.
**Objectif :** Construire un moteur de recommandation basé sur le sens des requêtes utilisateur.
**Approche :**
- Vectorisation des textes (TF-IDF et embeddings).
- Mesure de similarité cosinus.
- Intégration dans une interface Streamlit.
**Stack :** Python, scikit-learn, LangChain, OpenAI API, Streamlit.

---

## Prédiction du churn client
**Pitch :** Modèle prédictif pour identifier les risques de résiliation business.
**Objectif :** Anticiper le départ des clients pour prioriser les actions de rétention.
**Approche :**
- Comparaison de modèles : Régression logistique, Random Forest, XGBoost.
- Gestion du déséquilibre : SMOTE, SMOTE-Tomek.
- Interprétabilité : SHAP values et importance des variables.
**Déploiement :** API créée avec FastAPI et conteneurisation Docker.
**Stack :** Python, scikit-learn, XGBoost, FastAPI, Docker, Power BI.

---

## Plateforme d’aide à la décision A/B Testing
**Pitch :** Outil d'interprétation automatique de tests statistiques.
**Objectif :** Permettre aux décideurs de valider une version (Go/No Go) sans expertise statistique.
**Approche :**
- Calcul de significativité, intervalles de confiance et taille d’effet.
- Génération automatique d’interprétations textuelles.
**Interface :** Dashboard interactif.
**Stack :** Python, statistiques, Streamlit, Power BI.

---

## Agent IA personnel (LLM + RAG)
**Pitch :** Assistant conversationnel expert sur mon propre parcours (celui-ci même !).
**Objectif :** Répondre aux recruteurs de manière fiable en utilisant mes documents.
**Approche :**
- Recherche sémantique (RAG) sur CV et projets.
- Utilisation de vecteurs (embeddings) et base de données FAISS.
**Stack :** Python, LangChain, OpenAI API, FAISS, Streamlit.