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

## Customer Churn Decision Engine
**Pitch :** Système d’aide à la décision basé sur le machine learning pour identifier les clients à risque et optimiser les campagnes de rétention.

**Objectif :** Prédire le churn et prioriser les actions commerciales en maximisant le retour sur investissement des campagnes de fidélisation.

**Approche :**
- Feature engineering basé sur le comportement client (usage des services, facturation, ancienneté).
- Comparaison de modèles (Régression logistique, Random Forest, XGBoost) avec pipeline ML reproductible.
- Gestion du déséquilibre des classes avec SMOTE et SMOTE-Tomek.
- Interprétabilité du modèle via SHAP pour expliquer les facteurs de résiliation.
- Optimisation du seuil de décision à partir d’une analyse **coût-bénéfice** pour cibler les clients les plus rentables à retenir.

**Déploiement :**
- API REST développée avec **FastAPI** pour le scoring des clients.
- Conteneurisation avec **Docker** pour un déploiement reproductible.
- Tableau de bord interactif pour analyser les risques de churn et simuler des stratégies de rétention.

**Stack :** Python, scikit-learn, XGBoost, SHAP, FastAPI, Docker, Power BI.

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