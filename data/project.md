---
type: projet
candidat: Séphora MITOSSEDE
competences_cles: Machine Learning, NLP, Déploiement API
---

# Projets Techniques

## Système de recommandation culinaire & IA conersationnelle (NLP)
**Pitch :** Application conversationnelle suggérant des recettes via similarité sémantique.
**Objectif :** Construire un moteur de recommandation basé sur le sens des requêtes utilisateur.
**Approche :**
- Développement d’un moteur de recommandation NLP combinant TF-IDF, similarité cosinus et KMeans pour proposer les top 5 recettes les plus pertinentes à partir des ingrédients saisis.
- Intégration d’une IA conversationnelle basée sur GPT-4o-mini pour affiner les recommandations selon les préférences et contraintes utilisateurs.
- Conception d’une application interactive sous Streamlit avec personnalisation des résultats en temps réel.
**Stack :** Python, scikit-learn, spacy, TF-IDF, KMeans, LLM, OpenAI API, Streamlit.

---

## AI-POWERED BANKING CUSTOMER RISK INTELLIGENCE PLATFORM
**Pitch :** Plateforme intelligente d’aide à la décision bancaire combinant scoring client, Explainable AI et assistant IA agentique pour prioriser les actions de rétention.
**Objectif :** Identifier les clients à risque de churn, expliquer les facteurs de risque et recommander des actions commerciales adaptées.
**Approche :**
- Développement de modèles de churn et de scoring client avec Logistic Regression, Random Forest et XGBoost.
- Gestion du déséquilibre des classes avec SMOTE.
- Interprétation des prédictions avec SHAP pour identifier les facteurs de risque par client.
- Segmentation intelligente des clients selon leur niveau de risque, leur valeur métier et les leviers de rétention possibles.
- Intégration d’un assistant IA agentique capable d’exploiter les scores, les explications SHAP et les règles métier pour proposer des recommandations d’action.
- Développement d’un dashboard Power BI combinant scoring, segmentation client, visualisations Explainable AI et aide à la décision.
**Stack :** Python, scikit-learn, XGBoost, SMOTE, SHAP, Power BI, IA agentique.

---

## Plateforme IA d’aide à la décision A/B Testing — Projet en cours
**Pitch :** Assistant statistique intelligent capable d’analyser les résultats d’un test A/B et de générer une recommandation Go/No Go compréhensible par les équipes métier.
**Objectif :** Concevoir une application combinant tests statistiques, visualisation et IA générative pour transformer les résultats d’expérimentation en décisions actionnables.
**Approche prévue :**
- Calcul de la significativité statistique, des intervalles de confiance et de la taille d’effet.
- Analyse automatique des résultats selon le taux de conversion, le volume d’échantillon et l’incertitude statistique.
- Génération d’interprétations textuelles orientées décision à l’aide d’un LLM.
- Recommandation automatique Go/No Go avec justification statistique et métier.
- Visualisation des résultats dans un dashboard interactif.
**Stack prévue :** Python, statistiques, Streamlit, Power BI, OpenAI API.

---

## Agent IA personnel (LLM + RAG)
**Pitch :** Assistant conversationnel expert sur mon propre parcours.
**Objectif :** Répondre aux recruteurs de manière fiable en utilisant mes documents.
**Approche :**
- Développement d’un assistant IA permettant d’interroger un parcours académique et professionnel à partir de documents structurés.
- Conception d’une architecture RAG hybride combinant recherche vectorielle (embeddings) et recherche lexicale (BM25) avec reranking neuronal.
- Génération de réponses contextualisées, fiables et sans hallucination via une interface interactive sous Streamlit.
**Stack :** Python, LangChain, OpenAI API, FAISS, Streamlit.

---

## AGRISAVE — Détection de maladies des plantes (CNN)
**Pitch :** Application intelligente de détection précoce des maladies végétales à partir d’images de feuilles.
**Objectif :** Aider à l’identification rapide des pathologies agricoles afin d’améliorer la prise de décision et limiter les pertes de production.
**Approche :**
- Développement d’une application interactive sous RShiny pour l’analyse d’images végétales.
- Intégration d’un modèle EfficientNetB0 pré-entraîné utilisant le transfer learning pour la classification des maladies.
- Génération d’un diagnostic détaillé avec symptômes et recommandations adaptées.
**Stack :** R, RShiny, Deep Learning, EfficientNetB0, Transfer Learning.

---

## PIPELINE ETL DE WEB SCRAPING À GRANDE ÉCHELLE
**Pitch :** Pipeline automatisé de collecte et de structuration de données culinaires à grande échelle à partir du web.
**Objectif :** Construire une base de données exploitable pour des projets de recommandation, d’analyse textuelle et de NLP.
**Approche :**
- Extraction automatisée de plus de 42 000 recettes à partir de sitemaps et pages HTML via des techniques de web scraping.
- Conception d’un pipeline ETL pour le nettoyage, la transformation et le chargement des données dans un dataset structuré sous Pandas.
- Optimisation des performances du pipeline grâce à la parallélisation des traitements avec ThreadPoolExecutor.
**Stack :** Python, BeautifulSoup, Pandas, Web Scraping, ETL, ThreadPoolExecutor.
