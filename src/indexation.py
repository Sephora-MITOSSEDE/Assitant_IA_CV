from __future__ import annotations
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# On importe la fonction du fichier précédent
from src.chargement import preparer_chunks_depuis_markdown

DOSSIER_INDEX = Path("index_faiss")
# Modèle multilingue pour bien comprendre le français technique (imputation, MICE)
MODELE_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def creer_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=MODELE_EMBEDDINGS)

def generer_index_vectoriel(force: bool = False) -> FAISS:
    """
    Crée l'index vectoriel. Si force=True, il recrée tout à partir des .md
    """
    # Si l'index existe déjà et qu'on ne force pas, on le charge simplement
    if DOSSIER_INDEX.exists() and not force:
        print("Chargement de l'index existant...")
        return FAISS.load_local(
            str(DOSSIER_INDEX), 
            creer_embeddings(), 
            allow_dangerous_deserialization=True
        )

    print("Construction de l'index structuré par Headers...")
    
    # ÉTAPE 1 : Récupérer les chunks préparés dans chargement.py
    chunks = preparer_chunks_depuis_markdown()
    
    if not chunks:
        raise ValueError("Erreur : Aucun fichier .md trouvé dans le dossier /data")

    # ÉTAPE 2 : Transformer ces chunks en vecteurs (Embeddings)
    embeddings = creer_embeddings()
    index = FAISS.from_documents(chunks, embeddings)
    
    # ÉTAPE 3 : Sauvegarder pour ne pas avoir à tout refaire à chaque fois
    index.save_local(str(DOSSIER_INDEX))
    print(f"✅ Index FAISS sauvegardé avec succès.")
    return index

if __name__ == "__main__":
    # Commande pour créer l'index la première fois
    generer_index_vectoriel(force=True)