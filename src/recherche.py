from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.indexation import generer_index_vectoriel
from src.rerank import rerank_docs


# ---------------------------
# Réglages (notebook-like)
# ---------------------------
K_FINAL = 4
CANDIDATES_DENSE = 40
CANDIDATES_SPARSE = 40
RRF_TOP_N = 30
RERANK_TOP_N = 12
RRF_K = 60


# ---------------------------
# Tokenisation BM25 légère
# ---------------------------
_STOP = {
    "le","la","les","un","une","des","de","du","d","et","ou","a","à","au","aux",
    "en","dans","sur","pour","par","avec","sans","ce","cet","cette","ces",
    "je","tu","il","elle","on","nous","vous","ils","elles",
    "qu","que","qui","quoi","dont","où",
    "est","sont","etre","être","fait","faire",
}

def _tokenize(txt: str) -> List[str]:
    txt = txt.lower().replace("’", "'").strip()
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", txt)
    return [t for t in tokens if len(t) > 2 and t not in _STOP]


# ---------------------------
# Résultat
# ---------------------------
@dataclass
class ResultatRecherche:
    source: str
    header1: str
    header2: str
    contenu: str


# ---------------------------
# Cache (évite de tout recalculer)
# ---------------------------
_INDEX: FAISS | None = None
_DOCS: List[Document] | None = None
_BM25: BM25Okapi | None = None


def _load_index_and_docs() -> Tuple[FAISS, List[Document]]:
    """
    Charge FAISS (depuis disque si déjà construit).
    Récupère les documents depuis le docstore FAISS (ne relit pas les .md).
    """
    global _INDEX, _DOCS
    if _INDEX is None:
        _INDEX = generer_index_vectoriel(force=False)
    if _DOCS is None:
        store_dict = _INDEX.docstore._dict  # type: ignore[attr-defined]
        _DOCS = list(store_dict.values())
    return _INDEX, _DOCS


def _get_bm25() -> BM25Okapi:
    """Construit BM25 une seule fois sur les docs FAISS."""
    global _BM25
    if _BM25 is None:
        _, docs = _load_index_and_docs()
        corpus = [_tokenize(d.page_content) for d in docs]
        _BM25 = BM25Okapi(corpus)
    return _BM25


# ---------------------------
# Hybrid retrieval
# ---------------------------
def _dense_candidates(index: FAISS, question: str, k: int) -> List[Document]:
    return index.similarity_search(question, k=k)


def _sparse_candidates(question: str, k: int) -> List[int]:
    bm25 = _get_bm25()
    _, docs = _load_index_and_docs()
    scores = bm25.get_scores(_tokenize(question))
    ranked = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
    return ranked[:k]


def _rrf_fusion(
    dense_docs: List[Document],
    sparse_idx: List[int],
    all_docs: List[Document],
    top_n: int,
    rrf_k: int = RRF_K,
) -> List[Document]:
    scores: Dict[int, float] = {}

    def key(d: Document) -> Tuple[str, str, str, str]:
        md = d.metadata or {}
        return (md.get("source",""), md.get("header1",""), md.get("header2",""), d.page_content)

    all_map = {key(d): i for i, d in enumerate(all_docs)}

    for rank, d in enumerate(dense_docs):
        idx = all_map.get(key(d))
        if idx is not None:
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    for rank, idx in enumerate(sparse_idx):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    ranked_idx = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_n]
    return [all_docs[i] for i in ranked_idx]


# ---------------------------
# Ancrage entités (simple)
# ---------------------------
def _anchors(question: str) -> List[str]:
    q = question.lower()
    anc = []
    for a in ["insee", "ehesp", "ubo", "rennes", "rennes 2", "eneam", "bsic", "dspssel"]:
        if a in q:
            anc.append(a)
    return anc


def _prioritize_by_anchors(docs: List[Document], anchors: List[str]) -> List[Document]:
    if not anchors:
        return docs

    def has_anchor(d: Document) -> bool:
        text = (d.page_content + " " + str(d.metadata)).lower()
        return any(a in text for a in anchors)

    prior = [d for d in docs if has_anchor(d)]
    rest = [d for d in docs if not has_anchor(d)]
    return prior + rest


# ---------------------------
# Fonction principale
# ---------------------------
def rechercher(
    question: str,
    k_final: int = K_FINAL,
    candidates_dense: int = CANDIDATES_DENSE,
    candidates_sparse: int = CANDIDATES_SPARSE,
    rrf_top_n: int = RRF_TOP_N,
    rerank_top_n: int = RERANK_TOP_N,
) -> List[ResultatRecherche]:
    index, all_docs = _load_index_and_docs()

    dense = _dense_candidates(index, question, candidates_dense)
    sparse_idx = _sparse_candidates(question, candidates_sparse)

    candidats = _rrf_fusion(dense, sparse_idx, all_docs, top_n=rrf_top_n)
    candidats = _prioritize_by_anchors(candidats, _anchors(question))

    # 👉 Rerank (sans répéter la logique, on appelle rerank.py)
    n = min(rerank_top_n, len(candidats))
    if n > 0:
        order = rerank_docs(question, [d.page_content for d in candidats[:n]])
        candidats = [candidats[i] for i in order] + candidats[n:]

    selection = candidats[:k_final]

    resultats: List[ResultatRecherche] = []
    for d in selection:
        md = d.metadata or {}
        resultats.append(
            ResultatRecherche(
                source=md.get("source", "inconnu"),
                header1=md.get("header1", "inconnu"),
                header2=md.get("header2", "inconnu"),
                contenu=d.page_content,
            )
        )

    return resultats


if __name__ == "__main__":
    q = "Qu’ai-je fait à l’INSEE ?"
    res = rechercher(q)
    print(f"Question : {q}\n")
    for i, r in enumerate(res, start=1):
        print(f"[{i}] {r.source} | {r.header1} > {r.header2}")
        print(r.contenu[:350], "...\n")