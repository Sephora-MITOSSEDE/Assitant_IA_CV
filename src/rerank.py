from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from sentence_transformers import CrossEncoder


MODELE_RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Charge le modèle une seule fois."""
    return CrossEncoder(MODELE_RERANK)


def rerank_docs(question: str, passages: List[str]) -> List[int]:
    """
    Retourne les indices des passages triés par pertinence décroissante.
    """
    model = get_reranker()
    pairs: List[Tuple[str, str]] = [(question, p) for p in passages]
    scores = model.predict(pairs)
    return sorted(range(len(passages)), key=lambda i: float(scores[i]), reverse=True)