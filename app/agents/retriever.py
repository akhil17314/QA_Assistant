"""Page-aware similarity retrieval from the local Chroma collection."""

from functools import lru_cache
from typing import Any

from app.config import settings
from app.ingest import get_vector_store


@lru_cache(maxsize=1)
def _get_reranker() -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def has_documents() -> bool:
    """Return whether the configured Chroma collection contains documents."""
    try:
        return get_vector_store()._collection.count() > 0
    except Exception:
        return False


def indexed_sources() -> list[str]:
    """Return distinct source names currently present in the index."""
    try:
        metadata = get_vector_store()._collection.get(include=["metadatas"]).get("metadatas") or []
    except Exception:
        return []
    return sorted({item["source"] for item in metadata if item and item.get("source")})


def retrieve(query: str, k: int | None = None) -> list[dict]:
    """Return text, source, page, and relevance score for the best matching chunks."""
    count = k or settings.top_k
    if not query.strip() or count < 1:
        return []
    try:
        store = get_vector_store()
        if store._collection.count() == 0:
            return []
        matches = store.similarity_search_with_relevance_scores(
            query, k=count * 3 if settings.use_reranker else count
        )
    except Exception:
        return []

    results = [
        {
            "text": document.page_content,
            "source": str(document.metadata.get("source", "unknown")),
            "page": int(document.metadata.get("page", 1)),
            "score": float(score),
        }
        for document, score in matches
    ]
    if settings.use_reranker and results:
        reranker = _get_reranker()
        scores = reranker.predict([(query, item["text"]) for item in results])
        for item, score in zip(results, scores):
            item["score"] = float(score)
        results.sort(key=lambda item: item["score"], reverse=True)
    return results[:count]
