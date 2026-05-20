"""Hybrid retrieval: vector + BM25 → RRF merge → cross-encoder rerank (Step 3)."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.retrieval.embedder import embed
from src.retrieval.indexer import IndexManager

_RRF_K = 60
_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(_RERANK_MODEL)
    return _cross_encoder


def retrieve(
    query: str,
    doc_id: str,
    index_manager: IndexManager,
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[dict]:
    """Hybrid retrieve: dual search → RRF merge → cross-encoder rerank.

    Returns top_k results, each dict containing:
        chunk_id, chunk_text, region_indices, bboxes, page,
        rrf_score, rerank_score
    """
    # Stage 1: dual retrieval (scoped strictly to doc_id)
    query_embedding = embed(query)
    query_tokens = query.lower().split()

    vector_results = index_manager.vector_search(query_embedding, doc_id, top_k=candidate_k)
    bm25_results = index_manager.bm25_search(query_tokens, doc_id, top_k=candidate_k)

    # Stage 2: Reciprocal Rank Fusion
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(vector_results):
        cid = chunk["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        chunk_map.setdefault(cid, chunk)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:candidate_k]

    # Stage 3: cross-encoder rerank (cross-attention, not cosine similarity)
    cross_encoder = _get_cross_encoder()
    pairs = [(query, chunk_map[cid]["chunk_text"]) for cid, _ in merged]
    rerank_scores = cross_encoder.predict(pairs).tolist()

    ranked = sorted(
        zip(merged, rerank_scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    results = []
    for (cid, rrf_score), rerank_score in ranked:
        chunk = chunk_map[cid]
        results.append({
            "chunk_id": cid,
            "chunk_text": chunk["chunk_text"],
            "region_indices": chunk.get("region_indices", []),
            "bboxes": chunk.get("bboxes", []),
            "page": chunk.get("page"),
            "rrf_score": rrf_score,
            "rerank_score": float(rerank_score),
        })

    return results
