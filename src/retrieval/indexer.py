"""Dual indexing: MongoDB Atlas vector store + BM25 in-memory (Step 2)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from pymongo import MongoClient
from pymongo.collection import Collection
from rank_bm25 import BM25Okapi

from src.retrieval.embedder import embed_batch

_CHUNKS_COLLECTION = "doc_chunks"
_CITATIONS_COLLECTION = "citation_index"


@dataclass
class _BM25State:
    bm25: BM25Okapi
    chunk_ids: list[str]
    corpus: list[list[str]]


class IndexManager:
    """Manages the vector index (MongoDB) and BM25 index (in-memory) per doc_id.

    Both indexes are scoped to doc_id. Cross-document retrieval is impossible:
    vector search uses a filter and BM25 states are keyed by doc_id.
    """

    def __init__(self) -> None:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise EnvironmentError("MONGODB_URI not set. Add it to .env.")
        db_name = os.environ.get("MONGODB_DB", "grounded_retrieval")
        self._client: MongoClient = MongoClient(uri)
        self._db = self._client[db_name]
        self._chunks: Collection = self._db[_CHUNKS_COLLECTION]
        self._citations: Collection = self._db[_CITATIONS_COLLECTION]
        self._bm25_cache: dict[str, _BM25State] = {}

    # ── Vector + BM25 indexing ────────────────────────────────────────────────

    def index_chunks(self, chunks: list[dict]) -> None:
        """Embed and store chunks in MongoDB; build BM25 for their doc_id."""
        if not chunks:
            return

        doc_id = chunks[0]["doc_id"]

        # Embed contextualized_text (NOT chunk_text)
        texts_to_embed = [c["contextualized_text"] for c in chunks]
        embeddings = embed_batch(texts_to_embed)

        docs = []
        for chunk, embedding in zip(chunks, embeddings):
            docs.append({
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "region_indices": chunk["region_indices"],
                "chunk_text": chunk["chunk_text"],
                "contextualized_text": chunk["contextualized_text"],
                "embedding": embedding,
                "bboxes": chunk["bboxes"],
                "page": chunk["page"],
            })

        # Upsert by chunk_id so re-indexing is idempotent
        for doc in docs:
            self._chunks.replace_one(
                {"chunk_id": doc["chunk_id"]},
                doc,
                upsert=True,
            )

        self._build_bm25(doc_id, docs)

    def rebuild_bm25(self, doc_id: str) -> None:
        """Reconstruct the in-memory BM25 index for a doc_id from MongoDB."""
        stored = list(self._chunks.find({"doc_id": doc_id}))
        if not stored:
            raise ValueError(f"No chunks found in MongoDB for doc_id={doc_id!r}")
        self._build_bm25(doc_id, stored)

    # ── Vector search ─────────────────────────────────────────────────────────

    def vector_search(
        self,
        query_embedding: list[float],
        doc_id: str,
        top_k: int = 20,
        index_name: str = "vector_index",
    ) -> list[dict]:
        """Run MongoDB Atlas $vectorSearch filtered to doc_id."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                    "filter": {"doc_id": doc_id},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "chunk_text": 1,
                    "region_indices": 1,
                    "bboxes": 1,
                    "page": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return list(self._chunks.aggregate(pipeline))

    # ── BM25 search ───────────────────────────────────────────────────────────

    def bm25_search(
        self,
        query_tokens: list[str],
        doc_id: str,
        top_k: int = 20,
    ) -> list[dict]:
        """BM25 search over the in-memory index for doc_id."""
        if doc_id not in self._bm25_cache:
            self.rebuild_bm25(doc_id)

        state = self._bm25_cache[doc_id]
        scores = state.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            chunk_id = state.chunk_ids[idx]
            doc = self._chunks.find_one(
                {"chunk_id": chunk_id},
                {"_id": 0, "chunk_id": 1, "chunk_text": 1, "region_indices": 1, "bboxes": 1, "page": 1},
            )
            if doc:
                results.append({**doc, "bm25_score": float(score)})
        return results

    # ── Citation index ────────────────────────────────────────────────────────

    def store_citations(self, doc_id: str, citation_index: dict) -> None:
        self._citations.replace_one(
            {"doc_id": doc_id},
            {"doc_id": doc_id, "index": citation_index},
            upsert=True,
        )

    def load_citations(self, doc_id: str) -> dict:
        record = self._citations.find_one({"doc_id": doc_id}, {"_id": 0, "index": 1})
        return record["index"] if record else {}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_bm25(self, doc_id: str, docs: list[dict]) -> None:
        corpus = [doc["contextualized_text"].lower().split() for doc in docs]
        chunk_ids = [doc["chunk_id"] for doc in docs]
        self._bm25_cache[doc_id] = _BM25State(
            bm25=BM25Okapi(corpus),
            chunk_ids=chunk_ids,
            corpus=corpus,
        )
