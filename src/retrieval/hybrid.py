#!/usr/bin/env python3
"""
Hybrid retrieval: dense similarity fused with BM25 lexical search.

Dense retrieval captures paraphrase but blurs exact tokens; BM25 nails exact terms (a metric
name, a year, a company) but misses paraphrase. Financial questions carry both, so fusing the
two rankings with reciprocal rank fusion (RRF) recovers passages that either method alone
ranks too low.
"""

import re
from typing import List, Dict, Any

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase word/number tokens for BM25."""
    return _TOKEN.findall(text.lower())


def build_bm25(vector_store):
    """
    Build a BM25 index over the chunks already stored in a vector store.

    Args:
        vector_store: A store exposing metadata_map (chunk_id -> {"text": ...})

    Returns:
        Tuple of (bm25 model, list of chunk_ids aligned to the corpus, id->metadata dict)
    """
    from rank_bm25 import BM25Okapi

    meta = vector_store.metadata_map
    chunk_ids = list(meta.keys())
    corpus = [_tokenize(meta[cid].get("text", "")) for cid in chunk_ids]
    return BM25Okapi(corpus), chunk_ids, meta


def hybrid_search(vector_store, bm25_index, query: str, embedding_model: str, k: int,
                  pool: int = 50, rrf_c: int = 60) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks by fusing dense and BM25 rankings with RRF.

    Args:
        vector_store: The vector store to search
        bm25_index: Output of build_bm25 (model, chunk_ids, metadata map)
        query: The question text
        embedding_model: Embedding model name (must match the index)
        k: Number of chunks to return
        pool: Candidate depth taken from each retriever before fusion
        rrf_c: RRF damping constant; larger flattens the rank weighting

    Returns:
        List of retrieved chunk dictionaries, best first
    """
    from src.retrieval.retrieve import search_index

    bm25, chunk_ids, meta = bm25_index

    dense = search_index(vector_store, query, embedding_model, k=pool)
    dense_rank = {c["chunk_id"]: i for i, c in enumerate(dense)}

    scores = bm25.get_scores(_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool]
    bm25_rank = {chunk_ids[i]: r for r, i in enumerate(top)}

    fused = {}
    for cid in set(dense_rank) | set(bm25_rank):
        s = 0.0
        if cid in dense_rank:
            s += 1.0 / (rrf_c + dense_rank[cid])
        if cid in bm25_rank:
            s += 1.0 / (rrf_c + bm25_rank[cid])
        fused[cid] = s

    ranked = sorted(fused, key=lambda c: fused[c], reverse=True)[:k]
    dense_by_id = {c["chunk_id"]: c for c in dense}
    results = []
    for cid in ranked:
        if cid in dense_by_id:
            results.append(dense_by_id[cid])
        else:
            m = meta[cid]
            results.append({
                "chunk_id": cid, "score": fused[cid], "text": m.get("text", ""),
                "doc_name": m.get("doc_name", ""), "company": m.get("company", ""),
                "chunk_strategy": m.get("chunk_strategy", ""),
                "chunk_index": m.get("chunk_index", 0), "metadata": m,
            })
    return results
