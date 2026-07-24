#!/usr/bin/env python3
"""
Cross-encoder reranking over a deep candidate pool.

Dense retrieval scores a question and a passage independently, so it ranks on broad topical
similarity and cannot tell which of twenty passages about the same company and year holds the
figure being asked for. A cross-encoder reads the pair together and scores them jointly, which
is slower but far better at that final ordering.

The gain is bounded by what the pool contains: recall keeps climbing well past k=10 on this
corpus (Recall@50 is 0.80 against 0.51 at k=5 under the company filter), so retrieving deep and
reordering recovers evidence that dense search puts just out of reach.
"""

from typing import List, Dict, Any, Optional


def build_reranker(model_name: str = "BAAI/bge-reranker-base", max_length: int = 512):
    """
    Load a cross-encoder, preferring the Apple GPU when present.

    Args:
        model_name: sentence-transformers cross-encoder name
        max_length: token budget for the concatenated question and passage

    Returns:
        A loaded CrossEncoder
    """
    from sentence_transformers import CrossEncoder

    device = None
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    except Exception:
        device = None

    return CrossEncoder(model_name, max_length=max_length, device=device)


def rerank_search(vector_store, reranker, query: str, embedding_model: str, k: int,
                  pool: int = 50, companies: Optional[List[str]] = None,
                  batch_size: int = 64) -> List[Dict[str, Any]]:
    """
    Retrieve a deep pool, reorder it with the cross-encoder, and return the top k.

    Args:
        vector_store: The vector store to search
        reranker: Output of build_reranker
        query: The question text
        embedding_model: Embedding model name (must match the index)
        k: Number of chunks to return
        pool: Candidate depth fetched before reranking
        companies: When given, restrict the pool to the company named in the question
        batch_size: Pairs scored per forward pass

    Returns:
        List of retrieved chunk dictionaries, best first, carrying the cross-encoder score
    """
    from src.retrieval.retrieve import search_index

    if companies is not None:
        from src.retrieval.metadata_filter import filtered_search
        candidates = filtered_search(vector_store, query, embedding_model, pool, companies)
    else:
        candidates = search_index(vector_store, query, embedding_model, k=pool)

    if not candidates:
        return []

    scores = reranker.predict([[query, c.get("text", "")] for c in candidates],
                              batch_size=batch_size, show_progress_bar=False)

    order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
    ranked = []
    for i in order[:k]:
        c = dict(candidates[i])
        c["dense_score"] = c.get("score")
        c["score"] = float(scores[i])
        ranked.append(c)
    return ranked
