#!/usr/bin/env python3
"""
Query-side metadata filtering for retrieval.

FinanceBench questions name their target company and year in the question text ("... FY2018
capital expenditure for 3M"), and every chunk carries a company and period in its metadata.
Restricting the search to the chunks of the company named in the question removes most of the
cross-company distractors that a pure dense search ranks above the right passage.

The company is inferred only from the question text, never from the gold answer, so this is a
retrieval strategy rather than a leak.
"""

import json
import re
from typing import List, Dict, Any, Optional

# Short forms that appear in questions but not in the catalogue company names.
COMPANY_ALIASES = {
    "amex": "American Express",
    "jnj": "Johnson & Johnson",
    "j&j": "Johnson & Johnson",
    "coke": "Coca-Cola",
    "p&g": "Procter & Gamble",
}


def _norm(s: str) -> str:
    """Lowercase and collapse non-alphanumeric runs to single spaces."""
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def load_company_list(doc_info_path: str) -> List[str]:
    """Return the distinct company names in the document catalogue."""
    companies = set()
    for line in open(doc_info_path):
        companies.add(json.loads(line)["company"])
    return sorted(companies)


def match_company(question: str, companies: List[str]) -> Optional[str]:
    """
    Infer the company a question is about from its text.

    Args:
        question: The question text
        companies: Candidate company names from the catalogue

    Returns:
        The matched company name, or None if nothing matched
    """
    qn = _norm(question)
    best, best_len = None, 0
    for company in companies:
        cn = _norm(company)
        if re.search(r"\b" + re.escape(cn) + r"\b", qn) and len(cn) > best_len:
            best, best_len = company, len(cn)
    if best is None:
        for alias, company in COMPANY_ALIASES.items():
            if re.search(r"\b" + re.escape(_norm(alias)) + r"\b", qn):
                return company
    return best


def filtered_search(vector_store, query: str, embedding_model: str, k: int,
                    companies: List[str], pool: int = 3000) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks restricted to the company named in the query.

    Retrieves a large candidate pool by dense similarity, keeps only the chunks whose company
    matches the one inferred from the question, and returns the top-k of those. Falls back to
    the unfiltered top-k when no company is matched.

    Args:
        vector_store: The vector store to search
        query: The question text
        embedding_model: Embedding model name (must match the index)
        k: Number of chunks to return
        companies: Candidate company names
        pool: Candidate-pool size fetched before filtering

    Returns:
        List of retrieved chunk dictionaries, best first
    """
    from src.retrieval.retrieve import search_index

    company = match_company(query, companies)
    if company is None:
        return search_index(vector_store, query, embedding_model, k=k)

    pool = min(pool, vector_store.index.ntotal) if hasattr(vector_store, "index") else pool
    candidates = search_index(vector_store, query, embedding_model, k=pool)
    matched = [c for c in candidates if c.get("company") == company]
    if not matched:
        return candidates[:k]
    return matched[:k]
