#!/usr/bin/env python3
"""
Evaluation module for FinanceBench RAG pipeline.
Handles metrics calculation, evaluation pipeline, and result aggregation.
"""

import json
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from rapidfuzz import fuzz, distance
from rouge_score import rouge_scorer
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def json_safe(obj: Any) -> Any:
    """
    Convert numpy scalars and arrays into plain Python types.

    Counts and means come back from numpy as int64 and float64, which the json module
    refuses to serialise.

    Args:
        obj: Any nested structure of dicts, lists and scalars

    Returns:
        The same structure using built-in types
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# Refusals in the model's own words, not just the wording it was told to use. Instructed to
# reply "Not enough information in the provided context", it often declines some other way
# ("The provided excerpts do not contain specific figures for..."). Matching only the exact
# string undercounts refusals, which made a prompt change look like it had reduced abstention
# when it had only rephrased it.
_REFUSAL_PATTERN = re.compile(
    r"not enough information"
    r"|do(?:es)? not (?:contain|provide|specify|include|state|identify|disclose)"
    r"|no (?:specific )?(?:information|figures?|data|mention|details)"
    r"|not (?:stated|provided|specified|available|disclosed|mentioned) (?:in|within|by)"
    r"|cannot (?:be )?(?:determined|calculated|answered?)",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    """
    Whether an answer declines to answer, however it is worded.

    Only the opening sentence is examined. A refusal is how an answer starts; the same wording
    later in a reply is usually a caveat attached to a real answer ("Revenue was $5,412 million
    [X_c12]. The excerpts do not contain the FY2019 comparison."), which should not count.

    Args:
        answer: Generated answer text

    Returns:
        True if the answer opens by declining
    """
    if not answer:
        return False
    opening = re.split(r'(?<=[.!?])\s', str(answer).strip(), maxsplit=1)[0]
    return bool(_REFUSAL_PATTERN.search(opening))


_EVIDENCE_STOPWORDS = set(
    "the a an and or but in on at to for of with by is are was were be as from that this it "
    "its their his her our your they we you i he she has have had will would can could".split()
)


def _content_tokens(text: str) -> set:
    """
    Reduce text to its content words for overlap scoring.

    Keeps alphanumeric tokens longer than three characters that are not common stopwords, so
    that figures and named terms drive the match rather than filler.

    Args:
        text: Raw text

    Returns:
        Set of content tokens
    """
    return {w for w in re.findall(r'[a-z0-9]+', text.lower())
            if len(w) > 3 and w not in _EVIDENCE_STOPWORDS}


def evidence_overlap(evidence: str, chunk_text: str) -> float:
    """
    Fraction of the evidence's content words that appear in a retrieved chunk.

    FinanceBench evidence passages are often longer than a single chunk (they can span a whole
    table or page), so a symmetric string ratio understates a genuine hit. Coverage of the
    evidence's content words by the chunk is robust to that length gap: a chunk that contains
    the figures and terms of the evidence scores high regardless of how much extra text either
    side carries.

    Args:
        evidence: Ground-truth evidence text
        chunk_text: Retrieved chunk text

    Returns:
        Coverage in [0, 1]
    """
    ev = _content_tokens(evidence)
    if not ev:
        return 0.0
    return len(ev & _content_tokens(chunk_text)) / len(ev)


class RetrievalEvaluator:
    """Evaluates retrieval performance."""

    @staticmethod
    def calculate_recall_at_k(retrieved_chunks: List[Dict[str, Any]],
                              ground_truth_evidence: List[str],
                              k: int = 5,
                              threshold: float = 0.5) -> float:
        """
        Calculate Recall@k for retrieval.

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            ground_truth_evidence: List of ground truth evidence strings
            k: Number of top results to consider
            threshold: Similarity threshold for considering a match

        Returns:
            Recall@k score
        """
        if not ground_truth_evidence:
            return 0.0

        # Take top-k chunks
        top_k_chunks = retrieved_chunks[:k]
        top_k_texts = [chunk.get('text', '').strip() for chunk in top_k_chunks]

        # Check how many ground truth evidence items are found in top-k
        matches = 0
        for evidence in ground_truth_evidence:
            evidence_found = False
            for chunk_text in top_k_texts:
                # Use fuzzy matching for text comparison
                similarity = evidence_overlap(evidence, chunk_text)
                if similarity >= threshold:
                    evidence_found = True
                    break
            if evidence_found:
                matches += 1

        return matches / len(ground_truth_evidence)

    @staticmethod
    def calculate_precision_at_k(retrieved_chunks: List[Dict[str, Any]],
                                 ground_truth_evidence: List[str],
                                 k: int = 5,
                                 threshold: float = 0.5) -> float:
        """
        Calculate Precision@k for retrieval.

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            ground_truth_evidence: List of ground truth evidence strings
            k: Number of top results to consider
            threshold: Similarity threshold for considering a match

        Returns:
            Precision@k score
        """
        if k == 0 or not retrieved_chunks:
            return 0.0

        # Take top-k chunks
        top_k_chunks = retrieved_chunks[:k]
        top_k_texts = [chunk.get('text', '').strip() for chunk in top_k_chunks]

        # Count how many retrieved chunks contain relevant evidence
        relevant_count = 0
        for chunk_text in top_k_texts:
            chunk_relevant = False
            for evidence in ground_truth_evidence:
                similarity = evidence_overlap(evidence, chunk_text)
                if similarity >= threshold:
                    chunk_relevant = True
                    break
            if chunk_relevant:
                relevant_count += 1

        # Divide by k, not by how many chunks came back. A run that retrieves 5 passages has a
        # precision@10 of at most 0.5 by definition; dividing by 5 reported it as if 10 had been
        # examined, which doubled the score.
        return relevant_count / k

    @staticmethod
    def calculate_mrr(retrieved_chunks: List[Dict[str, Any]],
                      ground_truth_evidence: List[str],
                      threshold: float = 0.5) -> float:
        """
        Calculate Mean Reciprocal Rank.

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            ground_truth_evidence: List of ground truth evidence strings
            threshold: Similarity threshold for considering a match

        Returns:
            MRR score
        """
        if not ground_truth_evidence:
            return 0.0

        ranks = []
        for evidence in ground_truth_evidence:
            rank = None
            for i, chunk in enumerate(retrieved_chunks):
                chunk_text = chunk.get('text', '').strip()
                similarity = evidence_overlap(evidence, chunk_text)
                if similarity >= threshold:
                    rank = i + 1  # 1-indexed rank
                    break
            if rank is not None:
                ranks.append(1.0 / rank)
            else:
                ranks.append(0.0)  # Not found

        return np.mean(ranks) if ranks else 0.0


class GenerationEvaluator:
    """Evaluates generation performance."""

    def __init__(self):
        """Initialize generation evaluator."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def calculate_rouge_l(self, generated_text: str,
                          reference_text: str) -> float:
        """
        Calculate ROUGE-L score.

        Args:
            generated_text: Generated answer text
            reference_text: Ground truth answer text

        Returns:
            ROUGE-L F1 score
        """
        scores = self.rouge_scorer.score(reference_text, generated_text)
        return scores['rougeL'].fmeasure

    def calculate_semantic_similarity(self, generated_text: str,
                                      reference_text: str,
                                      model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> float:
        """
        Calculate semantic similarity using sentence transformers.

        Args:
            generated_text: Generated answer text
            reference_text: Ground truth answer text
            model_name: Name of sentence-transformers model to use

        Returns:
            Cosine similarity score
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Load once, keep on CPU: it encodes two short strings per call so the GPU buys
            # nothing, and a fresh MPS load can crash after a generator has used Metal.
            if getattr(GenerationEvaluator, "_semantic_model", None) is None:
                GenerationEvaluator._semantic_model = SentenceTransformer(model_name, device="cpu")
            model = GenerationEvaluator._semantic_model

            # Encode texts
            gen_embedding = model.encode([generated_text])
            ref_embedding = model.encode([reference_text])

            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(gen_embedding, ref_embedding)[0][0]
            return float(similarity)
        except ImportError:
            logger.warning("sentence-transformers or scikit-learn not available for semantic similarity")
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0

    @staticmethod
    def _figures(text: str) -> List[float]:
        """
        Comparable numeric values in free text, handling $ , % ( ) and negatives.

        Citation tags are stripped first, and a figure whose digits touch a letter is skipped so
        that 3M, 10K, FY2016 and c367 are not read as quantities. The letter test anchors on the
        first digit because the pattern may absorb a leading "$", "(" or space.

        Args:
            text: Any answer or reference string

        Returns:
            The figures found, in order of appearance
        """
        text = re.sub(r'\[[^\]]*\]', ' ', str(text))
        out = []
        for m in re.finditer(r'[-+]?\$?\s?\(?\d[\d,]*\.?\d*\)?%?', text):
            raw = m.group()
            first_digit = next((i for i, ch in enumerate(raw) if ch.isdigit()), None)
            if first_digit is None:
                continue
            d = m.start() + first_digit
            before = text[d - 1] if d > 0 else " "
            after = text[m.end()] if m.end() < len(text) else " "
            if before.isalpha() or after.isalpha():
                continue
            neg = "(" in raw or raw.lstrip().startswith("-")
            s = re.sub(r'[^\d.]', '', raw).strip('.')
            if not s:
                continue
            try:
                value = float(s)
            except ValueError:
                continue
            out.append(-value if neg else value)
        return out

    def calculate_numeric_agreement(self, generated_text: str, reference_text: str,
                                    rel_tol: float = 0.005) -> Optional[float]:
        """
        Whether a figure from the reference answer appears in the generated answer.

        ROUGE-L cannot score these questions: FinanceBench answers are frequently bare figures
        such as "$1577.00", so a correct answer written as a sentence scores near zero on word
        overlap. This compares the figures instead.

        Fiscal years are dropped from both sides, since nearly every question and answer names
        one and leaving them in matches "FY2023" against "FY2023" and scores a wrong answer as
        correct. The tolerance is tight for the same reason: a loose one lets the 31 in
        "December 31" match a reference value of 30.8. A 1000x difference is accepted, because
        filings quote millions and answers sometimes restate in billions.

        Args:
            generated_text: Generated answer text
            reference_text: Ground truth answer text
            rel_tol: Relative tolerance for a match

        Returns:
            1.0 on agreement, 0.0 on disagreement, or None when the reference answer carries no
            figure and the question is therefore not scorable this way
        """
        def not_year(v):
            return not (float(v).is_integer() and 1900 <= v <= 2100)

        gold = [v for v in self._figures(reference_text) if not_year(v)]
        gen = [v for v in self._figures(generated_text) if not_year(v)]
        if not gold:
            return None
        if not gen:
            return 0.0
        for a in gold:
            for b in gen:
                denom = max(abs(a), 1e-9)
                if a == b or abs(a - b) / denom <= rel_tol:
                    return 1.0
                for scale in (1000.0, 0.001, 1e6, 1e-6):
                    if abs(a - b * scale) / denom <= rel_tol:
                        return 1.0
        return 0.0

    def calculate_exact_match(self, generated_text: str,
                              reference_text: str) -> float:
        """
        Calculate exact match score (normalized).

        Args:
            generated_text: Generated answer text
            reference_text: Ground truth answer text

        Returns:
            1.0 if exact match (after normalization), 0.0 otherwise
        """
        # Normalize texts: lowercase, strip whitespace, remove extra spaces
        def normalize(text):
            return re.sub(r'\s+', ' ', text.lower().strip())

        return 1.0 if normalize(generated_text) == normalize(reference_text) else 0.0


class CitationEvaluator:
    """Evaluates citation performance."""

    CITATION_PATTERN = r'\[[^\]]+_[cp]\d+\]'

    @staticmethod
    def extract_citations(answer: str) -> List[str]:
        """
        Pull the citation tags out of an answer.

        Args:
            answer: Generated answer text

        Returns:
            Citation tags in the order they appear, including brackets
        """
        return re.findall(CitationEvaluator.CITATION_PATTERN, answer)

    @staticmethod
    def valid_citation_keys(contexts: List[Dict[str, Any]]) -> set:
        """
        Build the set of citation tags that the retrieved contexts justify.

        Args:
            contexts: List of retrieved context dictionaries

        Returns:
            Set of valid citation tags
        """
        keys = set()
        for ctx in contexts:
            doc_name = ctx.get('doc_name', 'unknown')
            chunk_index = ctx.get('chunk_index')
            page_number = ctx.get('evidence_page_num', ctx.get('page_number'))
            if chunk_index is not None:
                keys.add(f"[{doc_name}_c{chunk_index}]")
            if page_number is not None:
                keys.add(f"[{doc_name}_p{page_number}]")
        return keys

    @staticmethod
    def validate_citations(answer: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check an answer's citations against the contexts it was given.

        Args:
            answer: Generated answer text
            contexts: List of retrieved context dictionaries

        Returns:
            Counts of total and valid citations, and whether every citation is valid
        """
        citations = CitationEvaluator.extract_citations(answer)
        valid_keys = CitationEvaluator.valid_citation_keys(contexts)
        valid = [c for c in citations if c in valid_keys]
        return {
            "num_citations": len(citations),
            "num_valid_citations": len(valid),
            "all_citations_valid": len(citations) > 0 and len(valid) == len(citations),
        }

    @staticmethod
    def citation_precision(answer: str, contexts: List[Dict[str, Any]]) -> float:
        """
        Calculate citation precision: proportion of citations that are valid.

        Args:
            answer: Generated answer text
            contexts: List of retrieved context dictionaries

        Returns:
            Citation precision score
        """
        # Extract citations (assuming format [docname_cindex] or [docname_pagenum])
        citation_pattern = r'\[[^\]]+_[cp]\d+\]'
        citations = re.findall(citation_pattern, answer)

        if not citations:
            return 0.0

        # Build set of valid citation keys
        valid_citations = set()
        for ctx in contexts:
            doc_name = ctx.get('doc_name', 'unknown')
            chunk_index = ctx.get('chunk_index')
            page_number = ctx.get('evidence_page_num', ctx.get('page_number'))

            if chunk_index is not None:
                valid_citations.add(f"[{doc_name}_c{chunk_index}]")
            if page_number is not None:
                valid_citations.add(f"[{doc_name}_p{page_number}]")

        # Count valid citations
        valid_count = sum(1 for cite in citations if cite in valid_citations)
        return valid_count / len(citations)

    @staticmethod
    def citation_recall(answer: str, contexts: List[Dict[str, Any]]) -> float:
        """
        Calculate citation recall: proportion of relevant context that is cited.
        Simplified version: checks if answer contains information from contexts.

        Args:
            answer: Generated answer text
            contexts: List of retrieved context dictionaries

        Returns:
            Citation recall score (simplified)
        """
        if not contexts:
            return 0.0

        # Check if answer contains key information from contexts
        answer_lower = answer.lower()
        cited_info = 0
        total_info = len(contexts)

        for ctx in contexts:
            text = ctx.get('text', '').lower().strip()
            if text and len(text) > 10:  # Only consider substantial text chunks
                # Check if any significant portion of the chunk appears in answer
                # Simple approach: check if some keywords from chunk are in answer
                # Keep the chunk's own word order and drop repeats. Building the list from a
                # set instead made this metric non-deterministic: Python randomises string
                # hashing per process, so "the first five keywords" was a different five on
                # every run and the same answer scored differently each time. That drift was
                # previously mistaken for the llama.cpp Metal backend being irreproducible.
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                keywords = [w for w in dict.fromkeys(text.split())
                            if len(w) > 3 and w not in stop_words]

                if keywords:
                    # Check if at least some keywords appear in answer
                    matches = sum(1 for w in keywords[:5] if w in answer_lower)  # Check first 5 keywords
                    if matches > 0:
                        cited_info += 1

        return cited_info / total_info if total_info > 0 else 0.0

    @staticmethod
    def citation_f1(answer: str, contexts: List[Dict[str, Any]]) -> float:
        """
        Calculate F1 score for citations.

        Args:
            answer: Generated answer text
            contexts: List of retrieved context dictionaries

        Returns:
            Citation F1 score
        """
        precision = CitationEvaluator.citation_precision(answer, contexts)
        recall = CitationEvaluator.citation_recall(answer, contexts)

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class RAGEvaluator:
    """Main RAG evaluation orchestrator."""

    def __init__(self):
        """Initialize RAG evaluator."""
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.citation_evaluator = CitationEvaluator()

    def evaluate_single_example(self, question: str, answer: str,
                                retrieved_chunks: List[Dict[str, Any]],
                                ground_truth_answer: str,
                                ground_truth_evidence: List[str],
                                k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Evaluate a single question-answer pair.

        Args:
            question: User's question
            answer: Generated answer
            retrieved_chunks: List of retrieved chunks
            ground_truth_answer: Ground truth answer
            ground_truth_evidence: List of ground truth evidence strings
            k_values: List of k values for Recall@k and Precision@k

        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            "question": question,
            "generated_answer": answer,
            "ground_truth_answer": ground_truth_answer,
            "num_retrieved": len(retrieved_chunks)
        }

        # Retrieval metrics
        for k in k_values:
            metrics[f"recall@{k}"] = self.retrieval_evaluator.calculate_recall_at_k(
                retrieved_chunks, ground_truth_evidence, k=k
            )
            metrics[f"precision@{k}"] = self.retrieval_evaluator.calculate_precision_at_k(
                retrieved_chunks, ground_truth_evidence, k=k
            )

        metrics["mrr"] = self.retrieval_evaluator.calculate_mrr(
            retrieved_chunks, ground_truth_evidence
        )

        # Generation metrics
        metrics["rouge_l"] = self.generation_evaluator.calculate_rouge_l(
            answer, ground_truth_answer
        )
        metrics["exact_match"] = self.generation_evaluator.calculate_exact_match(
            answer, ground_truth_answer
        )
        # None on questions whose reference answer holds no figure, so they are excluded from the
        # mean rather than counted as failures.
        metrics["numeric_agreement"] = self.generation_evaluator.calculate_numeric_agreement(
            answer, ground_truth_answer
        )

        # Semantic similarity (optional, can be slow)
        try:
            metrics["semantic_similarity"] = self.generation_evaluator.calculate_semantic_similarity(
                answer, ground_truth_answer
            )
        except Exception as e:
            logger.warning(f"Could not calculate semantic similarity: {e}")
            metrics["semantic_similarity"] = 0.0

        # Citation metrics
        metrics["citation_precision"] = self.citation_evaluator.citation_precision(
            answer, retrieved_chunks
        )
        metrics["citation_recall"] = self.citation_evaluator.citation_recall(
            answer, retrieved_chunks
        )
        metrics["citation_f1"] = self.citation_evaluator.citation_f1(
            answer, retrieved_chunks
        )

        # Flag an abstention, counting refusals phrased in the model's own words as well as the
        # exact string it was told to use. See is_refusal.
        metrics["has_refusal"] = is_refusal(answer)

        return metrics

    def evaluate_batch(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of examples.

        Args:
            examples: List of example dictionaries, each containing:
                - question
                - generated_answer (or will be generated)
                - ground_truth_answer
                - retrieved_chunks
                - ground_truth_evidence

        Returns:
            List of evaluation result dictionaries
        """
        results = []

        for i, example in enumerate(examples):
            logger.info(f"Evaluating example {i+1}/{len(examples)}")

            # If generated answer not provided, we would need to generate it
            # For now, assume it's provided
            if "generated_answer" not in example:
                logger.warning("No generated answer provided, skipping generation metrics")
                # We could add generation here if we had access to the model

            result = self.evaluate_single_example(
                question=example["question"],
                answer=example.get("generated_answer", ""),
                retrieved_chunks=example["retrieved_chunks"],
                ground_truth_answer=example["ground_truth_answer"],
                ground_truth_evidence=example["ground_truth_evidence"]
            )

            # Add any additional metadata from example
            for key in ["doc_name", "question_type", "company"]:
                if key in example:
                    result[key] = example[key]

            results.append(result)

        return results

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate evaluation results across multiple examples.

        Args:
            results: List of individual evaluation result dictionaries

        Returns:
            Dictionary with aggregated metrics
        """
        if not results:
            return {}

        # Collect metric keys across every result, not just the first. Some metrics are undefined
        # for some questions and report None: numeric agreement cannot score a question whose gold
        # answer holds no figure. Sampling only results[0] made the whole metric appear or vanish
        # depending on which question happened to come first, and mixing a None into the mean
        # aborted the run outright.
        numeric_keys = []
        for r in results:
            for key, value in r.items():
                if key not in numeric_keys and isinstance(value, (int, float)) \
                        and not isinstance(value, bool):
                    numeric_keys.append(key)

        aggregated = {}
        for key in numeric_keys:
            values = [r[key] for r in results
                      if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
                aggregated[f"{key}_median"] = np.median(values)
                aggregated[f"{key}_min"] = np.min(values)
                aggregated[f"{key}_max"] = np.max(values)
                # Record the denominator when a metric skipped questions, so a mean over a subset
                # is never mistaken for a mean over the whole run.
                if len(values) < len(results):
                    aggregated[f"{key}_n"] = len(values)

        # Add count
        aggregated["num_examples"] = len(results)

        return aggregated

    def save_results(self, results: List[Dict[str, Any]],
                     aggregated: Dict[str, Any],
                     output_dir: str = "results"):
        """
        Save evaluation results to disk.

        Args:
            results: List of individual evaluation results
            aggregated: Aggregated metrics dictionary
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_df = pd.DataFrame(results)
        results_path = os.path.join(output_dir, f"detailed_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved detailed results to {results_path}")

        # Save aggregated results
        aggregated_path = os.path.join(output_dir, f"aggregated_results_{timestamp}.json")
        with open(aggregated_path, 'w') as f:
            json.dump(json_safe(aggregated), f, indent=2)
        logger.info(f"Saved aggregated results to {aggregated_path}")

        # Save as CSV too for easy viewing
        aggregated_df = pd.DataFrame([aggregated])
        aggregated_csv_path = os.path.join(output_dir, f"aggregated_results_{timestamp}.csv")
        aggregated_df.to_csv(aggregated_csv_path, index=False)
        logger.info(f"Saved aggregated results CSV to {aggregated_csv_path}")

        # Generate some basic plots
        try:
            self._plot_metrics(results, output_dir, timestamp)
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")

    def _plot_metrics(self, results: List[Dict[str, Any]],
                      output_dir: str, timestamp: str):
        """Generate basic plots of evaluation metrics."""
        plt.style.use('seaborn-v0_8')

        # Extract numeric metrics for plotting
        numeric_keys = []
        sample = results[0]
        for key, value in sample.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_keys.append(key)

        # Limit to key metrics for readability
        key_metrics = [k for k in numeric_keys if any(x in k for x in
                      ['recall', 'precision', 'mrr', 'rouge', 'exact', 'citation'])][:6]

        if not key_metrics:
            return

        # Create box plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(key_metrics):
            if i >= len(axes):
                break
            values = [r.get(metric, 0.0) for r in results]
            axes[i].boxplot(values)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylabel('Score')
            axes[i].grid(True, alpha=0.3)

        # Remove empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "figures", f"metrics_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved metrics plot to {plot_path}")

    def run_evaluation_pipeline(self, questions: List[str],
                                ground_truth_answers: List[str],
                                ground_truth_evidence_list: List[List[str]],
                                retrieved_chunks_list: List[List[Dict[str, Any]]],
                                generated_answers: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run complete evaluation pipeline.

        Args:
            questions: List of questions
            ground_truth_answers: List of ground truth answers
            ground_truth_evidence_list: List of ground truth evidence lists
            retrieved_chunks_list: List of retrieved chunks lists
            generated_answers: List of generated answers (if None, would need to generate)

        Returns:
            Tuple of (detailed_results, aggregated_results)
        """
        # Validate input lengths
        n = len(questions)
        assert len(ground_truth_answers) == n, "Mismatch in questions and answers"
        assert len(ground_truth_evidence_list) == n, "Mismatch in questions and evidence"
        assert len(retrieved_chunks_list) == n, "Mismatch in questions and retrieved chunks"
        if generated_answers is not None:
            assert len(generated_answers) == n, "Mismatch in questions and generated answers"

        # Prepare examples for evaluation
        examples = []
        for i in range(n):
            example = {
                "question": questions[i],
                "ground_truth_answer": ground_truth_answers[i],
                "ground_truth_evidence": ground_truth_evidence_list[i],
                "retrieved_chunks": retrieved_chunks_list[i]
            }
            if generated_answers is not None:
                example["generated_answer"] = generated_answers[i]

            examples.append(example)

        # Evaluate batch
        results = self.evaluate_batch(examples)

        # Aggregate results
        aggregated = self.aggregate_results(results)

        return results, aggregated


def create_evaluation_pipeline() -> RAGEvaluator:
    """Factory function to create evaluation pipeline."""
    return RAGEvaluator()


