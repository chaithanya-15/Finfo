#!/usr/bin/env python3
"""
Ablation sweep for the FinanceBench RAG project.

Reads configs/experiments.yaml, applies each entry's overrides to configs/base_config.yaml,
and runs the pipeline once per configuration. Indexes are cached and shared between runs
that use the same embedding model and chunk strategy, so the sweep pays for each index once.

Every metric is reported twice: over all evaluation questions, and over the subset whose
source document is present in data/raw_pdfs. Around a fifth of FinanceBench's document
links are dead, and questions about a missing document are unanswerable by construction.
Mixing the two would understate the system rather than measure it.

Usage:
    python run_experiments.py                      # whole sweep
    python run_experiments.py --only baseline k_3  # named runs
    python run_experiments.py --limit 20           # short smoke run
"""

import os

# Set before torch, faiss, or sentence-transformers import. faiss and torch both link an
# OpenMP runtime on macOS; without this they abort with a duplicate-libomp segfault the first
# time a FAISS search runs alongside a torch model. Offline mode stops sentence-transformers
# making slow network HEAD requests to check for model updates on every load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import copy
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.ingest import process_document
from src.retrieval.retrieve import build_index, load_index, search_index, load_chunks_from_directory
from src.evaluation.evaluate import create_evaluation_pipeline, json_safe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Quieten the libraries that log per batch or per page.
for noisy in ["sentence_transformers", "pdfminer", "chromadb", "httpx"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge override into a copy of base, recursing into nested dictionaries.

    Args:
        base: Base configuration
        override: Values that take precedence

    Returns:
        A new merged dictionary, inputs untouched
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_qa_data(path: str, pdf_dir: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Load the evaluation questions and mark which ones are answerable.

    Args:
        path: Path to financebench_open_source.jsonl
        pdf_dir: Directory of downloaded PDFs
        limit: Keep only the first N questions

    Returns:
        Dictionary of parallel lists plus an answerable mask
    """
    rows = [json.loads(line) for line in open(path)]
    if limit:
        rows = rows[:limit]

    # A document counts as available only if extraction produced usable text. Some
    # downloaded PDFs are corrupt (no valid root object) or near-empty stubs, so the mere
    # presence of a .pdf file overstates coverage. Fall back to PDF presence only when the
    # extraction cache has not been built yet.
    extract_cache = Path("data/extracted_text/pdfplumber")
    available = set()
    if extract_cache.exists():
        for jf in extract_cache.glob("*.json"):
            try:
                if json.load(open(jf)).get("text", "").strip():
                    available.add(jf.stem)
            except (json.JSONDecodeError, OSError):
                continue
    else:
        available = {p.stem for p in Path(pdf_dir).glob("*.pdf")}

    evidence_list = []
    for row in rows:
        evidence = []
        raw = row.get("evidence", [])
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and "evidence_text" in item:
                    evidence.append(item["evidence_text"])
                elif isinstance(item, str):
                    evidence.append(item)
        elif "evidence_text" in row:
            evidence.append(row["evidence_text"])
        evidence_list.append(evidence)

    answerable = [row.get("doc_name") in available for row in rows]
    logger.info(f"Loaded {len(rows)} questions, {sum(answerable)} answerable, "
                f"{len(rows) - sum(answerable)} reference a missing document")

    return {
        "questions": [r["question"] for r in rows],
        "answers": [r["answer"] for r in rows],
        "evidence": evidence_list,
        "doc_names": [r.get("doc_name") for r in rows],
        "question_types": [r.get("question_type", "unknown") for r in rows],
        "answerable": answerable,
    }


def ensure_chunks(strategy: str, config: Dict[str, Any]) -> int:
    """
    Chunk every extracted document under one strategy, skipping work already on disk.

    Extraction itself is cached by src.data_processing.ingest, so this is cheap on a
    second call.

    Args:
        strategy: Chunk strategy name
        config: Merged configuration

    Returns:
        Number of documents that have chunks for this strategy
    """
    cp = config["corpus_processing"]
    pdf_dir, out_dir = cp["pdf_dir"], cp["output_dir"]
    extract_method = cp.get("extract_method", "pdfplumber")

    docs = {}
    for line in open("data/financebench_document_information.jsonl"):
        row = json.loads(line)
        docs.setdefault(row["doc_name"], row)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    done = 0
    for i, (name, info) in enumerate(sorted(docs.items()), 1):
        if not Path(pdf_dir, f"{name}.pdf").exists():
            continue
        target = Path(out_dir, f"{name}_{strategy}.jsonl")
        if target.exists():
            done += 1
            continue
        chunks = process_document(info, pdf_dir=pdf_dir, output_dir=out_dir,
                                  chunk_strategy=strategy, extract_method=extract_method)
        if chunks:
            done += 1
        if i % 25 == 0:
            logger.info(f"  chunked {i} documents under {strategy}")
    return done


def get_index(strategy: str, config: Dict[str, Any], cache: Dict[str, Any]):
    """
    Build or reuse the vector index for one embedding model and chunk strategy.

    Args:
        strategy: Chunk strategy name
        config: Merged configuration
        cache: In-process cache keyed by model and strategy

    Returns:
        A loaded VectorStore
    """
    retr = config["retrieval"]
    model = retr["embedding_model"]
    key = f"{model}::{strategy}"
    if key in cache:
        return cache[key]

    index_name = f"indexes/{model.replace('/', '_')}__{strategy}"
    Path("indexes").mkdir(exist_ok=True)

    if Path(f"{index_name}.index").exists():
        logger.info(f"Loading cached index {index_name}")
        store = load_index(index_name=index_name, store_type=retr.get("store_type", "faiss"))
    else:
        all_chunks = load_chunks_from_directory(config["corpus_processing"]["output_dir"])
        chunks = [c for c in all_chunks if c.get("chunk_strategy") == strategy]
        if not chunks:
            raise RuntimeError(f"No chunks found for strategy {strategy}")
        logger.info(f"Building index for {model} over {len(chunks)} {strategy} chunks")
        t = time.time()
        store = build_index(chunks=chunks, embedding_model=model,
                            store_type=retr.get("store_type", "faiss"),
                            index_name=index_name, batch_size=retr.get("batch_size", 32))
        logger.info(f"  built in {(time.time() - t) / 60:.1f} min")

    cache[key] = store
    return store


def run_one(exp: Dict[str, Any], base: Dict[str, Any], qa: Dict[str, Any],
            index_cache: Dict[str, Any], out_root: str) -> Dict[str, Any]:
    """
    Run a single configuration end to end and write its results.

    Args:
        exp: One entry from configs/experiments.yaml
        base: Base configuration
        qa: Output of load_qa_data
        index_cache: Shared index cache
        out_root: Directory to write per-run results into

    Returns:
        A summary row for the comparison table
    """
    name = exp["name"]
    config = deep_merge(base, exp.get("overrides", {}))
    strategy = config["corpus_processing"]["chunk_strategies"][0]
    k = config["evaluation"]["retrieval_k"]

    logger.info(f"=== {name} | axis={exp.get('axis')} | strategy={strategy} | k={k} ===")
    started = time.time()

    ensure_chunks(strategy, config)
    store = get_index(strategy, config, index_cache)

    logger.info(f"Retrieving top-{k} for {len(qa['questions'])} questions")
    retrieved = [
        search_index(vector_store=store, query=q,
                     embedding_model=config["retrieval"]["embedding_model"], k=k)
        for q in qa["questions"]
    ]

    generated = None
    gen_stats = {}
    if exp.get("generate"):
        from src.generation.generate import create_qa_pipeline
        gen_cfg = config["generation"]
        model = create_qa_pipeline(model_name=gen_cfg["model_name"],
                                   device=gen_cfg.get("device", "auto"),
                                   load_in_4bit=gen_cfg.get("load_in_4bit", True))
        kwargs = gen_cfg.get("generation_kwargs", {})
        generated, dropped, abstained = [], 0, 0
        t = time.time()
        for i, (q, ctx) in enumerate(zip(qa["questions"], retrieved), 1):
            result = model.answer_question(q, ctx, **kwargs)
            generated.append(result["answer"])
            dropped += result["dropped_contexts"]
            if result["answer"].lower().startswith("not enough information"):
                abstained += 1
            if i % 25 == 0:
                rate = i / (time.time() - t)
                logger.info(f"  generated {i}/{len(qa['questions'])} ({rate:.2f} q/s)")
        gen_stats = {
            "gen_minutes": round((time.time() - t) / 60, 1),
            "abstained": abstained,
            "dropped_contexts": dropped,
        }
        del model

    run_dir = Path(out_root, name)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {"experiment": name, "axis": exp.get("axis"), "note": exp.get("note"),
               "chunk_strategy": strategy, "k": k,
               "embedding_model": config["retrieval"]["embedding_model"],
               "generation_model": config["generation"]["model_name"] if exp.get("generate") else None,
               **gen_stats}

    evaluator = create_evaluation_pipeline()
    for scope in ("all", "answerable"):
        if scope == "all":
            idx = list(range(len(qa["questions"])))
        else:
            idx = [i for i, ok in enumerate(qa["answerable"]) if ok]

        results, aggregated = evaluator.run_evaluation_pipeline(
            questions=[qa["questions"][i] for i in idx],
            ground_truth_answers=[qa["answers"][i] for i in idx],
            ground_truth_evidence_list=[qa["evidence"][i] for i in idx],
            retrieved_chunks_list=[retrieved[i] for i in idx],
            generated_answers=[generated[i] for i in idx] if generated else None,
        )

        for row, i in zip(results, idx):
            row["question_type"] = qa["question_types"][i]
            row["doc_name"] = qa["doc_names"][i]
            row["answerable"] = qa["answerable"][i]

        pd.DataFrame(results).to_csv(run_dir / f"detailed_{scope}.csv", index=False)
        with open(run_dir / f"aggregated_{scope}.json", "w") as f:
            json.dump(json_safe(aggregated), f, indent=2)

        summary[f"n_{scope}"] = len(idx)
        for metric, value in _flatten(aggregated).items():
            summary[f"{scope}.{metric}"] = value

    summary["minutes"] = round((time.time() - started) / 60, 1)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(json_safe(summary), f, indent=2)
    logger.info(f"=== {name} done in {summary['minutes']} min ===")
    return summary


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """
    Flatten nested aggregate metrics into dotted keys, keeping only numbers.

    Args:
        d: Nested dictionary of aggregated metrics
        prefix: Key prefix used during recursion

    Returns:
        Flat dictionary of numeric metrics
    """
    out = {}
    for key, value in d.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{full}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[full] = value
    return out


def main():
    parser = argparse.ArgumentParser(description="Run the RAG ablation sweep")
    parser.add_argument("--base", default="configs/base_config.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--out", default="results/experiments")
    parser.add_argument("--only", nargs="*", help="Run only the named experiments")
    parser.add_argument("--limit", type=int, help="Use only the first N questions")
    args = parser.parse_args()

    base = yaml.safe_load(open(args.base))
    experiments = yaml.safe_load(open(args.experiments))["experiments"]
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]
        if not experiments:
            raise SystemExit(f"No experiments matched {args.only}")

    qa = load_qa_data(base["evaluation"]["qa_data_path"],
                      base["corpus_processing"]["pdf_dir"], args.limit)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    index_cache: Dict[str, Any] = {}
    summaries: List[Dict[str, Any]] = []

    for exp in experiments:
        try:
            summaries.append(run_one(exp, base, qa, index_cache, args.out))
        except Exception:
            logger.exception(f"Experiment {exp['name']} failed, continuing with the rest")

        if summaries:
            pd.DataFrame(summaries).to_csv(Path(args.out, "comparison.csv"), index=False)

    if summaries:
        print("\n" + pd.DataFrame(summaries)[["experiment", "axis", "n_all", "minutes"]].to_string(index=False))
        print(f"\nWrote {Path(args.out, 'comparison.csv')}")


if __name__ == "__main__":
    main()
    # All results are flushed to disk by now. Exit before the interpreter tears down the
    # llama.cpp Metal context, whose destructor aborts (SIGABRT) on this platform and would
    # otherwise turn a successful run into a non-zero exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
