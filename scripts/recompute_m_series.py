#!/usr/bin/env python3
"""
Recompute M-series metrics (M2 chunk grid and M3 model family) dynamically over 114 answerable questions.
Every row is computed through the exact same retrieval and evaluation pipeline.
"""

import json
import glob
from pathlib import Path
import numpy as np
import yaml
from src.retrieval.retrieve import build_index, load_index, search_index
from src.evaluation.evaluate import RAGEvaluator
from run_experiments import load_qa_data, ensure_chunks

config = yaml.safe_load(open("configs/base_config.yaml"))
qa_data = load_qa_data("data/financebench_open_source.jsonl", "data/raw_pdfs")
ans_indices = [i for i, ok in enumerate(qa_data["answerable"]) if ok]
evaluator = RAGEvaluator()

print(f"Loaded {len(qa_data['questions'])} total questions, {len(ans_indices)} answerable questions.")

# -------------------------------------------------------------
# M2: Chunk Size Grid Sweep (Macro-averaged over 114 answerable)
# -------------------------------------------------------------
print("\n=== Recomputing M2: Chunk Size Grid (Macro-averaged over 114 answerable) ===")
grid_strategies = ["fixed_256_overlap32", "fixed_512_overlap64", "fixed_1024_overlap128"]
grid_models = ["BAAI/bge-base-en-v1.5"]

m2_results = []

for strat in grid_strategies:
    ensure_chunks(strat, config)
    chunk_files = glob.glob(f"data/processed_chunks/*_{strat}.jsonl")
    chunks = []
    for cf in chunk_files:
        for line in open(cf):
            chunks.append(json.loads(line))
            
    for model_name in grid_models:
        safe_name = model_name.replace("/", "_")
        idx_path = Path(f"indexes/{safe_name}__{strat}")
        
        if Path(f"{idx_path}.index").exists():
            print(f"Loading index for {model_name} over {strat} ({len(chunks)} chunks)...")
            vector_store = load_index(str(idx_path))
        else:
            print(f"Building index for {model_name} over {strat} ({len(chunks)} chunks with batch_size=64)...")
            vector_store = build_index(chunks, model_name, index_name=str(idx_path), batch_size=64)
            
        eval_metrics = []
        for idx in ans_indices:
            q = qa_data["questions"][idx]
            gold_doc = qa_data["doc_names"][idx]
            ev_items = qa_data["evidence"][idx]
            
            res_chunks = search_index(vector_store, q, model_name, k=5)
            
            m = evaluator.evaluate_single_example(
                q, "", res_chunks, qa_data["answers"][idx], ev_items, k_values=[1, 3, 5, 10]
            )
            doc_hit = 1.0 if any(c.get("doc_name") == gold_doc for c in res_chunks[:5]) else 0.0
            m["doc_hit_at_5"] = doc_hit
            eval_metrics.append(m)
            
        r5_mean = float(np.mean([m["recall@5"] for m in eval_metrics]))
        r1_mean = float(np.mean([m["recall@1"] for m in eval_metrics]))
        mrr_mean = float(np.mean([m["mrr"] for m in eval_metrics]))
        hit_mean = float(np.mean([m["doc_hit_at_5"] for m in eval_metrics]))
        
        res = {
            "strategy": strat,
            "model_name": model_name,
            "n_chunks": len(chunks),
            "recall_at_5": r5_mean,
            "recall_at_1": r1_mean,
            "mrr": mrr_mean,
            "doc_hit_at_5": hit_mean
        }
        m2_results.append(res)
        print(f"[{strat} | {model_name}] R@5: {res['recall_at_5']:.16f} ({r5_mean*114:.1f}/114) | MRR: {res['mrr']:.4f} | DocHit@5: {res['doc_hit_at_5']:.4f}")

with open("results/experiments/chunk_grid.json", "w") as f:
    json.dump(m2_results, f, indent=2)
print("Saved results/experiments/chunk_grid.json!\n")


# -------------------------------------------------------------
# M3: Model Family Comparison Sweep (Macro-averaged over 114 answerable)
# -------------------------------------------------------------
print("=== Recomputing M3: Model Family Comparison (Macro-averaged over 114 answerable) ===")
m3_models = ["BAAI/bge-base-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]
strat = "fixed_512_overlap64"

chunk_files = glob.glob(f"data/processed_chunks/*_{strat}.jsonl")
chunks = []
for cf in chunk_files:
    for line in open(cf):
        chunks.append(json.loads(line))

m3_results = []

for model_name in m3_models:
    safe_name = model_name.replace("/", "_")
    idx_path = Path(f"indexes/{safe_name}__{strat}")
    
    if Path(f"{idx_path}.index").exists():
        print(f"Loading index for {model_name}...")
        vector_store = load_index(str(idx_path))
    else:
        print(f"Building index for {model_name} with batch_size=64...")
        vector_store = build_index(chunks, model_name, index_name=str(idx_path), batch_size=64)
        
    eval_metrics = []
    for idx in ans_indices:
        q = qa_data["questions"][idx]
        gold_doc = qa_data["doc_names"][idx]
        ev_items = qa_data["evidence"][idx]
        
        res_chunks = search_index(vector_store, q, model_name, k=5)
        
        m = evaluator.evaluate_single_example(
            q, "", res_chunks, qa_data["answers"][idx], ev_items, k_values=[1, 3, 5, 10]
        )
        doc_hit = 1.0 if any(c.get("doc_name") == gold_doc for c in res_chunks[:5]) else 0.0
        m["doc_hit_at_5"] = doc_hit
        eval_metrics.append(m)
        
    r5_mean = float(np.mean([m["recall@5"] for m in eval_metrics]))
    r1_mean = float(np.mean([m["recall@1"] for m in eval_metrics]))
    mrr_mean = float(np.mean([m["mrr"] for m in eval_metrics]))
    hit_mean = float(np.mean([m["doc_hit_at_5"] for m in eval_metrics]))
    
    res = {
        "model_name": model_name,
        "strategy": strat,
        "n_chunks": len(chunks),
        "recall_at_5": r5_mean,
        "recall_at_1": r1_mean,
        "mrr": mrr_mean,
        "doc_hit_at_5": hit_mean
    }
    m3_results.append(res)
    print(f"[{model_name}] R@5: {res['recall_at_5']:.16f} ({r5_mean*114:.1f}/114) | MRR: {res['mrr']:.4f} | DocHit@5: {res['doc_hit_at_5']:.4f}")

with open("results/experiments/model_family.json", "w") as f:
    json.dump(m3_results, f, indent=2)
print("Saved results/experiments/model_family.json!")
