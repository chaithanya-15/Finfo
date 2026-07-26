#!/usr/bin/env python3
"""
M2: Chunk Size & Overlap Grid Sweep
M3: Model Family Comparison Sweep
Chunk text is indexed in full: the stored text is what evidence_overlap scores against.
"""

import json
import glob
from pathlib import Path
import numpy as np
import yaml
from src.retrieval.retrieve import build_index, load_index, search_index
from src.evaluation.evaluate import RetrievalEvaluator, evidence_overlap
from run_experiments import load_qa_data, ensure_chunks

config = yaml.safe_load(open("configs/base_config.yaml"))
data = load_qa_data("data/financebench_open_source.jsonl", "data/raw_pdfs")
indices = [i for i, ans in enumerate(data["answerable"]) if ans]

Path("results/experiments").mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# M2: Chunk Size & Overlap Grid (256/32, 512/64, 1024/128)
# -------------------------------------------------------------
print("=== Running M2: Chunk Size Grid ===")
grid_strategies = ["fixed_256_overlap32", "fixed_512_overlap64", "fixed_1024_overlap128"]
grid_models = ["BAAI/bge-base-en-v1.5"]

m2_results = []

for strat in grid_strategies:
    ensure_chunks(strat, config)
    chunk_files = glob.glob(f"data/processed_chunks/*_{strat}.jsonl")
    chunks = []
    for cf in chunk_files:
        for line in open(cf):
            c = json.loads(line)
            # Chunk text is stored in full. The embedder truncates at its own token limit,
            # but evidence_overlap scores against this stored text, so cutting it here
            # silently lowers recall for any chunk whose evidence sits past the cut.
            chunks.append(c)

    for model_name in grid_models:
        safe_name = model_name.replace("/", "_")
        idx_path = Path(f"indexes/{safe_name}__{strat}")
        
        if idx_path.with_suffix(".index").exists():
            print(f"Loading index for {model_name} over {strat} ({len(chunks)} chunks)...")
            vector_store = load_index(str(idx_path))
        else:
            print(f"Building index for {model_name} over {strat} ({len(chunks)} chunks with batch_size=128)...")
            vector_store = build_index(chunks, model_name, index_name=str(idx_path), batch_size=128)
            
        r5s, r1s, mrrs, ovs, hits = [], [], [], [], []
        for idx in indices:
            q = data["questions"][idx]
            gold_doc = data["doc_names"][idx]
            ev_items = data["evidence"][idx]
            
            res_chunks = search_index(vector_store, q, model_name, k=5)
            
            r5s.append(RetrievalEvaluator.calculate_recall_at_k(res_chunks, ev_items, k=5, threshold=0.5))
            r1s.append(RetrievalEvaluator.calculate_recall_at_k(res_chunks, ev_items, k=1, threshold=0.5))
            mrrs.append(RetrievalEvaluator.calculate_mrr(res_chunks, ev_items, threshold=0.5))
            
            o = [evidence_overlap(ev, c.get("text", "")) for ev in ev_items for c in res_chunks[:5]]
            ovs.append(max(o) if o else 0.0)
            hits.append(1.0 if any(c.get("doc_name") == gold_doc for c in res_chunks[:5]) else 0.0)
            
        res = {
            "strategy": strat,
            "model_name": model_name,
            "n_chunks": len(chunks),
            "recall_at_5": float(np.mean(r5s)),
            "recall_at_1": float(np.mean(r1s)),
            "mrr": float(np.mean(mrrs)),
            "mean_max_overlap": float(np.mean(ovs)),
            "doc_hit_at_5": float(np.mean(hits))
        }
        m2_results.append(res)
        print(f"[{strat} | {model_name}] R@5: {res['recall_at_5']:.4f} | MRR: {res['mrr']:.4f} | DocHit@5: {res['doc_hit_at_5']:.4f}")

with open("results/experiments/chunk_grid.json", "w") as f:
    json.dump(m2_results, f, indent=2)
print("Saved results/experiments/chunk_grid.json!")


# -------------------------------------------------------------
# M3: Model Family Comparison (bge-base vs all-MiniLM-L6-v2 vs e5-large-v2)
# -------------------------------------------------------------
print("\n=== Running M3: Model Family Comparison ===")
m3_models = ["BAAI/bge-base-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2", "intfloat/e5-large-v2"]
strat = "fixed_512_overlap64"

chunk_files = glob.glob(f"data/processed_chunks/*_{strat}.jsonl")
chunks = []
for cf in chunk_files:
    for line in open(cf):
        c = json.loads(line)
        chunks.append(c)

m3_results = []

for model_name in m3_models:
    safe_name = model_name.replace("/", "_")
    idx_path = Path(f"indexes/{safe_name}__{strat}")
    
    if idx_path.with_suffix(".index").exists():
        print(f"Loading index for {model_name}...")
        vector_store = load_index(str(idx_path))
    else:
        print(f"Building index for {model_name} with batch_size=128...")
        vector_store = build_index(chunks, model_name, index_name=str(idx_path), batch_size=128)
        
    r5s, r1s, mrrs, ovs, hits = [], [], [], [], []
    for idx in indices:
        q = data["questions"][idx]
        gold_doc = data["doc_names"][idx]
        ev_items = data["evidence"][idx]
        
        res_chunks = search_index(vector_store, q, model_name, k=5)
        
        r5s.append(RetrievalEvaluator.calculate_recall_at_k(res_chunks, ev_items, k=5, threshold=0.5))
        r1s.append(RetrievalEvaluator.calculate_recall_at_k(res_chunks, ev_items, k=1, threshold=0.5))
        mrrs.append(RetrievalEvaluator.calculate_mrr(res_chunks, ev_items, threshold=0.5))
        
        o = [evidence_overlap(ev, c.get("text", "")) for ev in ev_items for c in res_chunks[:5]]
        ovs.append(max(o) if o else 0.0)
        hits.append(1.0 if any(c.get("doc_name") == gold_doc for c in res_chunks[:5]) else 0.0)
        
    res = {
        "model_name": model_name,
        "strategy": strat,
        "n_chunks": len(chunks),
        "recall_at_5": float(np.mean(r5s)),
        "recall_at_1": float(np.mean(r1s)),
        "mrr": float(np.mean(mrrs)),
        "mean_max_overlap": float(np.mean(ovs)),
        "doc_hit_at_5": float(np.mean(hits))
    }
    m3_results.append(res)
    print(f"[{model_name}] R@5: {res['recall_at_5']:.4f} | MRR: {res['mrr']:.4f} | DocHit@5: {res['doc_hit_at_5']:.4f}")

with open("results/experiments/model_family.json", "w") as f:
    json.dump(m3_results, f, indent=2)
print("Saved results/experiments/model_family.json!")
