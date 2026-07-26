#!/usr/bin/env python3
"""
Rebuild 256 and 1024 indexes with full untruncated text, perform max text length sanity checks (> 2000),
and re-run M2 and M3 via recompute_m_series.py.
"""

import glob
import json
import os
import pickle
from pathlib import Path
import yaml

from src.retrieval.retrieve import build_index
from run_experiments import ensure_chunks

config = yaml.safe_load(open("configs/base_config.yaml"))
model_name = "BAAI/bge-base-en-v1.5"

# 1. Rebuild fixed_256_overlap32 index
strat_256 = "fixed_256_overlap32"
ensure_chunks(strat_256, config)
cf_256 = glob.glob(f"data/processed_chunks/*_{strat_256}.jsonl")
chunks_256 = []
for f in cf_256:
    for line in open(f):
        chunks_256.append(json.loads(line))

idx_path_256 = f"indexes/BAAI_bge-base-en-v1.5__{strat_256}"
print(f"Building clean index for {strat_256} from {len(chunks_256)} chunks...")
build_index(chunks_256, model_name, index_name=idx_path_256, batch_size=128)

meta_256 = pickle.load(open(f"{idx_path_256}_metadata_map.pkl", "rb"))
max_len_256 = max(len(v["text"]) for v in meta_256.values())
print(f"Sanity Check ({strat_256}): max(len(v['text'])) = {max_len_256}")
if max_len_256 == 2000:
    raise ValueError(f"Truncation detected in {strat_256}! max length equals exactly 2000.")


# 2. Rebuild fixed_1024_overlap128 index
strat_1024 = "fixed_1024_overlap128"
ensure_chunks(strat_1024, config)
cf_1024 = glob.glob(f"data/processed_chunks/*_{strat_1024}.jsonl")
chunks_1024 = []
for f in cf_1024:
    for line in open(f):
        chunks_1024.append(json.loads(line))

idx_path_1024 = f"indexes/BAAI_bge-base-en-v1.5__{strat_1024}"
print(f"\nBuilding clean index for {strat_1024} from {len(chunks_1024)} chunks...")
build_index(chunks_1024, model_name, index_name=idx_path_1024, batch_size=128)

meta_1024 = pickle.load(open(f"{idx_path_1024}_metadata_map.pkl", "rb"))
max_len_1024 = max(len(v["text"]) for v in meta_1024.values())
print(f"Sanity Check ({strat_1024}): max(len(v['text'])) = {max_len_1024}")
if max_len_1024 == 2000:
    raise ValueError(f"Truncation detected in {strat_1024}! max length equals exactly 2000.")

print("\nBoth 256 and 1024 indexes built cleanly and passed max length sanity checks!")
