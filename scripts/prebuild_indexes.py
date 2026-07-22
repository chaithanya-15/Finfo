#!/usr/bin/env python3
"""
Build the vector indexes the sweep needs, ahead of time, in one clean process.

Index building embeds the whole corpus on the GPU. When it runs inside run_experiments.py
right after a generation config, llama.cpp still holds Metal allocations and the embedding
throughput collapses (roughly 9 texts/s instead of ~60). Building every index here first,
before any generator loads, keeps embedding on a clean GPU, and the sweep then loads each
index from disk.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.ingest import process_document
from src.retrieval.retrieve import build_index, load_chunks_from_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# (embedding_model, chunk_strategy, index_name). One entry per index the sweep references.
TARGETS = [
    ("BAAI/bge-base-en-v1.5", "fixed_512_overlap64", "indexes/BAAI_bge-base-en-v1.5__fixed_512_overlap64"),
    ("BAAI/bge-base-en-v1.5", "fixed_256_overlap32", "indexes/BAAI_bge-base-en-v1.5__fixed_256_overlap32"),
    ("BAAI/bge-base-en-v1.5", "structure", "indexes/BAAI_bge-base-en-v1.5__structure"),
    ("all-MiniLM-L6-v2", "fixed_512_overlap64", "indexes/all-MiniLM-L6-v2__fixed_512_overlap64"),
]

PDF_DIR = "data/raw_pdfs"
CHUNK_DIR = "data/processed_chunks"
BATCH = 128


def ensure_chunks(strategy: str) -> None:
    """Chunk every extracted document under one strategy, skipping files already present."""
    docs = {}
    for line in open("data/financebench_document_information.jsonl"):
        row = json.loads(line)
        docs.setdefault(row["doc_name"], row)
    made = 0
    for name, info in sorted(docs.items()):
        if not Path(PDF_DIR, f"{name}.pdf").exists():
            continue
        if Path(CHUNK_DIR, f"{name}_{strategy}.jsonl").exists():
            continue
        if process_document(info, pdf_dir=PDF_DIR, output_dir=CHUNK_DIR,
                            chunk_strategy=strategy, extract_method="pdfplumber"):
            made += 1
    if made:
        logger.info(f"chunked {made} documents under {strategy}")


def main():
    for model, strategy, index_name in TARGETS:
        if Path(f"{index_name}.index").exists():
            logger.info(f"skip, already built: {index_name}")
            continue
        logger.info(f"=== building {index_name} ===")
        ensure_chunks(strategy)
        chunks = [c for c in load_chunks_from_directory(CHUNK_DIR)
                  if c.get("chunk_strategy") == strategy]
        if not chunks:
            logger.warning(f"no chunks for {strategy}, skipping")
            continue
        t = time.time()
        build_index(chunks=chunks, embedding_model=model, store_type="faiss",
                    index_name=index_name, batch_size=BATCH)
        logger.info(f"=== built {index_name}: {len(chunks)} chunks in {(time.time()-t)/60:.1f} min ===")


if __name__ == "__main__":
    main()
