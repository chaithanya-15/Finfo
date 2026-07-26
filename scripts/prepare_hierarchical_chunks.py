#!/usr/bin/env python3
"""
M1 Hierarchical Chunking Data Prep Script:
Generates 128-token chunks with 16-token overlap across all extracted document JSON files
in data/extracted_text/pdfplumber, saving them to data/processed_chunks/<doc>_fixed_128_overlap16.jsonl
and creating a parent-child mapping dictionary (128-token child chunk -> 1024-token parent context window).
Logs total chunk count.
"""

import os
import sys
import glob
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data_processing.ingest import clean_text, chunk_text_hierarchical

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("hierarchical_chunking_prep")


def process_single_doc(file_and_meta: Tuple[str, Dict[str, Any], str, str]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    json_path_str, doc_meta, extracted_dir_str, output_dir_str = file_and_meta
    json_path = Path(json_path_str)
    doc_name = json_path.stem

    with open(json_path, "r", encoding="utf-8") as f:
        extracted_data = json.load(f)

    raw_text = extracted_data.get("text", "")
    if not raw_text.strip():
        return doc_name, [], {}

    cleaned_text = clean_text(raw_text)
    doc_period = doc_meta.get("doc_period", "unknown")
    company = doc_meta.get("company", "unknown")
    gics_sector = doc_meta.get("gics_sector", "unknown")
    doc_type = doc_meta.get("doc_type", "unknown")
    page_number = doc_meta.get("evidence_page_num", 0)

    # 128-token child chunk, 16 overlap, 1024 parent window
    chunk_dicts = chunk_text_hierarchical(
        cleaned_text,
        child_size=128,
        child_overlap=16,
        parent_size=1024,
        encoding_name="cl100k_base"
    )

    doc_processed_chunks = []
    doc_mappings = {}

    for chunk in chunk_dicts:
        idx = chunk["chunk_index"]
        chunk_id = f"{doc_name}_p{doc_period}_c{idx:04d}"

        chunk_record = {
            "chunk_id": chunk_id,
            "doc_name": doc_name,
            "company": company,
            "gics_sector": gics_sector,
            "doc_type": doc_type,
            "doc_period": doc_period,
            "page_number": page_number,
            "section": "unknown",
            "text": chunk["text"],
            "chunk_strategy": "fixed_128_overlap16",
            "chunk_index": idx,
            "token_count": chunk["token_count"]
        }
        doc_processed_chunks.append(chunk_record)

        doc_mappings[chunk_id] = {
            "child_chunk_id": chunk_id,
            "doc_name": doc_name,
            "company": company,
            "doc_period": doc_period,
            "child_start_token": chunk["start_token"],
            "child_end_token": chunk["end_token"],
            "child_token_count": chunk["token_count"],
            "parent_start_token": chunk["parent_start_token"],
            "parent_end_token": chunk["parent_end_token"],
            "parent_token_count": chunk["parent_token_count"],
            "parent_text": chunk["parent_text"]
        }

    # Write per-document JSONL
    out_jsonl_path = Path(output_dir_str) / f"{doc_name}_fixed_128_overlap16.jsonl"
    with open(out_jsonl_path, "w", encoding="utf-8") as out_f:
        for item in doc_processed_chunks:
            out_f.write(json.dumps(item) + "\n")

    return doc_name, doc_processed_chunks, doc_mappings


def main():
    extracted_dir = Path("data/extracted_text/pdfplumber")
    output_dir = Path("data/processed_chunks")
    doc_info_path = Path("data/financebench_document_information.jsonl")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata
    doc_meta_map: Dict[str, Dict[str, Any]] = {}
    if doc_info_path.exists():
        with open(doc_info_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    doc_meta_map[item["doc_name"]] = item

    json_files = sorted(extracted_dir.glob("*.json"))
    logger.info(f"Processing {len(json_files)} extracted document JSON files with ProcessPoolExecutor...")

    tasks = [(str(p), doc_meta_map.get(p.stem, {}), str(extracted_dir), str(output_dir)) for p in json_files]

    total_chunk_count = 0
    parent_child_mapping: Dict[str, Dict[str, Any]] = {}
    docs_processed = 0

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_single_doc, tasks)
        for doc_name, chunks, mappings in results:
            if chunks:
                total_chunk_count += len(chunks)
                parent_child_mapping.update(mappings)
                docs_processed += 1

    logger.info(f"Successfully processed {docs_processed} / {len(json_files)} documents.")
    logger.info(f"TOTAL CHUNK COUNT (128-token child chunks): {total_chunk_count}")

    # Save parent-child mapping dictionary
    mapping_path = output_dir / "parent_child_mapping_128_1024.json"
    with open(mapping_path, "w", encoding="utf-8") as map_f:
        json.dump(parent_child_mapping, map_f)
    logger.info(f"Saved parent-child mapping dictionary to {mapping_path} ({len(parent_child_mapping)} entries).")

    alias_path = output_dir / "parent_child_mapping.json"
    with open(alias_path, "w", encoding="utf-8") as map_f:
        json.dump(parent_child_mapping, map_f)
    logger.info(f"Saved alias parent-child mapping dictionary to {alias_path}.")



if __name__ == "__main__":
    main()
