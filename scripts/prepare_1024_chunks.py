#!/usr/bin/env python3
"""
Generate 1024-token chunks with 128-token overlap for all 263 extracted document JSONs.
"""

import json
from pathlib import Path
from src.data_processing.ingest import chunk_text

pdf_dir = "data/raw_pdfs"
extract_dir = "data/extracted_text/pdfplumber"
out_dir = "data/processed_chunks"
strategy = "fixed_1024_overlap128"

Path(out_dir).mkdir(parents=True, exist_ok=True)

docs = {}
for line in open("data/financebench_document_information.jsonl"):
    row = json.loads(line)
    docs.setdefault(row["doc_name"], row)

total_chunks = 0
processed_docs = 0

for name, info in sorted(docs.items()):
    json_path = Path(extract_dir, f"{name}.json")
    if not json_path.exists():
        continue
    
    with open(json_path) as f:
        doc_data = json.load(f)
    
    text = doc_data.get("text", "")
    if not text.strip():
        continue
        
    pages = doc_data.get("pages")
    doc_chunks = chunk_text(
        text=text,
        strategy=strategy,
        doc_name=name,
        company=info.get("company", ""),
        period=info.get("period", ""),
        pages=pages
    )
    
    out_file = Path(out_dir, f"{name}_{strategy}.jsonl")
    with open(out_file, "w") as f:
        for c in doc_chunks:
            f.write(json.dumps(c) + "\n")
            
    total_chunks += len(doc_chunks)
    processed_docs += 1

print(f"1024/128 Chunking complete across {processed_docs} documents: {total_chunks} total chunks generated.")
