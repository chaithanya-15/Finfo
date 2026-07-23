#!/usr/bin/env python3
"""
Smoke checks for the RAG pipeline.
Verifies that every module imports, the dataset files parse, and the text helpers behave,
without loading a model or building an index.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Every pipeline module imports cleanly."""
    from data_processing.download_pdfs import download_all_pdfs
    from data_processing.ingest import process_all_documents
    from retrieval.retrieve import build_index, load_index, search_index
    from generation.generate import FinancialQAModel, create_qa_pipeline
    from evaluation.evaluate import RAGEvaluator, create_evaluation_pipeline


def test_data_loading():
    """Both FinanceBench JSONL files are present and parse."""
    doc_info_path = "data/financebench_document_information.jsonl"
    qa_path = "data/financebench_open_source.jsonl"

    for path in (doc_info_path, qa_path):
        assert os.path.exists(path), f"dataset file not found: {path}"

    def load_jsonl(filepath):
        with open(filepath, 'r') as f:
            return [json.loads(line) for line in f]

    docs = load_jsonl(doc_info_path)
    qa_data = load_jsonl(qa_path)
    assert docs, "document catalogue is empty"
    assert qa_data, "question file is empty"
    logger.info(f"Loaded {len(docs)} documents and {len(qa_data)} QA pairs")


def test_basic_functions():
    """Cleaning and fixed-size chunking behave on sample input."""
    from data_processing.ingest import clean_text, chunk_text_fixed_size

    cleaned = clean_text("This  is   a  test.\n\nNew line!  ")
    assert cleaned == "This is a test. New line!"

    chunks = chunk_text_fixed_size("This is a test sentence. " * 50, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)


if __name__ == "__main__":
    failed = 0
    for check in (test_imports, test_data_loading, test_basic_functions):
        try:
            check()
            logger.info(f"{check.__name__} passed")
        except Exception as e:
            logger.error(f"{check.__name__} failed: {e}")
            failed += 1

    sys.exit(1 if failed else 0)
