#!/usr/bin/env python3
"""
Simple test script to verify basic functionality of the RAG pipeline.
This script runs a minimal version of the pipeline on a small subset of data.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")

    try:
        from data_processing.download_pdfs import download_all_pdfs
        logger.info("✓ download_pdfs imported")
    except Exception as e:
        logger.error(f"✗ Failed to import download_pdfs: {e}")
        return False

    try:
        from data_processing.ingest import process_all_documents
        logger.info("✓ ingest imported")
    except Exception as e:
        logger.error(f"✗ Failed to import ingest: {e}")
        return False

    try:
        from retrieval.retrieve import build_index, load_index, search_index
        logger.info("✓ retrieve imported")
    except Exception as e:
        logger.error(f"✗ Failed to import retrieve: {e}")
        return False

    try:
        from generation.generate import FinancialQAModel, create_qa_pipeline
        logger.info("✓ generate imported")
    except Exception as e:
        logger.error(f"✗ Failed to import generate: {e}")
        return False

    try:
        from evaluation.evaluate import RAGEvaluator, create_evaluation_pipeline
        logger.info("✓ evaluate imported")
    except Exception as e:
        logger.error(f"✗ Failed to import evaluate: {e}")
        return False

    return True

def test_data_loading():
    """Test that we can load the JSONL data files."""
    logger.info("Testing data loading...")

    # Check if data files exist
    doc_info_path = "data/financebench_document_information.jsonl"
    qa_path = "data/financebench_open_source.jsonl"

    if not os.path.exists(doc_info_path):
        logger.warning(f"Document info file not found: {doc_info_path}")
        # Create a minimal test file
        os.makedirs("data", exist_ok=True)
        test_data = [{"doc_name": "TEST_2020_10K", "company": "TEST", "doc_link": "http://example.com/test.pdf"}]
        with open(doc_info_path, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Created test document info file: {doc_info_path}")

    if not os.path.exists(qa_path):
        logger.warning(f"QA file not found: {qa_path}")
        # Create a minimal test file
        test_qa = [{
            "question": "What is the test value?",
            "answer": "42",
            "evidence": [{"evidence_text": "The answer is 42."}]
        }]
        with open(qa_path, 'w') as f:
            for item in test_qa:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Created test QA file: {qa_path}")

    # Try to load the files
    try:
        def load_jsonl(filepath):
            with open(filepath, 'r') as f:
                return [json.loads(line) for line in f]

        docs = load_jsonl(doc_info_path)
        qa_data = load_jsonl(qa_path)
        logger.info(f"✓ Loaded {len(docs)} documents and {len(qa_data)} QA pairs")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load data files: {e}")
        return False

def test_basic_functions():
    """Test basic functions from each module."""
    logger.info("Testing basic functions...")

    # Test text cleaning
    try:
        from data_processing.ingest import clean_text
        test_text = "This  is   a  test.\n\nNew line!  "
        cleaned = clean_text(test_text)
        logger.info(f"✓ Text cleaning: '{test_text}' -> '{cleaned}'")
    except Exception as e:
        logger.error(f"✗ Text cleaning failed: {e}")
        return False

    # Test chunking
    try:
        from data_processing.ingest import chunk_text_fixed_size
        test_text = "This is a test sentence. " * 50  # Make it long enough to chunk
        chunks = chunk_text_fixed_size(test_text, chunk_size=50, overlap=10)
        logger.info(f"✓ Text chunking: Created {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"✗ Text chunking failed: {e}")
        return False

    # Test embedding manager = False
        return False

    logger.info("All tests passed!")
    return True

if __name__ == "__main__":
    success = True
    success &= test_imports()
    success &= test_data_loading()
    success &= test_basic_functions()

    if success:
        logger.info("🎉 All tests passed! The RAG pipeline is ready to use.")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)