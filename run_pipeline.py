#!/usr/bin/env python3
"""
Main evaluation script for FinanceBench RAG project.
Orchestrates the complete pipeline from data processing to evaluation.
"""

import os
import sys
import json
import argparse
import yaml
from typing import List, Dict, Any
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_processing.download_pdfs import download_all_pdfs
from data_processing.ingest import process_all_documents
from retrieval.retrieve import build_index, load_index, search_index
from generation.generate import FinancialQAModel, create_qa_pipeline
from evaluation.evaluate import RAGEvaluator, create_evaluation_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_data_acquisition(config: Dict[str, Any]):
    """Step 1: Download PDFs."""
    logger.info("=== Step 1: Data Acquisition (PDF Download) ===")

    output_dir = config.get("data_acquisition", {}).get("output_dir", "data/raw_pdfs")
    download_all_pdfs(
        document_info_path="data/financebench_document_information.jsonl",
        output_dir=output_dir
    )


def run_corpus_processing(config: Dict[str, Any]):
    """Step 2: Process documents (extract, clean, chunk)."""
    logger.info("=== Step 2: Corpus Processing ===")

    pdf_dir = config.get("corpus_processing", {}).get("pdf_dir", "data/raw_pdfs")
    output_dir = config.get("corpus_processing", {}).get("output_dir", "data/processed_chunks")
    chunk_strategies = config.get("corpus_processing", {}).get("chunk_strategies",
                                                           ["fixed_256_overlap32", "fixed_512_overlap64", "structure"])
    extract_method = config.get("corpus_processing", {}).get("extract_method", "unstructured")

    process_all_documents(
        document_info_path="data/financebench_document_information.jsonl",
        pdf_dir=pdf_dir,
        output_dir=output_dir,
        chunk_strategies=chunk_strategies,
        extract_method=extract_method
    )


def run_retrieval_setup(config: Dict[str, Any]):
    """Step 3: Set up retrieval system (embeddings, indexing)."""
    logger.info("=== Step 3: Retrieval Setup ===")

    # Load processed chunks
    from retrieval.retrieve import load_chunks_from_directory
    chunks_dir = config.get("retrieval", {}).get("chunks_dir", "data/processed_chunks")
    chunks = load_chunks_from_directory(chunks_dir)

    if not chunks:
        logger.warning("No chunks found. Please run corpus processing first.")
        return None

    # Build index
    embedding_model = config.get("retrieval", {}).get("embedding_model", "BAAI/bge-base-en-v1.5")
    store_type = config.get("retrieval", {}).get("store_type", "faiss")
    index_name = config.get("retrieval", {}).get("index_name", "financebench_index")
    batch_size = config.get("retrieval", {}).get("batch_size", 32)

    vector_store = build_index(
        chunks=chunks,
        embedding_model=embedding_model,
        store_type=store_type,
        index_name=index_name,
        batch_size=batch_size
    )

    logger.info(f"Retrieval index built and saved as {index_name}")
    return vector_store


def run_generation_setup(config: Dict[str, Any]) -> FinancialQAModel:
    """Step 4: Set up generation model."""
    logger.info("=== Step 4: Generation Setup ===")

    model_name = config.get("generation", {}).get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
    device = config.get("generation", {}).get("device", "auto")
    load_in_4bit = config.get("generation", {}).get("load_in_4bit", True)

    qa_model = create_qa_pipeline(
        model_name=model_name,
        device=device,
        load_in_4bit=load_in_4bit
    )

    return qa_model


def run_evaluation(config: Dict[str, Any],
                   vector_store=None,
                   qa_model: FinancialQAModel = None):
    """Step 5: Run evaluation pipeline."""
    logger.info("=== Step 5: Evaluation Pipeline ===")

    # Load QA dataset
    qa_data_path = config.get("evaluation", {}).get("qa_data_path", "data/financebench_open_source.jsonl")

    # Load QA data
    def load_jsonl(filepath):
        with open(filepath, 'r') as f:
            return [json.loads(line) for line in f]

    qa_data = load_jsonl(qa_data_path)

    # Limit number of examples for testing if specified
    max_examples = config.get("evaluation", {}).get("max_examples", None)
    if max_examples:
        qa_data = qa_data[:max_examples]
        logger.info(f"Limiting evaluation to {max_examples} examples")

    # Extract components
    questions = [item["question"] for item in qa_data]
    ground_truth_answers = [item["answer"] for item in qa_data]

    # Extract evidence (handle different possible formats)
    ground_truth_evidence_list = []
    for item in qa_data:
        evidence = []
        if "evidence" in item and isinstance(item["evidence"], list):
            for ev in item["evidence"]:
                if isinstance(ev, dict) and "evidence_text" in ev:
                    evidence.append(ev["evidence_text"])
                elif isinstance(ev, str):
                    evidence.append(ev)
        elif "evidence_text" in item:
            evidence.append(item["evidence_text"])
        ground_truth_evidence_list.append(evidence)

    # Retrieve chunks for each question
    logger.info("Retrieving chunks for questions...")
    retrieved_chunks_list = []

    retrieval_k = config.get("evaluation", {}).get("retrieval_k", 5)
    embedding_model = config.get("retrieval", {}).get("embedding_model", "BAAI/bge-base-en-v1.5")

    for i, question in enumerate(questions):
        if i % 10 == 0:
            logger.info(f"Processing question {i+1}/{len(questions)}")

        # Use the vector store if provided, otherwise load default index
        if vector_store is not None:
            vs = vector_store
        else:
            index_name = config.get("retrieval", {}).get("index_name", "financebench_index")
            store_type = config.get("retrieval", {}).get("store_type", "faiss")
            vs = load_index(index_name=index_name, store_type=store_type)

        chunks = search_index(
            vector_store=vs,
            query=question,
            embedding_model=embedding_model,
            k=retrieval_k
        )
        retrieved_chunks_list.append(chunks)

    # Generate answers if model is provided
    generated_answers = None
    if qa_model is not None:
        logger.info("Generating answers...")
        generated_answers = []
        generation_kwargs = config.get("generation", {}).get("generation_kwargs", {
            "max_new_tokens": 256,
            "temperature": 0.1,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        })

        for i, (question, contexts) in enumerate(zip(questions, retrieved_chunks_list)):
            if i % 10 == 0:
                logger.info(f"Generating answer {i+1}/{len(questions)}")

            result = qa_model.answer_question(question, contexts, **generation_kwargs)
            generated_answers.append(result["answer"])
    else:
        logger.warning("No generation model provided. Skipping answer generation.")

    # Run evaluation
    evaluator = create_evaluation_pipeline()
    results, aggregated = evaluator.run_evaluation_pipeline(
        questions=questions,
        ground_truth_answers=ground_truth_answers,
        ground_truth_evidence_list=ground_truth_evidence_list,
        retrieved_chunks_list=retrieved_chunks_list,
        generated_answers=generated_answers
    )

    # Save results
    output_dir = config.get("evaluation", {}).get("output_dir", "results")
    evaluator.save_results(results, aggregated, output_dir)

    logger.info("Evaluation complete!")
    return results, aggregated


def main():
    """Main function to run the RAG pipeline."""
    parser = argparse.ArgumentParser(description="FinanceBench RAG Project")
    parser.add_argument("--step", choices=["all", "download", "process", "retrieve", "generate", "evaluate"],
                       default="all", help="Which step to run")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip PDF download step")
    parser.add_argument("--skip-process", action="store_true",
                       help="Skip corpus processing step")

    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found. Using default configuration.")
        # Create a default config
        config = {
            "data_acquisition": {"output_dir": "data/raw_pdfs"},
            "corpus_processing": {
                "pdf_dir": "data/raw_pdfs",
                "output_dir": "data/processed_chunks",
                "chunk_strategies": ["fixed_256_overlap32", "fixed_512_overlap64", "structure"],
                "extract_method": "unstructured"
            },
            "retrieval": {
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "store_type": "faiss",
                "index_name": "financebench_index",
                "batch_size": 32,
                "chunks_dir": "data/processed_chunks"
            },
            "generation": {
                "model_name": "meta-llama/Llama-3.1-8B-Instruct",
                "device": "auto",
                "load_in_4bit": True,
                "generation_kwargs": {
                    "max_new_tokens": 256,
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "repetition_penalty": 1.1
                }
            },
            "evaluation": {
                "qa_data_path": "data/financebench_open_source.jsonl",
                "max_examples": 50,  # Limit for testing
                "retrieval_k": 5,
                "output_dir": "results"
            }
        }
    else:
        config = load_config(config_path)

    # Run selected steps
    if args.step == "all" or args.step == "download":
        if not args.skip_download:
            run_data_acquisition(config)
        else:
            logger.info("Skipping data acquisition step")

    if args.step == "all" or args.step == "process":
        if not args.skip_process:
            run_corpus_processing(config)
        else:
            logger.info("Skipping corpus processing step")

    vector_store = None
    qa_model = None

    if args.step == "all" or args.step == "retrieve" or args.step == "evaluate":
        vector_store = run_retrieval_setup(config)

    if args.step == "all" or args.step == "generate" or args.step == "evaluate":
        qa_model = run_generation_setup(config)

    if args.step == "all" or args.step == "evaluate":
        run_evaluation(config, vector_store, qa_model)

    elif args.step == "download":
        run_data_acquisition(config)
    elif args.step == "process":
        run_corpus_processing(config)
    elif args.step == "retrieve":
        run_retrieval_setup(config)
    elif args.step == "generate":
        run_generation_setup(config)
    elif args.step == "evaluate":
        run_evaluation(config, vector_store, qa_model)


if __name__ == "__main__":
    main()