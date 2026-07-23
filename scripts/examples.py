#!/usr/bin/env python3
"""
Example usage of the FinanceBench RAG modules.
This demonstrates how to use each component individually.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def example_data_processing():
    """Example of using the data processing module."""
    print("=== Data Processing Example ===")

    # This would normally download PDFs
    # from data_processing.download_pdfs import download_all_pdfs
    # download_all_pdfs()

    # Example of text cleaning function
    from data_processing.ingest import clean_text

    dirty_text = "  This  is   a   test  \n\n  with  extra   spaces  and\nnewlines  "
    clean = clean_text(dirty_text)

    print(f"Original: '{dirty_text}'")
    print(f"Cleaned:  '{clean}'")
    print()

def example_retrieval_setup():
    """Example of setting up the retrieval system."""
    print("=== Retrieval Setup Example ===")

    # Check if we have any processed chunks
    chunks_dir = "data/processed_chunks"
    if os.path.exists(chunks_dir):
        from retrieve import load_chunks_from_directory, build_index

        print(f"Loading chunks from {chunks_dir}...")
        chunks = load_chunks_from_directory(chunks_dir)

        if chunks:
            print(f"Loaded {len(chunks)} chunks")

            # Build a small index for demonstration
            print("Building FAISS index...")
            vector_store = build_index(
                chunks=chunks[:10],  # Just use first 10 for demo
                embedding_model="BAAI/bge-base-en-v1.5",
                store_type="faiss",
                index_name="demo_index",
                batch_size=2
            )

            print("Index built successfully!")

            # Example search
            results = search_index(
                vector_store=vector_store,
                query="What is the revenue?",
                embedding_model="BAAI/bge-base-en-v1.5",
                k=3
            )

            print(f"Found {len(results)} results for query:")
            for i, result in enumerate(results):
                print(f"  {i+1}. Score: {result['score']:.3f} - {result['text'][:100]}...")
        else:
            print("No chunks found. Run data processing first.")
    else:
        print(f"Directory {chunks_dir} not found. Run data processing first.")

    print()

def example_generation_setup():
    """Example of setting up the generation model."""
    print("=== Generation Setup Example ===")

    try:
        from generate import create_qa_pipeline, FinancialQAModel

        print("Creating QA pipeline (this would download the model)...")
        # Note: This would actually download and load a large model
        # For demonstration, we'll just show how it would be used

        # qa_model = create_qa_pipeline(
        #     model_name="Qwen3.5-4B",
        #     device="auto",
        #     load_in_4bit=True
        # )

        print("QA pipeline would be created here.")
        print("To use it:")
        print("  result = qa_model.answer_question(question, contexts)")
        print("  print(result['answer'])")

    except Exception as e:
        print(f"Note: Full model loading skipped due to: {e}")
        print("This is expected in environments without GPU/model access.")

    print()

def example_evaluation():
    """Example of setting up evaluation."""
    print("=== Evaluation Setup Example ===")

    from evaluate import RAGEvaluator

    evaluator = RAGEvaluator()
    print("Evaluator created successfully!")
    print("To use it:")
    print("  results, aggregated = evaluator.run_evaluation_pipeline(")
    print("      questions, answers, evidence, retrieved_chunks)")
    print("  evaluator.save_results(results, aggregated)")
    print()

def main():
    """Run all examples."""
    print("FinanceBench RAG Project - Component Examples")
    print("=" * 50)

    example_data_processing()
    example_retrieval_setup()
    example_generation_setup()
    example_evaluation()

    print("=" * 50)
    print("Examples completed!")
    print("See the individual module files for detailed usage.")

if __name__ == "__main__":
    main()