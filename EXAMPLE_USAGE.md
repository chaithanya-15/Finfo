# FinanceBench RAG Project - Example Usage

This directory contains examples of how to use the FinanceBench RAG pipeline.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the complete pipeline on a small subset:
   ```bash
   python run_pipeline.py --step all --config configs/base_config.yaml
   ```

3. Or run individual steps:
   ```bash
   # Download PDFs
   python run_pipeline.py --step download
   
   # Process documents
   python run_pipeline.py --step process
   
   # Set up retrieval
   python run_pipeline.py --step retrieve
   
   # Set up generation
   python run_pipeline.py --step generate
   
   # Run evaluation
   python run_pipeline.py --step evaluate
   ```

## Python API Example

```python
from src.data_processing.download_pdfs import download_all_pdfs
from src.data_processing.ingest import process_all_documents
from src.retrieval.retrieve import build_index, search_index
from src.generation.generate import FinancialQAModel
from src.evaluation.evaluate import RAGEvaluator

# Step 1: Download PDFs (if needed)
download_all_pdfs(
    document_info_path="data/financebench_document_information.jsonl",
    output_dir="data/raw_pdfs"
)

# Step 2: Process documents
process_all_documents(
    document_info_path="data/financebench_document_information.jsonl",
    pdf_dir="data/raw_pdfs",
    output_dir="data/processed_chunks",
    chunk_strategies=["fixed_256_overlap32", "fixed_512_overlap64"],
    extract_method="unstructured"
)

# Step 3: Build retrieval index
from src.retrieval.retrieve import load_chunks_from_directory
chunks = load_chunks_from_directory("data/processed_chunks")
vector_store = build_index(
    chunks=chunks,
    embedding_model="BAAI/bge-base-en-v1.5",
    store_type="faiss",
    index_name="financebench_index"
)

# Step 4: Set up generation model
qa_model = FinancialQAModel(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True  # Use 4-bit quantization for lower memory usage
)

# Step 5: Evaluate on a few examples
evaluator = RAGEvaluator()

# Load some test questions
import json
def load_jsonl(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

qa_data = load_jsonl("data/financebench_open_source.jsonl")[:3]  # First 3 examples

questions = [item["question"] for item in qa_data]
ground_truth_answers = [item["answer"] for item in qa_data]
ground_truth_evidence_list = []
for item in qa_data:
    evidence = []
    if "evidence" in item:
        for ev in item["evidence"]:
            if isinstance(ev, dict) and "evidence_text" in ev:
                evidence.append(ev["evidence_text"])
    ground_truth_evidence_list.append(evidence)

# Retrieve and generate
retrieved_chunks_list = []
for question in questions:
    chunks = search_index(
        vector_store=vector_store,
        query=question,
        embedding_model="BAAI/bge-base-en-v1.5",
        k=5
    )
    retrieved_chunks_list.append(chunks)

generated_answers = []
for question, contexts in zip(questions, retrieved_chunks_list):
    result = qa_model.answer_question(question, contexts)
    generated_answers.append(result["answer"])

# Evaluate
results, aggregated = evaluator.run_evaluation_pipeline(
    questions=questions,
    ground_truth_answers=ground_truth_answers,
    ground_truth_evidence_list=ground_truth_evidence_list,
    retrieved_chunks_list=retrieved_chunks_list,
    generated_answers=generated_answers
)

# Save results
evaluator.save_results(results, aggregated, "results/example_run")

print("Evaluation completed! Check the results directory for outputs.")
```

## Configuration

The project uses YAML configuration files located in the `configs/` directory. See `configs/base_config.yaml` for an example.

## Expected Outputs

After running the pipeline, you'll find:

- `data/raw_pdfs/` - Downloaded PDF files
- `data/processed_chunks/` - Processed and chunked text documents
- `financebench_index.*` - Vector store index files
- `results/` - Evaluation results including:
  - Detailed results CSV (per-question metrics)
  - Aggregated results JSON (summary statistics)
  - Figures directory with plots

## Customization

You can customize the pipeline by:

1. Changing the embedding model in `configs/base_config.yaml`
2. Modifying chunking strategies
3. Adjusting retrieval parameters (k value, etc.)
4. Using different LLMs for generation
5. Changing evaluation metrics or adding new ones

## Troubleshooting

Common issues:

1. **Out of memory**: Reduce batch size or use a smaller model
2. **Missing dependencies**: Install required packages with `pip install -r requirements.txt`
3. **PDF download failures**: Check internet connection and SEC access restrictions
4. **Model loading errors**: Ensure you have sufficient RAM/VRAM and try a smaller model

For detailed troubleshooting, see the troubleshooting section in the main README.