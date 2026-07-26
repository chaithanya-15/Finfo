# FinanceBench RAG Project: example usage

Examples of how to run the pipeline and call it from Python. For the full setup and the
ablation sweep, see `README.md`.

## Quick start

Install dependencies (see the README for the `llama-cpp-python` build flags per platform):

```bash
uv pip install -r requirements.txt
```

Run the whole pipeline for a single configuration:

```bash
python run_pipeline.py --step all --config configs/base_config.yaml
```

Or run individual steps:

```bash
python run_pipeline.py --step download    # download the source PDFs
python run_pipeline.py --step process     # extract, clean, chunk
python run_pipeline.py --step retrieve    # build the vector index
python run_pipeline.py --step generate    # load the generator
python run_pipeline.py --step evaluate    # score retrieval and generation
```

To run the ablation across all configurations instead, use `python run_experiments.py`
(see the README).

## Python API

```python
import json
from src.data_processing.ingest import process_document
from src.retrieval.retrieve import build_index, load_chunks_from_directory, search_index
from src.generation.generate import create_qa_pipeline
from src.evaluation.evaluate import create_evaluation_pipeline

# 1. Process a document into chunks (extraction is cached under data/extracted_text/).
docs = {json.loads(l)["doc_name"]: json.loads(l)
        for l in open("data/financebench_document_information.jsonl")}
process_document(docs["APPLE_2022_10K"], pdf_dir="data/raw_pdfs",
                 output_dir="data/processed_chunks",
                 chunk_strategy="fixed_512_overlap64", extract_method="pdfplumber")

# 2. Build a FAISS index from the processed chunks.
chunks = load_chunks_from_directory("data/processed_chunks")
chunks = [c for c in chunks if c["chunk_strategy"] == "fixed_512_overlap64"]
store = build_index(chunks=chunks, embedding_model="BAAI/bge-base-en-v1.5",
                    store_type="faiss", index_name="indexes/demo")

# 3. Load a generator. Model names resolve to local GGUF weights (see LOCAL_MODEL_PATTERNS
#    in src/generation/generate.py); "gemma-4-12B" is the alternative generator.
model = create_qa_pipeline(model_name="Qwen3.5-4B")

# 4. Retrieve and answer.
question = "What were Apple's total net sales in FY2022?"
contexts = search_index(store, question, "BAAI/bge-base-en-v1.5", k=5)
result = model.answer_question(question, contexts)
print(result["answer"])            # cited answer, e.g. "... $394,328 million [APPLE_2022_10K_c17]."
print(result["valid_citations"])   # citations that match a retrieved chunk

# 5. Score a batch of questions.
evaluator = create_evaluation_pipeline()
qa = [json.loads(l) for l in open("data/financebench_open_source.jsonl")][:3]
questions = [q["question"] for q in qa]
answers = [q["answer"] for q in qa]
evidence = [[e["evidence_text"] for e in q.get("evidence", []) if isinstance(e, dict)] for q in qa]
retrieved = [search_index(store, q, "BAAI/bge-base-en-v1.5", k=5) for q in questions]
generated = [model.answer_question(q, c)["answer"] for q, c in zip(questions, retrieved)]
results, aggregated = evaluator.run_evaluation_pipeline(
    questions=questions, ground_truth_answers=answers,
    ground_truth_evidence_list=evidence, retrieved_chunks_list=retrieved,
    generated_answers=generated)
evaluator.save_results(results, aggregated, "results/example_run")
```

### Reproducing the best configuration

Step 4 above uses plain dense search, which is the `baseline` arm. The configuration the report
recommends is `rerank_filtered`: restrict candidates to the company named in the question, then
reorder a deep pool with a cross-encoder. Both compose, and the filter decides what the
cross-encoder is allowed to see.

```python
from src.retrieval.metadata_filter import load_company_list, filtered_search
from src.retrieval.rerank import build_reranker, rerank_search

companies = load_company_list("data/financebench_document_information.jsonl")

# Company filter alone (the `filtered_meta` arm).
contexts = filtered_search(store, question, "BAAI/bge-base-en-v1.5", 5, companies)

# Filter plus cross-encoder reranking (the `rerank_filtered` arm, best in the study).
# Downloads BAAI/bge-reranker-base (~1.1 GB) on first use; see README setup step 5.
reranker = build_reranker("BAAI/bge-reranker-base")
contexts = rerank_search(store, reranker, question, "BAAI/bge-base-en-v1.5", k=5,
                         pool=50, companies=companies)

result = model.answer_question(question, contexts)
```

Pass `companies=None` to rerank a plain dense pool instead of a company-filtered one. Load the
reranker before the generator: a cross-encoder on MPS is fine before llama.cpp takes Metal, but
not after.

## Configuration

Experiment settings are YAML files in `configs/`. `base_config.yaml` holds the reference
configuration; `experiments.yaml` lists the ablation runs as overrides on it.

## Outputs

- `data/raw_pdfs/`, `data/extracted_text/`, `data/processed_chunks/`: the corpus at each stage
- `indexes/`: FAISS index files, one per embedding model and chunk strategy
- `results/experiments/`: per-configuration metrics and `comparison.csv`
- `results/figures/`, `results/results_summary.md`: figures and tables for the report

## Notes

- Only free, locally executable models are used. Generation runs 4-bit or 8-bit GGUF weights
  through llama.cpp; on Apple silicon build it with Metal (`CMAKE_ARGS="-DGGML_METAL=on"`).
- Set `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` on macOS so faiss and torch do not
  clash over OpenMP; `run_experiments.py` sets these itself.
- About a fifth of the FinanceBench source URLs no longer resolve, so some documents will not
  download. See the corpus-coverage note in the README.
