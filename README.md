# FinanceBench RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system for the FinanceBench dataset, following the guidelines from the graded project implementation guide.

## Project Structure

```
project/
├── README.md                 # setup + how to reproduce experiments
├── EXAMPLE_USAGE.md          # runnable Python API examples
├── quick_start.sh            # end-to-end smoke run on a small subset
├── requirements.txt
├── run_experiments.py        # ablation sweep entry point
├── run_pipeline.py           # single-configuration pipeline runner
├── data/
│   ├── raw_pdfs/             # downloaded PDF files
│   └── processed_chunks/     # processed and chunked text
├── src/
│   ├── data_processing/      # PDF download, text extraction, cleaning, chunking
│   ├── retrieval/            # embedding, indexing, search
│   ├── generation/           # LLM loading, prompting, answer generation
│   └── evaluation/           # metrics, evaluation pipeline, figure generation
├── scripts/                  # helpers: prebuild_indexes, extract, reconcile, setup checks
├── tests/                    # pytest unit tests
├── configs/                  # YAML config files for experiments
├── results/                  # experiment results
│   ├── experiments/          # per-configuration metrics + comparison.csv
│   ├── results_summary.md    # tables generated from the sweep
│   └── figures/
└── report/
    └── report.md             # the experiment write-up
```

## Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh  # On Windows: irm https://astral.sh/uv/install.ps1 | iex
   ```

2. **Create a virtual environment**:
   ```bash
   uv venv --python 3.12
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
   Python 3.12 rather than 3.14, because faiss and chromadb do not publish 3.14 wheels.

3. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```
   The generator backend, `llama-cpp-python`, is built for the local accelerator. On Apple
   silicon that means Metal:
   ```bash
   CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python
   ```
   On Linux or Windows with an NVIDIA GPU, use `-DGGML_CUDA=on` instead; with no GPU, a plain
   `uv pip install llama-cpp-python` builds a CPU version. The embedding step selects MPS,
   CUDA, or CPU automatically, so the rest of the pipeline is unchanged across platforms.

4. **Fetch the generator weights.** Both generators are 4-bit or 8-bit GGUF files resolved
   from the Hugging Face cache (see `LOCAL_MODEL_PATTERNS` in `src/generation/generate.py`):
   ```bash
   huggingface-cli download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-UD-Q4_K_XL.gguf
   huggingface-cli download lmstudio-community/gemma-4-12B-it-GGUF gemma-4-12B-it-Q8_0.gguf
   ```
   Qwen3.5-4B is the baseline generator; gemma-4-12B is the second-generator ablation arm.
   An existing LM Studio copy of gemma-4-12B is reused if present.

5. **The FinanceBench metadata** is already in `data/`:
   `financebench_document_information.jsonl` and `financebench_open_source.jsonl`.

## Usage

### Step 1: Download the PDFs
```bash
python -m src.data_processing.download_pdfs
```
Expect around 282 of the 360 documents to arrive. See "Corpus coverage" below.

### Step 2: Run the full ablation sweep
```bash
python run_experiments.py
```
This extracts text (cached under `data/extracted_text/`), chunks each document under every
strategy the sweep needs, builds one vector index per embedding model and chunk strategy,
then retrieves, generates and scores. Results land in `results/experiments/`, one directory
per configuration plus a `comparison.csv` across all of them.

Useful variants:
```bash
python run_experiments.py --limit 20            # quick smoke run
python run_experiments.py --only baseline k_10  # named configurations
```

### Running a single configuration
```bash
python run_pipeline.py --step all --config configs/base_config.yaml
```

## Corpus coverage

FinanceBench's document manifest lists 360 filings, but 78 of the source URLs are dead:
they return timeouts, 404s or 403s from the companies' investor-relations hosts. The links
are broken at the source, so re-running the download does not recover them.

A further set of downloaded files are corrupt or empty, so text extraction succeeds for 263
documents. **36 of the 150 evaluation questions ask about a document that cannot be answered
from the local corpus.** `run_experiments.py` therefore reports every metric twice, over all
150 questions and over the 114 answerable ones. The first number describes the system as
deployed against this corpus; the second isolates retrieval and generation quality from the
missing data.

## Configuration

Experiment configurations are stored in the `configs/` directory as YAML files. Each configuration represents a different ablation study condition.

## Evaluation

The evaluation pipeline computes:
- Retrieval metrics (Recall@k, Precision@k, MRR)
- Generation metrics (ROUGE-L, embedding-based semantic similarity, exact match)
- Citation validation metrics (precision, recall, F1)
- Error analysis by question type

## Results and report

The full ablation has been run. Per-configuration metrics are in `results/experiments/`
(with `comparison.csv` across all runs), the figures and summary tables in `results/figures/`
and `results/results_summary.md`, and the write-up in `report/report.md`.

Raw PDFs, vector indexes, and the virtualenv are not tracked. Rebuild the corpus with
`python -m src.data_processing.download_pdfs`; indexes are rebuilt from the chunks by the sweep.

## Implementation Notes

This implementation follows all constraints from the project brief:
- Uses only free, locally executable models (via uv and Python 3.12)
- Requires citations in generated answers
- Uses only the provided corpus for answers
- Compares multiple configurations (ablation study)
- Avoids common pitfalls like API key leakage, silent truncation, etc.
