# FinanceBench RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system for the FinanceBench dataset, following the guidelines from the graded project implementation guide.

## Project Structure

```
project/
├── README.md                 # setup + how to reproduce experiments
├── requirements.txt
├── data/
│   ├── raw_pdfs/             # downloaded PDF files
│   └── processed_chunks/     # processed and chunked text
├── src/
│   ├── data_processing/      # PDF download, text extraction, cleaning, chunking
│   ├── retrieval/            # embedding, indexing, search
│   ├── generation/           # LLM loading, prompting, answer generation
│   ├── evaluation/           # metrics calculation, evaluation pipeline
│   └── utils/                # utility functions, config handling
├── configs/                  # YAML config files for experiments
├── results/                  # experiment results
│   ├── metrics.csv
│   └── figures/
└── report/
    └── report.pdf            # final report
```

## Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create a virtual environment with Python 3.14**:
   ```bash
   uv python install 3.14
   uv venv --python 3.14
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Download the FinanceBench dataset** (if not already present):
   - The dataset files should be in the `data/` directory:
     - `financebench_document_information.jsonl`
     - `financebench_open_source.jsonl`

## Usage

### Step 1: Download PDFs
```bash
python -m src.data_processing.download_pdfs
```

### Step 2: Process documents (extract text, clean, chunk)
```bash
python -m src.data_processing.ingest
```

### Step 3: Run evaluation pipeline
```bash
python -m src.evaluation.evaluate
```

Or use the provided pipeline runner:
```bash
python run_pipeline.py --step all --config configs/base_config.yaml
```

## Configuration

Experiment configurations are stored in the `configs/` directory as YAML files. Each configuration represents a different ablation study condition.

## Evaluation

The evaluation pipeline computes:
- Retrieval metrics (Recall@k, Precision@k, MRR)
- Generation metrics (lexical/semantic overlap, LLM-as-judge scores)
- Citation validation metrics
- Error analysis by question type

## Implementation Notes

This implementation follows all constraints from the project brief:
- Uses only free, locally executable models (via uv and Python 3.14)
- Requires citations in generated answers
- Uses only the provided corpus for answers
- Compares multiple configurations (ablation study)
- Avoids common pitfalls like API key leakage, silent truncation, etc.