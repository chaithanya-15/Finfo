#!/usr/bin/env bash
# =============================================================================
# FinanceBench RAG Project - Quick Start Script
# =============================================================================
#
# This script provides a quick way to test the RAG pipeline on a small
# subset of data to verify everything is working correctly.
#
# Usage: ./quick_start.sh [options]
#
# Options:
#   --skip-download   Skip PDF download step
#   --skip-process    Skip document processing step
#   --help            Show this help message

set -euo pipefail

# Default values
SKIP_DOWNLOAD=false
SKIP_PROCESS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-process)
            SKIP_PROCESS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--skip-download] [--skip-process]"
            echo "  --skip-download   Skip PDF download step"
            echo "  --skip-process    Skip document processing step"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to print section headers
print_header() {
    echo "================================================================================"
    echo "=$1"
    echo "================================================================================"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
if ! command_exists uv; then
    echo "Error: uv command not found. Please install uv from https://astral.sh/uv/"
    exit 1
fi

if ! command_exists python; then
    echo "Error: python command not found"
    exit 1
fi

# Change to script directory
cd "$(dirname "$0")"

# Create virtual environment with Python 3.14 if doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.14..."
    uv python install 3.14 3.14..."
    uv python install 3.14
    uv venv --python 3.14
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    echo "Activating virtual environment (Windows)..."
    source .venv/Scripts/activate
fi

# Install requirements if not already done
if [ ! -f ".requirements_installed" ]; then
    echo "Installing requirements..."
    uv pip install -r requirements.txt
    touch .requirements_installed
fi

# Create necessary directories
mkdir -p data/raw_pdfs data/processed_chunks results

# Step 1: Download PDFs (optional)
if [ "$SKIP_DOWNLOAD" = false ]; then
    print_header "STEP 1: DOWNLOADING PDFS"
    echo "Downloading PDFs (this may take a while)..."
    python -m src.data_processing.download_pdfs
else
    print_header "STEP 1: DOWNLOADING PDFS (SKIPPED)"
fi

# Step 2: Process documents (optional)
if [ "$SKIP_PROCESS" = false ]; then
    print_header "STEP 2: PROCESSING DOCUMENTS"
    echo "Processing documents (extracting text, cleaning, chunking)..."
    python -m src.data_processing.ingest
else
    print_header "STEP 2: PROCESSING DOCUMENTS (SKIPPED)"
fi

# Step 3: Build retrieval index
print_header "STEP 3: BUILDING RETRIEVAL INDEX"
echo "Building vector index from processed chunks..."
python -m src.retrieval.retrieve

# Step 4: Run evaluation on small subset
print_header "STEP 4: RUNNING EVALUATION"
echo "Running evaluation on a small subset of questions..."
python run_pipeline.py --step evaluate --config configs/base_config.yaml

echo
echo "================================================================================"
echo "= QUICK START COMPLETE!"
echo "================================================================================"
echo
echo "Results are available in the 'results/' directory."
echo "To run the full evaluation, remove the --max_examples limit in the config."
echo
echo "Next steps:"
echo "1. Review results in results/"
echo "2. Modify configs/base_config.yaml for different experiments"
echo "3. Run full pipeline: python run_pipeline.py --step all"
echo "4. Consult EXAMPLE_USAGE.md for more detailed usage instructions"