#!/usr/bin/env python3
#!/usr/bin/env python
"""
Simple verification script for the FinanceBench RAG project.
"""

import os
import sys
import logging

def check_structure():
    """Check that the project structure is correct."""
    print("Checking project structure...")

    required_dirs = [
        "data",
        "data/raw_pdfs",
        "data/processed_chunks",
        "src",
        "src/data_processing",
        "src/retrieval",
        "src/generation",
        "src/evaluation",
        "src/utils",
        "configs",
        "results",
        "results/figures",
        "report"
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)

    if missing_dirs:
        print("❌ Missing directories:")
        for d in missing_dirs:
            print(f"  - {d}")
        return False
    else:
        print("✅ All required directories present")
        return True

def check_files():
    """Check that key files exist."""
    print("\nChecking for key files...")

    required_files = [
        "requirements.txt",
        "README.md",
        "run_pipeline.py",
        "configs/base_config.yaml",
        "src/data_processing/download_pdfs.py",
        "src/data_processing/ingest.py",
        "src/retrieval/retrieve.py",
        "src/generation/generate.py",
        "src/evaluation/evaluate.py"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing files:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    else:
        print("✅ All required files present")
        return True

def check_data_files():
    """Check if data files exist (warn if missing)."""
    print("\nChecking for data files...")

    data_files = [
        "data/financebench_document_information.jsonl",
        "data/financebench_open_source.jsonl"
    ]

    missing_files = []
    for file_path in data_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("⚠️  Data files not found (these should be provided):")
        for f in missing_files:
            print(f"  - {f}")
        print("   You'll need to obtain the FinanceBench dataset to run the full pipeline.")
        return False
    else:
        print("✅ Data files present")
        return True

def check_python_deps():
    """Check if key Python dependencies are available."""
    print("\nChecking Python dependencies...")

    required_packages = [
        "pandas",
        "numpy",
        "torch",
        "transformers",
        "sentence_transformers",
        "faiss",
        "chromadb",
        "yaml",
        "tqdm"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == "yaml":
                import yaml
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("⚠️  Some packages not installed (install with uv pip install -r requirements.txt):")
        for p in missing_packages:
            print(f"  - {p}")
        return False
    else:
        print("✅ Core dependencies available")
        return True

def main():
    """Run all checks."""
    print("=" * 50)
    print("FinanceBench RAG Project - Setup Verification")
    print("=" * 50)

    checks = [
        check_structure,
        check_files,
        check_data_files,
        check_python_deps
    ]

    results = []
    for check in checks:
        results.append(check())

    print("\n" + "=" * 50)
    if all(results):
        print("All checks passed. The project is ready to use.")
        print("\nNext steps:")
        print("1. Obtain the FinanceBench dataset files")
        print("2. Run: python run_pipeline.py --step all")
        print("3. Or run individual steps as needed")
    else:
        print("❌ Some checks failed. Please address the issues above.")
    print("=" * 50)

    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())