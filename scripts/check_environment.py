#!/usr/bin/env python3
"""
Environment verification script for FinanceBench RAG project.
Checks dependencies, GPU availability, and basic functionality.
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("=== Python Version ===")
    version = sys.version
    print(f"Python {version}")
    version_info = sys.version_info
    if version_info.major == 3 and version_info.minor >= 14:
        print("✓ Python version OK (3.14+)")
        return True
    else:
        print("✗ Python 3.14+ required")
        return False

def check_packages(required_packages):
    """Check if required packages are installed."""
    print("\n=== Package Check ===")
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (NOT INSTALLED)")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: uv pip install " + " ".join(missing))
        return False
    else:
        print("\nAll required packages installed!")
        return True

def check_gpu():
    """Check GPU availability."""
    print("\n=== GPU Check ===")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA available: {gpu_count} GPU(s)")
            print(f"  GPU 0: {gpu_name}")
            return True
        else:
            print("⚠ CUDA not available (CPU-only mode)")
            return False
    except ImportError:
        print("⚠ PyTorch not installed - cannot check GPU")
        return False

def check_disk_space(path="."):
    """Check available disk space."""
    print("\n=== Disk Space Check ===")
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free // (2**30)
        total_gb = total // (2**30)
        print(f"Disk space: {free_gb} GB free of {total_gb} GB total")
        if free_gb < 10:
            print("⚠ Warning: Less than 10 GB free space")
            return False
        else:
            print("✓ Sufficient disk space available")
            return True
    except Exception as e:
        print(f"⚠ Could not check disk space: {e}")
        return False

def check_data_files():
    """Check if required data files exist."""
    print("\n=== Data Files Check ===")
    required_files = [
        "data/financebench_document_information.jsonl",
        "data/financebench_open_source.jsonl"
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            # Count lines
            try:
                with open(file_path, 'r') as f:
                    lines = sum(1 for _ in f)
                print(f"✓ {file_path} ({lines} entries)")
            except Exception as e:
                print(f"✓ {file_path} (could not count lines: {e})")
        else:
            print(f"✗ {file_path} (NOT FOUND)")
            missing_files.append(file_path)

    if missing_files:
        print(f"\nMissing data files: {', '.join(missing_files)}")
        print("Please ensure the FinanceBench dataset is downloaded.")
        return False
    else:
        print("\nAll required data files present!")
        return True

def check_directory_structure():
    """Check if project directory structure exists."""
    print("\n=== Directory Structure ===")
    required_dirs = [
        "src",
        "src/data_processing",
        "src/retrieval",
        "src/generation",
        "src/evaluation",
        "configs",
        "data",
        "results"
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (MISSING)")
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"\nMissing directories: {', '.join(missing_dirs)}")
        return False
    else:
        print("\nDirectory structure OK!")
        return True

def test_imports():
    """Test that project modules can be imported."""
    print("\n=== Module Import Test ===")
    sys.path.append(str(Path("src").absolute()))

    modules_to_test = [
        "data_processing.download_pdfs",
        "data_processing.ingest",
        "retrieval.retrieve",
        "generation.generate",
        "evaluation.evaluate"
    ]

    failed_imports = []
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name} - {e}")
            failed_imports.append((module_name, str(e)))

    if failed_imports:
        print(f"\nFailed imports: {len(failed_imports)}")
        for module, error in failed_imports:
            print(f"  {module}: {error}")
        return False
    else:
        print("\nAll modules imported successfully!")
        return True

def main():
    """Run all checks."""
    print("FinanceBench RAG Project - Environment Verification")
    print("=" * 55)

    checks = [
        check_python_version,
        lambda: check_packages([
            "torch", "transformers", "sentence_transformers",
            "faiss_cpu", "chromadb", "langchain", "llama_index",
            "pandas", "numpy", "scikit-learn", "rouge_score",
            "rapidfuzz", "tqdm", "pyyaml", "matplotlib", "seaborn",
            "pypdf", "pdfplumber", "unstructured", "bitsandbytes",
            "accelerate", "datasets", "ragas"
        ]),
        check_gpu,
        check_disk_space,
        check_directory_structure,
        check_data_files,
        test_imports
    ]

    passed = 0
    total = len(checks)

    for check in checks:
        try:
            if check():
                passed += 1
        except Exception as e:
            print(f"✗ Check failed with exception: {e}")

    print("\n" + "=" * 55)
    print(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All checks passed! The environment is ready.")
        return 0
    else:
        print("⚠ Some checks failed. Please address the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())