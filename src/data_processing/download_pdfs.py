#!/usr/bin/env python3
"""
PDF downloader for FinanceBench dataset.
Handles downloading SEC filings with proper headers and error handling.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
from pathlib import Path


def load_jsonl(filepath: str) -> List[dict]:
    """Load JSONL file and return list of dictionaries."""
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]


def download_pdf(doc_name: str, url: str, out_dir: str,
                 retries: int = 3, backoff_factor: float = 1.0) -> Optional[str]:
    """
    Download a PDF file with retry logic and SEC-compliant headers.

    Args:
        doc_name: Name of the document (used for filename)
        url: URL to download the PDF from
        out_dir: Directory to save the PDF
        retries: Number of retry attempts
        backoff_factor: Backoff factor for retry delays

    Returns:
        Path to downloaded file or None if failed
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # SEC requires a descriptive User-Agent with contact email
    headers = {
        "User-Agent": "research-project contact@example.com"
    }

    # Define output path
    path = os.path.join(out_dir, f"{doc_name}.pdf")

    # Skip if file already exists
    if os.path.exists(path):
        print(f"File already exists: {path}")
        return path

    # Attempt download with retries
    for attempt in range(retries):
        try:
            print(f"Downloading {doc_name} (attempt {attempt + 1}/{retries})")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()  # Raise exception for bad status codes

            # Validate that response content is actually a PDF
            content = response.content
            if not content.startswith(b"%PDF-") or len(content) < 1024:
                raise ValueError("Downloaded content is not a valid PDF (missing %PDF- header or too small)")

            # Write content to file
            with open(path, "wb") as f:
                f.write(content)

            print(f"Successfully downloaded: {path}")
            return path

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:  # Don't sleep on last attempt
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"Waiting {sleep_time} seconds before retry...")
                time.sleep(sleep_time)

    print(f"Failed to download {doc_name} after {retries} attempts")
    return None


def download_all_pdfs(document_info_path: str = "data/financebench_document_information.jsonl",
                      output_dir: str = "data/raw_pdfs") -> Dict[str, Optional[str]]:
    """
    Download all PDFs from the document information file.

    Args:
        document_info_path: Path to JSONL file containing document metadata
        output_dir: Directory to save downloaded PDFs

    Returns:
        Dictionary mapping doc_name to download status (path or None)
    """
    # Load document information
    docs_data = load_jsonl(document_info_path)
    docs_df = pd.DataFrame(docs_data)

    # Remove duplicates to avoid re-downloading the same document
    docs_df = docs_df.drop_duplicates(subset=['doc_name'])

    print(f"Found {len(docs_df)} unique documents to download")

    # Download each document
    results = {}
    for _, row in docs_df.iterrows():
        doc_name = row['doc_name']
        url = row['doc_link']

        # Validate URL
        if not url or not isinstance(url, str) or not url.startswith('http'):
            print(f"Skipping invalid URL for {doc_name}: {url}")
            results[doc_name] = None
            continue

        result_path = download_pdf(doc_name, url, output_dir)
        results[doc_name] = result_path

        # Be respectful to SEC servers - rate limit requests
        time.sleep(1.0)  # 1 second between requests

    # Summary
    successful = sum(1 for v in results.values() if v is not None)
    total = len(results)
    print(f"\nDownload complete: {successful}/{total} files downloaded successfully")

    return results


if __name__ == "__main__":
    # When run as a script, download all PDFs
    download_all_pdfs()