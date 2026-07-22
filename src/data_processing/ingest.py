#!/usr/bin/env python3
"""
Document ingestion module for FinanceBench RAG pipeline.
Handles PDF text extraction, cleaning, and chunking.
"""

import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import pypdf
import pdfplumber
# unstructured is imported inside extract_text_unstructured. It pulls in layout-detection
# models that take minutes to load and are only needed for that one extraction method.
import tiktoken
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing headers/footers, fixing hyphenation, etc.

    Args:
        text: Raw text extracted from PDF

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Fix hyphenated words split across lines
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # Remove standalone numbers that might be page numbers (simple heuristic)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just numbers (likely page numbers)
        if stripped.isdigit() and len(stripped) <= 4:
            continue
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)
    return text.strip()


def extract_text_pypdf(pdf_path: str) -> str:
    """
    Extract text using PyPDF (basic extraction).

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"Error extracting text with PyPDF from {pdf_path}: {e}")
        return ""
    return text


def extract_text_pdfplumber(pdf_path: str) -> tuple[str, list]:
    """
    Extract text and tables using pdfplumber (better for tables).

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (text, tables)
    """
    text = ""
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

                # Extract tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table:  # Skip empty tables
                        tables.append(table)
    except Exception as e:
        logger.error(f"Error extracting text with pdfplumber from {pdf_path}: {e}")
        return "", []
    return text, tables


def extract_text_unstructured(pdf_path: str) -> tuple[str, list]:
    """
    Extract text and detect structural elements using unstructured.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (text, elements)
    """
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.documents.elements import Table, Title, NarrativeText

        elements = partition_pdf(
            pdf_path,
            strategy="hi_res",  # High-resolution strategy for better structure detection
            infer_table_structure=True  # Try to extract table structure
        )

        text_parts = []
        table_data = []

        for element in elements:
            if isinstance(element, (Title, NarrativeText)):
                text_parts.append(str(element))
            elif isinstance(element, Table):
                # Convert table to text representation
                table_text = str(element.metadata.text_as_html) if hasattr(element.metadata, 'text_as_html') else str(element)
                table_data.append(table_text)
                text_parts.append(f"\n{table_text}\n")  # Include table in text flow

        text = "\n\n".join(text_parts)
        return text, table_data
    except Exception as e:
        logger.error(f"Error extracting text with unstructured from {pdf_path}: {e}")
        return "", []


def extract_document_text(pdf_path: str, method: str = "unstructured",
                          cache_dir: str = "data/extracted_text") -> tuple[str, list]:
    """
    Extract text from PDF using specified method.

    Results are cached per document and method. Every chunking strategy re-reads the same
    PDF, and extraction dominates the runtime, so without the cache a three-strategy run
    over the corpus pays for extraction three times.

    Args:
        pdf_path: Path to PDF file
        method: Extraction method ('pypdf', 'pdfplumber', or 'unstructured')
        cache_dir: Directory holding cached extractions, set to None to disable

    Returns:
        Tuple of (text, tables_or_elements)
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return "", []

    cache_path = None
    if cache_dir:
        doc_name = Path(pdf_path).stem
        cache_path = Path(cache_dir) / method / f"{doc_name}.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                return cached["text"], cached["extra"]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Ignoring unreadable cache {cache_path}: {e}")

    text, extra = _extract_uncached(pdf_path, method)

    if cache_path is not None and text.strip():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({"text": text, "extra": extra}, f)

    return text, extra


def _extract_uncached(pdf_path: str, method: str) -> tuple[str, list]:
    """
    Run the requested extractor without consulting the cache.

    Args:
        pdf_path: Path to PDF file
        method: Extraction method ('pypdf', 'pdfplumber', or 'unstructured')

    Returns:
        Tuple of (text, tables_or_elements)
    """
    if method == "pypdf":
        text = extract_text_pypdf(pdf_path)
        return text, []
    elif method == "pdfplumber":
        text, tables = extract_text_pdfplumber(pdf_path)
        return text, tables
    elif method == "unstructured":
        text, elements = extract_text_unstructured(pdf_path)
        return text, elements
    else:
        raise ValueError(f"Unknown extraction method: {method}")


def chunk_text_fixed_size(text: str, chunk_size: int = 512,
                          overlap: int = 64, encoding_name: str = "cl100k_base") -> List[str]:
    """
    Split text into fixed-size chunks with overlap.

    Args:
        text: Text to chunk
        chunk_size: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks
        encoding_name: Tokenizer encoding to use

    Returns:
        List of text chunks
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except KeyError:
        # Fallback to cl100k_base if specified encoding not found
        encoding = tiktoken.get_encoding("cl100k_base")

    # Encode text to tokens
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        # Calculate end position
        end = min(start + chunk_size, len(tokens))

        # Extract chunk tokens
        chunk_tokens = tokens[start:end]

        # Decode back to text
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Stop once the final chunk has reached the end. Advancing by chunk_size - overlap
        # otherwise leaves start stuck at len(tokens) - overlap, which never satisfies a
        # start-based guard, so the last window repeats forever.
        if end >= len(tokens):
            break

        # Move start position (accounting for overlap)
        start = end - overlap

    return chunks


def chunk_by_structure(text: str, elements: list = None,
                       max_chunk_size: int = 512) -> List[Dict[str, Any]]:
    """
    Chunk text respecting structural boundaries (sections, tables).

    Args:
        text: Text to chunk
        elements: Structural elements from unstructured (optional)
        max_chunk_size: Maximum tokens per chunk

    Returns:
        List of chunks with metadata
    """
    # Structure-aware chunking packs whole sentences up to the size limit, so a chunk never
    # ends mid-sentence the way fixed-size windows do. Paragraph breaks are used as unit
    # boundaries when they survive cleaning; where the text has been flattened to a single
    # block, sentences become the units. Any unit still larger than the limit (for example a
    # wide table rendered as one long line) is split into fixed token windows as a fallback so
    # no single chunk overflows.
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    raw_sections = re.split(r'\n\s*\n', text)
    sections = []
    for raw in raw_sections:
        raw = raw.strip()
        if not raw:
            continue
        if len(encoding.encode(raw)) <= max_chunk_size:
            sections.append(raw)
            continue
        # Oversized section: break into sentences.
        for sentence in re.split(r'(?<=[.!?])\s+', raw):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(encoding.encode(sentence)) <= max_chunk_size:
                sections.append(sentence)
            else:
                # Still too large (e.g. a flattened table): fall back to token windows.
                sections.extend(chunk_text_fixed_size(sentence, chunk_size=max_chunk_size, overlap=0))

    chunks = []
    current_chunk = []
    current_length = 0

    for section in sections:
        section_tokens = encoding.encode(section)
        section_length = len(section_tokens)

        # If adding this section would exceed max size, finalize current chunk
        if current_length + section_length > max_chunk_size and current_chunk:
            # Create chunk from current sections
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "length": current_length
            })
            # Start new chunk with overlap (keep last section for context)
            overlap_sections = current_chunk[-1:] if current_chunk else []
            current_chunk = overlap_sections + [section]
            current_length = sum(len(encoding.encode(s)) for s in current_chunk)
        else:
            # Add section to current chunk
            current_chunk.append(section)
            current_length += section_length

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "length": current_length
        })

    return chunks


def process_document(doc_info: Dict[str, Any],
                     pdf_dir: str = "data/raw_pdfs",
                     output_dir: str = "data/processed_chunks",
                     chunk_strategy: str = "fixed_512_overlap64",
                     extract_method: str = "unstructured") -> List[Dict[str, Any]]:
    """
    Process a single document: extract text, clean, chunk, and add metadata.

    Args:
        doc_info: Dictionary containing document metadata
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save processed chunks
        chunk_strategy: Chunking strategy to use
        extract_method: Text extraction method to use

    Returns:
        List of chunk dictionaries with metadata
    """
    doc_name = doc_info['doc_name']
    pdf_path = os.path.join(pdf_dir, f"{doc_name}.pdf")

    # Check if PDF exists
    if not os.path.exists(pdf_path):
        logger.warning(f"PDF not found for {doc_name}: {pdf_path}")
        return []

    # Extract text
    logger.info(f"Processing {doc_name} using {extract_method} extraction")
    text, extra_data = extract_document_text(pdf_path, method=extract_method)

    if not text.strip():
        logger.warning(f"No text extracted from {doc_name}")
        return []

    # Clean text
    cleaned_text = clean_text(text)

    # Chunk text based on strategy
    chunks = []
    if chunk_strategy.startswith("fixed_"):
        # Extract chunk size and overlap from strategy name
        # Format: fixed_512_overlap64 or fixed_256 etc.
        parts = chunk_strategy.split("_")
        chunk_size = int(parts[1])
        overlap = int(parts[3]) if len(parts) > 3 and "overlap" in parts[2] else chunk_size // 8

        text_chunks = chunk_text_fixed_size(cleaned_text, chunk_size=chunk_size, overlap=overlap)
        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                "text": chunk_text,
                "chunk_index": i
            })
    elif chunk_strategy == "structure":
        chunks = chunk_by_structure(cleaned_text, elements=extra_data)
        # Add chunk index
        for i, chunk in enumerate(chunks):
            chunk["chunk_index"] = i
    else:
        # Default to fixed size chunking
        text_chunks = chunk_text_fixed_size(cleaned_text)
        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                "text": chunk_text,
                "chunk_index": i
            })

    # Add metadata to each chunk
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "chunk_id": f"{doc_name}_p{doc_info.get('doc_period', 'unknown')}_c{i:04d}",
            "doc_name": doc_name,
            "company": doc_info.get("company", "unknown"),
            "gics_sector": doc_info.get("gics_sector", "unknown"),
            "doc_type": doc_info.get("doc_type", "unknown"),
            "doc_period": doc_info.get("doc_period", "unknown"),
            "page_number": doc_info.get("evidence_page_num", 0),  # This is approximate
            "section": "unknown",  # Would need more sophisticated parsing
            "text": chunk["text"],
            "chunk_strategy": chunk_strategy,
            "chunk_index": chunk.get("chunk_index", i),
            "token_count": len(
                __import__('tiktoken').get_encoding("cl100k_base").encode(chunk["text"])
            ) if chunk["text"] else 0
        }
        processed_chunks.append(chunk_data)

    # Save chunks to file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{doc_name}_{chunk_strategy}.jsonl")
    with open(output_file, 'w') as f:
        for chunk in processed_chunks:
            f.write(json.dumps(chunk) + '\n')

    logger.info(f"Processed {doc_name}: {len(processed_chunks)} chunks saved to {output_file}")
    return processed_chunks


def process_all_documents(document_info_path: str = "data/financebench_document_information.jsonl",
                          pdf_dir: str = "data/raw_pdfs",
                          output_dir: str = "data/processed_chunks",
                          chunk_strategies: List[str] = None,
                          extract_method: str = "unstructured") -> Dict[str, List[Dict[str, Any]]]:
    """
    Process all documents in the dataset.

    Args:
        document_info_path: Path to JSONL file containing document metadata
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save processed chunks
        chunk_strategies: List of chunking strategies to apply
        extract_method: Text extraction method to use

    Returns:
        Dictionary mapping doc_name to list of processed chunks
    """
    if chunk_strategies is None:
        chunk_strategies = ["fixed_256_overlap32", "fixed_512_overlap64", "structure"]

    # Load document information
    docs_data = []
    with open(document_info_path, 'r') as f:
        for line in f:
            docs_data.append(json.loads(line))

    docs_df = pd.DataFrame(docs_data)
    # Remove duplicates
    docs_df = docs_df.drop_duplicates(subset=['doc_name'])

    all_results = {}

    for _, doc_info in docs_df.iterrows():
        doc_name = doc_info['doc_name']
        print(f"\nProcessing document: {doc_name}")

        doc_results = {}
        for strategy in chunk_strategies:
            print(f"  Using strategy: {strategy}")
            chunks = process_document(
                doc_info.to_dict(),
                pdf_dir=pdf_dir,
                output_dir=output_dir,
                chunk_strategy=strategy,
                extract_method=extract_method
            )
            doc_results[strategy] = chunks

        all_results[doc_name] = doc_results

    return all_results


if __name__ == "__main__":
    # When run as a script, process all documents with default settings
    process_all_documents()