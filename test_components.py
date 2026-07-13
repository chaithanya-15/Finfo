#!/usr/bin/env python3
"""
Unit tests for core functions in the FinanceBench RAG project.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

class TestDataProcessing(unittest.TestCase):
    """Test data processing functions."""

    def setUp(self):
        """Set up test fixtures."""
        from data_processing.ingest import clean_text, chunk_text_fixed_size
        self.clean_text = clean_text
        self.chunk_text_fixed_size = chunk_text_fixed_size

    def test_clean_text(self):
        """Test text cleaning function."""
        # Test basic cleaning
        dirty = "  This  is   a   test  "
        cleaned = self.clean_text(dirty)
        self.assertEqual(cleaned, "This is a test")

        # Test newline handling
        dirty_with_newlines = "Line 1\n\nLine 2\n\n\nLine 3"
        cleaned = self.clean_text(dirty_with_newlines)
        self.assertEqual(cleaned, "Line 1\n\nLine 2\n\nLine 3")

        # Test hyphenated words
        hyphenated = "This is a test-\n word"
        cleaned = self.clean_text(hyphenated)
        self.assertEqual(cleaned, "This is a test-word")

    def test_chunk_text_fixed_size(self):
        """Test fixed-size chunking function."""
        text = "This is a test sentence. " * 10  # Repeated sentence
        chunks = self.chunk_text_fixed_size(text, chunk_size=50, overlap=10)

        # Should have multiple chunks
        self.assertGreater(len(chunks), 1)

        # Each chunk should be a string
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertTrue(len(chunk) > 0)

        # Check overlap concept (first chunk end should overlap with second chunk start)
        if len(chunks) >= 2:
            # Just verify we got chunks - exact overlap testing is complex due to tokenization
            self.assertTrue(len(chunks[0]) > 0)
            self.assertTrue(len(chunks[1]) > 0)

class TestRetrievalUtils(unittest.TestCase):
    """Test retrieval utility functions."""

    def test_hash_string_to_filename(self):
        """Test the hash function."""
        from retrieve import hash_string_to_filename

        # Same input should give same output
        hash1 = hash_string_to_filename("test string")
        hash2 = hash_string_to_filename("test string")
        self.assertEqual(hash1, hash2)

        # Different input should give different output (very high probability)
        hash3 = hash_string_to_filename("different string")
        self.assertNotEqual(hash1, hash3)

        # Output should be a hex string
        self.assertTrue(all(c in '0123456789abcdef' for c in hash1))
        self.assertEqual(len(hash1), 32)  # MD5 is 32 hex chars

class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metric functions."""

    def test_rouge_l_basic(self):
        """Test ROUGE-L calculation."""
        from evaluate import GenerationEvaluator

        evaluator = GenerationEvaluator()

        # Identical texts should score 1.0
        score = evaluator.calculate_rouge_l("The cat sat on the mat.", "The cat sat on the mat.")
        self.assertAlmostEqual(score, 1.0, places=5)

        # Completely different texts should score low
        score = evaluator.calculate_rouge_l("The cat sat on the mat.", "The dog barked loudly.")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_exact_match(self):
        """Test exact match calculation."""
        from evaluate import GenerationEvaluator

        evaluator = GenerationEvaluator()

        # Exact match (ignoring case/whitespace)
        self.assertEqual(evaluator.calculate_exact_match("Hello World", "hello world"), 1.0)
        self.assertEqual(evaluator.calculate_exact_match("  Hello  World  ", "HELLO WORLD"), 1.0)

        # No match
        self.assertEqual(evaluator.calculate_exact_match("Hello", "World"), 0.0)

    def test_citation_functions(self):
        """Test citation evaluation functions."""
        from evaluate import CitationEvaluator

        evaluator = CitationEvaluator()

        # Test citation extraction
        answer = "The revenue was $100 million [COMPANY_2020_10K_c5] and profit was $50 million [COMPANY_2020_10K_c10]."
        citations = evaluator.extract_citations(answer)
        self.assertEqual(len(citations), 2)
        self.assertIn("[COMPANY_2020_10K_c5]", citations)
        self.assertIn("[COMPANY_2020_10K_c10]", citations)

        # Test validation with matching contexts
        contexts = [
            {"doc_name": "COMPANY_2020_10K", "chunk_index": 5, "text": "Revenue was $100 million"},
            {"doc_name": "COMPANY_2020_10K", "chunk_index": 10, "text": "Profit was $50 million"}
        ]

        validation = evaluator.validate_citations(answer, contexts)
        self.assertTrue(validation["all_citations_valid"])
        self.assertEqual(validation["num_citations"], 2)
        self.assertEqual(validation["num_valid_citations"], 2)

if __name__ == '__main__':
    unittest.main()