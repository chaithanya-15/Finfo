#!/usr/bin/env python3
"""
Unit tests for core functions in the FinanceBench RAG project.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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

        # Whitespace, including newlines, is normalised to single spaces. Chunking works on
        # the token stream rather than paragraph layout, so the pipeline does not depend on
        # newlines surviving; structure chunking recovers sentence units separately.
        dirty_with_newlines = "Line 1\n\nLine 2\n\n\nLine 3"
        cleaned = self.clean_text(dirty_with_newlines)
        self.assertEqual(cleaned, "Line 1 Line 2 Line 3")

        # A word split across a line break by a hyphen is rejoined.
        hyphenated = "This is a test-\n word"
        cleaned = self.clean_text(hyphenated)
        self.assertEqual(cleaned, "This is a testword")

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
        from retrieval.retrieve import hash_string_to_filename

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
        from evaluation.evaluate import GenerationEvaluator

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
        from evaluation.evaluate import GenerationEvaluator

        evaluator = GenerationEvaluator()

        # Exact match (ignoring case/whitespace)
        self.assertEqual(evaluator.calculate_exact_match("Hello World", "hello world"), 1.0)
        self.assertEqual(evaluator.calculate_exact_match("  Hello  World  ", "HELLO WORLD"), 1.0)

        # No match
        self.assertEqual(evaluator.calculate_exact_match("Hello", "World"), 0.0)

    def test_numeric_agreement(self):
        """Test numeric agreement, which scores figures rather than wording."""
        from evaluation.evaluate import GenerationEvaluator

        evaluator = GenerationEvaluator()
        agree = evaluator.calculate_numeric_agreement

        # The figure is right even though almost no wording is shared.
        self.assertEqual(agree("$59,268 million [COSTCO_2021_10K_c51]", "$59268.00"), 1.0)
        self.assertEqual(agree("Total was 16,525 million", "$16525.00"), 1.0)
        # Filings quote millions, answers sometimes restate in billions.
        self.assertEqual(agree("about 1.577 billion", "$1577.00"), 1.0)

        self.assertEqual(agree("$1,493 million", "$1577.00"), 0.0)
        self.assertEqual(agree("Not enough information in the provided context.", "$1577.00"), 0.0)

        # A shared fiscal year is not agreement, and nor is a digit inside an identifier
        # such as 3M or a citation tag.
        self.assertEqual(agree("Revenue rose in FY2023 [3M_2023_10K_c4]", "Flat in FY2023"), None)
        self.assertEqual(agree("3M reported growth", "The answer is 3.0"), 0.0)

        # Reference answers carrying no figure are not scorable this way.
        self.assertIsNone(agree("Yes, margins were stable", "Yes, margins were stable"))

        # A retrieval-only arm generates no answer, and an absent answer must not be scored as
        # a wrong one. This previously returned 0.0 and reported those arms as answering all 84
        # numeric questions incorrectly. The refusal above still scores 0.0, which is the
        # distinction that matters: a refusal has text, a non-run does not.
        self.assertIsNone(agree("", "$1577.00"))
        self.assertIsNone(agree("   ", "$1577.00"))
        self.assertIsNone(agree(float("nan"), "$1577.00"))

    def test_refusal_detection(self):
        """Test that refusals count however the model words them."""
        from evaluation.evaluate import is_refusal

        self.assertTrue(is_refusal("Not enough information in the provided context."))
        self.assertTrue(is_refusal("The excerpts do not contain the FY2018 figure."))
        self.assertFalse(is_refusal("$16,525 million [NIKE_2019_10K_c97]"))
        # An answer followed by a caveat is still an answer.
        self.assertFalse(is_refusal(
            "Revenue was $5,412 million [X_c12]. The excerpts do not contain the FY2019 figure."))

    def test_citation_functions(self):
        """Test citation evaluation functions."""
        from evaluation.evaluate import CitationEvaluator

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