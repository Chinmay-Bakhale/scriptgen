"""Unit tests for ReportEvaluator."""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from metrics.evaluator import ReportEvaluator


class TestReportEvaluator:
    """Test cases for ReportEvaluator class."""
    
    def test_initialization(self):
        """Test evaluator initializes correctly."""
        evaluator = ReportEvaluator()
        assert evaluator is not None
    
    def test_count_words(self):
        """Test word counting."""
        evaluator = ReportEvaluator()
        text = "This is a test sentence with seven words."
        assert evaluator._count_words(text) == 8
    
    def test_count_sentences(self):
        """Test sentence counting."""
        evaluator = ReportEvaluator()
        text = "First sentence. Second sentence! Third sentence?"
        assert evaluator._count_sentences(text) == 3
    
    def test_count_paragraphs(self):
        """Test paragraph counting."""
        evaluator = ReportEvaluator()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        assert evaluator._count_paragraphs(text) == 3
    
    def test_avg_sentence_length(self):
        """Test average sentence length calculation."""
        evaluator = ReportEvaluator()
        text = "Short. This is longer sentence."
        avg = evaluator._avg_sentence_length(text)
        assert avg > 0
        assert isinstance(avg, float)
    
    def test_count_citations(self):
        """Test citation counting."""
        evaluator = ReportEvaluator()
        text = "Some fact [1]. Another fact [2]. More info [3]."
        assert evaluator._count_citations(text) == 3
    
    def test_count_unique_domains(self):
        """Test unique domain counting."""
        evaluator = ReportEvaluator()
        sources = [
            {"url": "https://example.com/page1"},
            {"url": "https://example.com/page2"},
            {"url": "https://another.com/page"},
        ]
        assert evaluator._count_unique_domains(sources) == 2
    
    def test_has_sections(self):
        """Test section detection."""
        evaluator = ReportEvaluator()
        text_with_sections = "# Title\n\n## Section 1\n\nContent"
        text_without_sections = "Just plain text without sections"
        
        assert evaluator._has_sections(text_with_sections) is True
        assert evaluator._has_sections(text_without_sections) is False
    
    def test_count_sections(self):
        """Test section counting."""
        evaluator = ReportEvaluator()
        text = "# Title\n\n## Section 1\n\n## Section 2\n\n### Subsection"
        assert evaluator._count_sections(text) == 2
    
    def test_evaluate_report_basic(self):
        """Test full report evaluation."""
        evaluator = ReportEvaluator()
        
        report = """## Introduction
        
This is a test report about AI safety. It has multiple sentences.
The report discusses important topics.

## Conclusion

AI safety is crucial for the future."""
        
        sources = [
            {"url": "https://example.com/ai"},
            {"url": "https://test.com/safety"}
        ]
        
        metrics = evaluator.evaluate_report(
            report=report,
            topic="AI safety",
            sources=sources
        )
        
        assert "word_count" in metrics
        assert "sentence_count" in metrics
        assert "source_count" in metrics
        assert metrics["source_count"] == 2
        assert metrics["unique_domains"] == 2
        assert metrics["has_sections"] is True
        assert metrics["topic_mentions"] > 0
    
    def test_evaluate_report_with_execution_time(self):
        """Test evaluation with execution time."""
        evaluator = ReportEvaluator()
        
        metrics = evaluator.evaluate_report(
            report="Simple report.",
            topic="Test",
            sources=[{"url": "https://test.com"}],
            execution_time=45.67
        )
        
        assert "execution_time_seconds" in metrics
        assert metrics["execution_time_seconds"] == 45.67
    
    def test_format_metrics_report(self):
        """Test metrics formatting."""
        evaluator = ReportEvaluator()
        
        metrics = {
            "word_count": 500,
            "sentence_count": 25,
            "source_count": 5,
            "avg_sentence_length": 20.0,
            "has_sections": True
        }
        
        formatted = evaluator.format_metrics_report(metrics)
        
        assert "Report Evaluation Metrics" in formatted
        assert "500" in formatted
        assert "25" in formatted
