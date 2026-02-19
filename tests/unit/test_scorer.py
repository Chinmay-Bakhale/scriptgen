"""Unit tests for SourceQualityScorer."""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scriptgen.utils.scorer import SourceQualityScorer


class TestSourceQualityScorer:
    """Test cases for SourceQualityScorer."""

    def test_initialization(self):
        """Test scorer initializes with default min_score."""
        scorer = SourceQualityScorer()
        assert scorer.min_score == 0.3

    def test_custom_min_score(self):
        """Test custom min_score."""
        scorer = SourceQualityScorer(min_score=0.5)
        assert scorer.min_score == 0.5

    # --- Domain scoring ---

    def test_trusted_domain_scores_high(self):
        """Trusted domains should score 1.0."""
        scorer = SourceQualityScorer()
        assert scorer._score_domain("https://nature.com/article") == 1.0
        assert scorer._score_domain("https://arxiv.org/paper") == 1.0

    def test_low_quality_domain_scores_low(self):
        """Social media / low quality domains should score 0.1."""
        scorer = SourceQualityScorer()
        assert scorer._score_domain("https://pinterest.com/post") == 0.1
        assert scorer._score_domain("https://reddit.com/thread") == 0.1

    def test_edu_gov_domain_scores_high(self):
        """edu/gov TLDs should score 0.8."""
        scorer = SourceQualityScorer()
        assert scorer._score_domain("https://someuniversity.edu/paper") == 0.8
        assert scorer._score_domain("https://agency.gov/report") == 0.8

    def test_unknown_domain_scores_medium(self):
        """Unknown domain should score 0.5."""
        scorer = SourceQualityScorer()
        assert scorer._score_domain("https://somerandomblog.com/post") == 0.5

    # --- Content scoring ---

    def test_empty_content_scores_zero(self):
        """Empty content should score 0."""
        scorer = SourceQualityScorer()
        assert scorer._score_content("") == 0.0

    def test_short_content_scores_low(self):
        """Very short content should score low."""
        scorer = SourceQualityScorer()
        assert scorer._score_content("short") == 0.1

    def test_long_content_scores_high(self):
        """Long content should score high."""
        scorer = SourceQualityScorer()
        long_content = " ".join(["word"] * 2500)
        assert scorer._score_content(long_content) == 1.0

    # --- Relevance scoring ---

    def test_high_relevance_scores_high(self):
        """Content with many topic mentions should score high."""
        scorer = SourceQualityScorer()
        content = " ".join(["machine learning artificial intelligence"] * 100)
        score = scorer._score_relevance(content, "machine learning")
        assert score == 1.0

    def test_no_relevance_scores_low(self):
        """Content with no topic mentions should score low."""
        scorer = SourceQualityScorer()
        content = "cooking recipes pasta tomato sauce"
        score = scorer._score_relevance(content, "quantum computing physics")
        assert score == 0.0

    # --- URL structure scoring ---

    def test_https_scores_higher_than_http(self):
        """HTTPS URLs should score higher than HTTP."""
        scorer = SourceQualityScorer()
        https = scorer._score_url_structure("https://example.com/article")
        http = scorer._score_url_structure("http://example.com/article")
        assert https > http

    def test_long_url_penalized(self):
        """Very long URLs should be penalized."""
        scorer = SourceQualityScorer()
        short_url = "https://example.com/article"
        long_url = "https://example.com/" + "a" * 200
        assert scorer._score_url_structure(short_url) > scorer._score_url_structure(long_url)

    # --- Full pipeline ---

    def test_score_source_returns_all_dimensions(self):
        """Scored source should contain all quality dimensions."""
        scorer = SourceQualityScorer()
        source = {
            "url": "https://nature.com/article",
            "raw_content": " ".join(["AI safety research"] * 200)
        }
        scored = scorer.score_source(source, "AI safety")

        assert "quality_scores" in scored
        assert "final" in scored["quality_scores"]
        assert "domain" in scored["quality_scores"]
        assert "content" in scored["quality_scores"]
        assert "relevance" in scored["quality_scores"]
        assert "structure" in scored["quality_scores"]

    def test_score_source_final_between_0_and_1(self):
        """Final score should always be between 0 and 1."""
        scorer = SourceQualityScorer()
        source = {"url": "https://example.com", "raw_content": "some content here"}
        scored = scorer.score_source(source, "example topic")
        assert 0.0 <= scored["quality_scores"]["final"] <= 1.0

    def test_score_sources_sorted_best_first(self):
        """Scored sources should be sorted best-first."""
        scorer = SourceQualityScorer()
        sources = [
            {"url": "http://spam.com", "raw_content": "short"},
            {"url": "https://nature.com/article", "raw_content": " ".join(["AI"] * 500)},
            {"url": "https://example.edu/paper", "raw_content": " ".join(["research"] * 300)},
        ]
        scored = scorer.score_sources(sources, "AI research")
        scores = [s["quality_scores"]["final"] for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_filter_sources_removes_low_quality(self):
        """Filter should remove sources below min_score."""
        scorer = SourceQualityScorer(min_score=0.5)
        sources = [
            {"url": "http://spam.com", "raw_content": "x"},
            {"url": "https://nature.com/research", "raw_content": " ".join(["AI safety research"] * 300)},
        ]
        filtered = scorer.filter_sources(sources, "AI safety")
        # All kept sources must meet min_score
        for s in filtered:
            assert s["quality_scores"]["final"] >= 0.5

    def test_filter_always_keeps_at_least_one(self):
        """Filter should keep at least one source even if all fail."""
        scorer = SourceQualityScorer(min_score=0.99)
        sources = [{"url": "http://low.com", "raw_content": "tiny"}]
        filtered = scorer.filter_sources(sources, "complex scientific topic")
        assert len(filtered) >= 1

    def test_get_score_summary_structure(self):
        """Summary should contain all expected keys."""
        scorer = SourceQualityScorer()
        sources = [
            {"url": "https://nature.com", "raw_content": " ".join(["AI"] * 500)},
            {"url": "http://spam.com", "raw_content": "short"},
        ]
        scored = scorer.score_sources(sources, "AI")
        summary = scorer.get_score_summary(scored)

        assert "total_sources" in summary
        assert "avg_score" in summary
        assert "max_score" in summary
        assert "min_score" in summary
        assert "high_quality_count" in summary
        assert "medium_quality_count" in summary
        assert "low_quality_count" in summary
