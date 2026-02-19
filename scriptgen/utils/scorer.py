"""Source quality scoring module."""
import re
from datetime import datetime
from typing import Dict, List
from urllib.parse import urlparse


# Trusted high-authority domains get a boost
HIGH_AUTHORITY_DOMAINS = {
    ".edu", ".gov", ".org"
}

TRUSTED_SOURCES = {
    "nature.com", "science.org", "pubmed.ncbi.nlm.nih.gov",
    "arxiv.org", "scholar.google.com", "reuters.com",
    "bbc.com", "apnews.com", "economist.com", "wired.com",
    "technologyreview.com", "scientificamerican.com",
    "theguardian.com", "nytimes.com", "wsj.com",
    "forbes.com", "hbr.org", "mit.edu", "stanford.edu"
}

# Low-quality domains to penalize
LOW_QUALITY_DOMAINS = {
    "pinterest.com", "quora.com", "reddit.com",
    "facebook.com", "twitter.com", "instagram.com",
    "tiktok.com", "youtube.com", "amazon.com",
    "ebay.com", "yelp.com"
}


class SourceQualityScorer:
    """
    Scores web sources based on quality indicators:
    - Domain authority
    - Content richness
    - Relevance to topic
    - URL structure
    """

    def __init__(self, min_score: float = 0.3):
        """
        Initialize scorer.

        Args:
            min_score: Minimum quality score threshold (0.0 - 1.0).
                       Sources below this are filtered out.
        """
        self.min_score = min_score

    def score_source(self, source: dict, topic: str) -> Dict:
        """
        Score a single source on multiple quality dimensions.

        Args:
            source: Source dict with 'url' and 'raw_content' keys
            topic: Research topic for relevance scoring

        Returns:
            Source dict enriched with quality scores
        """
        url = source.get("url", "")
        content = source.get("raw_content", "") or source.get("content", "")

        # Calculate individual dimension scores
        domain_score = self._score_domain(url)
        content_score = self._score_content(content)
        relevance_score = self._score_relevance(content, topic)
        structure_score = self._score_url_structure(url)

        # Weighted final score
        # Relevance is most important, then content richness, then domain
        final_score = round(
            (relevance_score * 0.40) +
            (content_score  * 0.30) +
            (domain_score   * 0.20) +
            (structure_score * 0.10),
            4
        )

        return {
            **source,
            "quality_scores": {
                "final": final_score,
                "domain": domain_score,
                "content": content_score,
                "relevance": relevance_score,
                "structure": structure_score,
            }
        }

    def score_sources(self, sources: List[dict], topic: str) -> List[dict]:
        """
        Score a list of sources.

        Args:
            sources: List of source dicts
            topic: Research topic

        Returns:
            Sources with quality scores, sorted best-first
        """
        scored = [self.score_source(s, topic) for s in sources]
        return sorted(scored, key=lambda s: s["quality_scores"]["final"], reverse=True)

    def filter_sources(self, sources: List[dict], topic: str) -> List[dict]:
        """
        Score sources and filter out low-quality ones.

        Args:
            sources: List of source dicts
            topic: Research topic

        Returns:
            Filtered and sorted sources above min_score threshold
        """
        scored = self.score_sources(sources, topic)
        filtered = [s for s in scored if s["quality_scores"]["final"] >= self.min_score]

        # Always keep at least 1 source even if all fail threshold
        return filtered if filtered else scored[:1]

    # ------------------------------------------------------------------
    # Private scoring helpers
    # ------------------------------------------------------------------

    def _score_domain(self, url: str) -> float:
        """
        Score domain authority.

        Scoring:
        - Trusted source list  → 1.0
        - High authority TLD   → 0.8
        - Low quality domain   → 0.1
        - Unknown domain       → 0.5
        """
        if not url:
            return 0.0

        try:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
        except Exception:
            return 0.0

        # Remove www. prefix for matching
        domain = hostname.replace("www.", "")

        if domain in TRUSTED_SOURCES:
            return 1.0

        if domain in LOW_QUALITY_DOMAINS:
            return 0.1

        for tld in HIGH_AUTHORITY_DOMAINS:
            if domain.endswith(tld):
                return 0.8

        return 0.5

    def _score_content(self, content: str) -> float:
        """
        Score content richness based on length and structure.

        Scoring bands (word count):
        - < 100 words  → very short, score 0.1
        - 100–300      → short, 0.3
        - 300–600      → medium, 0.5
        - 600–1200     → good, 0.7
        - 1200–2000    → detailed, 0.9
        - > 2000       → comprehensive, 1.0
        """
        if not content:
            return 0.0

        word_count = len(content.split())

        if word_count < 100:
            return 0.1
        elif word_count < 300:
            return 0.3
        elif word_count < 600:
            return 0.5
        elif word_count < 1200:
            return 0.7
        elif word_count < 2000:
            return 0.9
        else:
            return 1.0

    def _score_relevance(self, content: str, topic: str) -> float:
        """
        Score topic relevance using keyword density.

        Extracts significant words from the topic (> 3 chars),
        counts their occurrences in content, and normalises
        against content length.
        """
        if not content or not topic:
            return 0.0

        content_lower = content.lower()
        topic_words = [w.lower() for w in topic.split() if len(w) > 3]

        if not topic_words:
            return 0.5

        total_mentions = sum(content_lower.count(w) for w in topic_words)
        word_count = max(len(content.split()), 1)

        # Keyword density as a fraction, capped at 1.0
        # A density of 2 % or above → perfect relevance score
        density = total_mentions / word_count
        return min(density / 0.02, 1.0)

    def _score_url_structure(self, url: str) -> float:
        """
        Score URL structure quality.

        Good signals: HTTPS, short clean path, no excessive parameters.
        Bad signals: HTTP, very long URLs, many query parameters.
        """
        if not url:
            return 0.0

        score = 0.5

        # HTTPS is a positive signal
        if url.startswith("https://"):
            score += 0.2

        # Penalise excessively long URLs (spam / tracking links)
        if len(url) > 200:
            score -= 0.2

        # Penalise many query parameters
        param_count = url.count("&")
        if param_count > 3:
            score -= 0.2
        elif param_count == 0:
            score += 0.1

        # Penalise deeply nested paths (> 5 slashes after domain)
        path_depth = url.count("/") - 2
        if path_depth > 5:
            score -= 0.1

        return max(0.0, min(score, 1.0))

    def get_score_summary(self, scored_sources: List[dict]) -> Dict:
        """
        Generate a summary of quality scores across all sources.

        Args:
            scored_sources: Sources that have already been scored

        Returns:
            Summary statistics dict
        """
        if not scored_sources:
            return {}

        finals = [s["quality_scores"]["final"] for s in scored_sources]

        return {
            "total_sources": len(scored_sources),
            "avg_score": round(sum(finals) / len(finals), 4),
            "max_score": round(max(finals), 4),
            "min_score": round(min(finals), 4),
            "high_quality_count": sum(1 for f in finals if f >= 0.7),
            "medium_quality_count": sum(1 for f in finals if 0.4 <= f < 0.7),
            "low_quality_count": sum(1 for f in finals if f < 0.4),
        }
