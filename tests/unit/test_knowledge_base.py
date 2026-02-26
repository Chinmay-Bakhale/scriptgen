"""Unit tests for KnowledgeBase."""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestKnowledgeBase:
    """Tests for KnowledgeBase using a temp directory."""

    @pytest.fixture
    def kb(self, tmp_path):
        """Create a fresh KnowledgeBase in a temp directory."""
        from scriptgen.utils.knowledge_base import KnowledgeBase
        return KnowledgeBase(persist_directory=str(tmp_path / "test_chroma"))

    def test_initialization(self, kb):
        """Test KB initializes correctly."""
        assert kb is not None
        assert kb.index is not None          # FAISS index, not collection
        assert kb.encoder is not None        # sentence-transformer encoder

    def test_initial_count_is_zero(self, kb):
        """Test KB starts empty."""
        assert len(kb.metadata) == 0         # metadata list, not collection.count()

    def test_add_documents_stores_pages(self, kb):
        """Test adding documents increases count."""
        pages = [
            {"url": "https://example.com/1", "raw_content": "AI safety is important " * 50},
            {"url": "https://example.com/2", "raw_content": "Machine learning research " * 50},
        ]
        kb.add_documents(pages, topic="AI safety", iteration=1)
        assert len(kb.metadata) == 2         # check metadata list length

    def test_add_documents_skips_duplicates(self, kb):
        """Test same URL not stored twice."""
        pages = [{"url": "https://example.com/1", "raw_content": "content " * 50}]
        kb.add_documents(pages, topic="test", iteration=1)
        kb.add_documents(pages, topic="test", iteration=1)
        assert len(kb.metadata) == 1         # still only 1

    def test_add_documents_skips_empty_content(self, kb):
        """Test empty content pages are skipped."""
        pages = [{"url": "https://example.com", "raw_content": ""}]
        kb.add_documents(pages, topic="test", iteration=1)
        assert len(kb.metadata) == 0         # nothing stored

    def test_retrieve_returns_results(self, kb):
        """Test retrieval returns relevant documents."""
        pages = [
            {"url": "https://example.com/ai", "raw_content": "artificial intelligence research " * 100},
        ]
        kb.add_documents(pages, topic="AI", iteration=1)

        results = kb.retrieve("artificial intelligence", n_results=1)

        assert len(results) > 0
        assert "content" in results[0]
        assert "url" in results[0]
        assert "relevance_score" in results[0]

    def test_retrieve_empty_kb_returns_empty(self, kb):
        """Test retrieval on empty KB returns empty list."""
        results = kb.retrieve("any query")
        assert results == []

    def test_retrieve_relevance_score_between_0_and_1(self, kb):
        """Test relevance scores are normalized."""
        pages = [
            {"url": "https://example.com", "raw_content": "climate change global warming " * 100},
        ]
        kb.add_documents(pages, topic="climate", iteration=1)
        results = kb.retrieve("climate change")

        for r in results:
            assert 0.0 <= r["relevance_score"] <= 1.0

    def test_get_stats_structure(self, kb):
        """Test stats returns expected keys."""
        stats = kb.get_stats()
        assert "total_documents" in stats
        assert "persist_directory" in stats
        assert "embedding_model" in stats

    def test_format_context_with_results(self, kb):
        """Test context formatting with retrieved docs."""
        docs = [
            {"content": "test content", "url": "https://example.com", "relevance_score": 0.9}
        ]
        context = kb.format_context(docs)
        assert "Prior Research Context" in context
        assert "example.com" in context

    def test_format_context_empty_returns_message(self, kb):
        """Test empty retrieval returns no-context message."""
        context = kb.format_context([])
        assert "No prior knowledge" in context
