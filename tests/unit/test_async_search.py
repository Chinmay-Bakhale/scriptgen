"""Unit tests for async SearchAgent."""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestSearchAgent:
    """Test cases for async SearchAgent."""

    @patch('scriptgen.agents.researcher.TavilySearch')
    @patch('scriptgen.agents.base.ChatGoogleGenerativeAI')
    def test_initialization(self, mock_llm, mock_tavily, mock_env_vars):
        """Test SearchAgent initializes correctly."""
        from scriptgen.agents.researcher import SearchAgent
        agent = SearchAgent()
        assert agent is not None
        assert hasattr(agent, 'search_tool')

    @patch('scriptgen.agents.researcher.TavilySearch')
    @patch('scriptgen.agents.base.ChatGoogleGenerativeAI')
    def test_execute_returns_results(self, mock_llm, mock_tavily, mock_env_vars):
        """Test execute returns search results."""
        from scriptgen.agents.researcher import SearchAgent

        mock_search = Mock()
        mock_search.invoke.return_value = {
            "results": [
                {"url": "https://example.com", "content": "test content"}
            ]
        }
        mock_tavily.return_value = mock_search

        agent = SearchAgent()
        state = {
            "search_queries": ["AI safety", "machine learning"],
            "topic": "AI safety"
        }

        result = agent.execute(state)

        assert "raw_search_results" in result
        assert "search_latency_seconds" in result
        assert isinstance(result["raw_search_results"], list)
        assert isinstance(result["search_latency_seconds"], float)

    @patch('scriptgen.agents.researcher.TavilySearch')
    @patch('scriptgen.agents.base.ChatGoogleGenerativeAI')
    def test_execute_empty_queries(self, mock_llm, mock_tavily, mock_env_vars):
        """Test execute handles empty query list."""
        from scriptgen.agents.researcher import SearchAgent

        agent = SearchAgent()
        state = {"search_queries": [], "topic": "test"}
        result = agent.execute(state)

        assert result["raw_search_results"] == []
        assert result["search_latency_seconds"] == 0.0

    @patch('scriptgen.agents.researcher.TavilySearch')
    @patch('scriptgen.agents.base.ChatGoogleGenerativeAI')
    def test_execute_filters_empty_strings(self, mock_llm, mock_tavily, mock_env_vars):
        """Test execute filters out empty/whitespace queries."""
        from scriptgen.agents.researcher import SearchAgent

        mock_search = Mock()
        mock_search.invoke.return_value = {"results": []}
        mock_tavily.return_value = mock_search

        agent = SearchAgent()
        state = {"search_queries": ["", "  ", "valid query"], "topic": "test"}
        result = agent.execute(state)

        # Only 'valid query' should have been searched
        assert mock_search.invoke.call_count == 1

    @patch('scriptgen.agents.researcher.TavilySearch')
    @patch('scriptgen.agents.base.ChatGoogleGenerativeAI')
    def test_execute_handles_search_failure(self, mock_llm, mock_tavily, mock_env_vars):
        """Test execute handles individual query failures gracefully."""
        from scriptgen.agents.researcher import SearchAgent

        mock_search = Mock()
        mock_search.invoke.side_effect = Exception("API Error")
        mock_tavily.return_value = mock_search

        agent = SearchAgent()
        state = {"search_queries": ["failing query"], "topic": "test"}

        # Should not raise, should return empty results
        result = agent.execute(state)
        assert "raw_search_results" in result
        assert isinstance(result["raw_search_results"], list)

    @patch('scriptgen.agents.researcher.TavilySearch')
    @patch('scriptgen.agents.base.ChatGoogleGenerativeAI')
    def test_concurrent_execution_faster_than_sequential(self, mock_llm, mock_tavily, mock_env_vars):
        """Test that concurrent execution completes queries together."""
        import time
        from scriptgen.agents.researcher import SearchAgent

        call_times = []

        def slow_search(payload):
            call_times.append(time.time())
            import time as t
            t.sleep(0.1)
            return {"results": [{"url": "https://example.com", "content": "result"}]}

        mock_search = Mock()
        mock_search.invoke.side_effect = slow_search
        mock_tavily.return_value = mock_search

        agent = SearchAgent()
        state = {
            "search_queries": ["query1", "query2", "query3"],
            "topic": "test"
        }

        start = time.time()
        result = agent.execute(state)
        elapsed = time.time() - start

        # 3 queries × 0.1s each = 0.3s sequential
        # Concurrent should finish much faster than 0.3s
        assert elapsed < 0.3, f"Expected concurrent execution under 0.3s, got {elapsed:.2f}s"
        assert len(result["raw_search_results"]) == 3

    @pytest.mark.asyncio
    @patch('scriptgen.agents.researcher.TavilySearch')
    @patch('scriptgen.agents.base.ChatGoogleGenerativeAI')
    async def test_search_all_collects_all_results(self, mock_llm, mock_tavily, mock_env_vars):
        """Test _search_all collects results from all queries."""
        from scriptgen.agents.researcher import SearchAgent

        mock_search = Mock()
        mock_search.invoke.return_value = {
            "results": [{"url": "https://example.com", "content": "test"}]
        }
        mock_tavily.return_value = mock_search

        agent = SearchAgent()
        results = await agent._search_all(["query1", "query2"])

        # 2 queries × 1 result each = 2 results
        assert len(results) == 2
