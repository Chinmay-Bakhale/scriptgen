"""Pytest configuration and shared fixtures."""
import pytest
import os
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("SARVAM_API_KEY", "test-sarvam-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")


@pytest.fixture
def sample_research_state():
    """Sample research state for testing."""
    return {
        "topic": "AI safety in 2026",
        "iteration": 1,
        "plan": "",
        "search_queries": [],
        "raw_search_results": [],
        "extracted_pages": [],
        "draft_report": "",
        "critique": "",
        "research_history": [],
        "final_report": "",
        "quality_summary": {},
        "search_latency_seconds": 0.0,
        "prior_context": ""
    }

@pytest.fixture
def mock_llm_response():
    """Mock LLM response object."""
    mock = MagicMock()
    mock.content = "Plan:\nTest research plan\n\nQueries:\n- query 1\n- query 2"
    return mock


@pytest.fixture
def mock_search_results():
    """Mock search results from Tavily."""
    return {
        "results": [
            {
                "title": "Test Article 1",
                "url": "https://example.com/article1",
                "content": "Test content 1",
                "score": 0.95
            },
            {
                "title": "Test Article 2",
                "url": "https://example.com/article2",
                "content": "Test content 2",
                "score": 0.85
            }
        ]
    }
