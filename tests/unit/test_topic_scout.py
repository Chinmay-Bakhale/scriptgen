"""Unit tests for TopicScout agent."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestTopicScout:
    """Test cases for TopicScout class."""

    @patch('scriptgen.agents.topic_scout.TavilySearch')
    @patch('scriptgen.agents.base.ChatOpenAI')
    def test_topic_scout_initialization(self, mock_llm, mock_tavily, mock_env_vars):
        """Test TopicScout initializes correctly."""
        from scriptgen.agents.topic_scout import TopicScout
        
        scout = TopicScout()
        
        assert scout is not None
        assert hasattr(scout, 'llm')
        assert hasattr(scout, 'search_tool')

    @patch('scriptgen.agents.topic_scout.TavilySearch')
    @patch('scriptgen.agents.base.ChatOpenAI')
    def test_find_trending_topic_success(self, mock_llm, mock_tavily, mock_env_vars):
        """Test successful topic discovery."""
        from scriptgen.agents.topic_scout import TopicScout
        
        # Mock search results
        mock_search = Mock()
        mock_search.invoke.return_value = [
            {"title": "Trending AI", "content": "AI is trending"},
            {"title": "Tech News", "content": "Latest tech"}
        ]
        mock_tavily.return_value = mock_search
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = "AI Safety and Ethics in Modern Systems"
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance
        
        scout = TopicScout()
        topic = scout.find_trending_topic()
        
        assert topic is not None
        assert isinstance(topic, str)
        assert len(topic) > 0
        assert "Error" not in topic

    @patch('scriptgen.agents.topic_scout.TavilySearch')
    @patch('scriptgen.agents.base.ChatOpenAI')
    def test_find_trending_topic_search_failure(self, mock_llm, mock_tavily, mock_env_vars):
        """Test topic discovery handles search failures."""
        from scriptgen.agents.topic_scout import TopicScout
        
        # Mock search failure
        mock_search = Mock()
        mock_search.invoke.side_effect = Exception("API Error")
        mock_tavily.return_value = mock_search
        
        scout = TopicScout()
        topic = scout.find_trending_topic()
        
        assert "Error" in topic
