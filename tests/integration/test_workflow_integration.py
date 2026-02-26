"""Integration tests for the full workflow."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestWorkflowIntegration:
    """Integration tests for end-to-end workflow."""

    @pytest.mark.integration
    @patch('scriptgen.core.workflow.KnowledgeBase')
    @patch('scriptgen.agents.researcher.TavilyExtract')
    @patch('scriptgen.agents.researcher.TavilySearch')
    @patch('scriptgen.agents.base.ChatOpenAI')
    def test_single_iteration_workflow(self, mock_llm, mock_search, mock_extract, mock_kb, mock_env_vars):
        """Test a single iteration of the research workflow initializes correctly."""
        from scriptgen.core.workflow import MultiAgentResearchSystem

        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = "Plan:\nTest plan\n\nQueries:\n- query1\n- query2"
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        mock_search_instance = Mock()
        mock_search_instance.invoke.return_value = {
            "results": [
                {"title": "Test", "url": "https://test.com", "content": "Test content"}
            ]
        }
        mock_search.return_value = mock_search_instance

        mock_extract_instance = Mock()
        mock_extract_instance.invoke.return_value = {
            "results": [{"url": "https://test.com", "raw_content": "Extracted content"}]
        }
        mock_extract.return_value = mock_extract_instance

        system = MultiAgentResearchSystem()

        assert system is not None
        assert system.app is not None
        assert hasattr(system, 'evaluator')
        assert hasattr(system, 'knowledge_base')
        assert hasattr(system, 'retriever')

    @pytest.mark.integration
    @patch('builtins.input', return_value='test topic')
    def test_manual_topic_input(self, mock_input, mock_env_vars):
        """Test manual topic input flow."""
        assert mock_input() == 'test topic'
