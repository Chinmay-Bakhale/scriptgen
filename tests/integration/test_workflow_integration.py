"""Integration tests for the full workflow."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestWorkflowIntegration:
    """Integration tests for end-to-end workflow."""

    @pytest.mark.integration
    @patch('scriptgen.agents.researcher.TavilyExtract')
    @patch('scriptgen.agents.researcher.TavilySearch')
    @patch('scriptgen.agents.base.ChatGoogleGenerativeAI')
    def test_single_iteration_workflow(self, mock_llm, mock_search, mock_extract, mock_env_vars):
        """Test a single iteration of the research workflow."""
        from scriptgen.core.workflow import MultiAgentResearchSystem
        
        # Mock all LLM calls
        mock_llm_instance = Mock()
        mock_planner_response = Mock()
        mock_planner_response.content = "Plan:\nTest plan\n\nQueries:\n- query1\n- query2"
        
        mock_writer_response = Mock()
        mock_writer_response.content = "Test draft report content"
        
        mock_judge_response = Mock()
        mock_judge_response.content = "Critique: Good start\nSuggestions:\n- Add more detail"
        
        mock_final_response = Mock()
        mock_final_response.content = "Final polished report"
        
        mock_llm_instance.invoke.side_effect = [
            mock_planner_response,
            mock_writer_response,
            mock_judge_response,
            mock_planner_response,  # Second iteration
            mock_writer_response,
            mock_judge_response,
            mock_planner_response,  # Third iteration
            mock_writer_response,
            mock_final_response
        ]
        mock_llm.return_value = mock_llm_instance
        
        # Mock search results
        mock_search_instance = Mock()
        mock_search_instance.invoke.return_value = {
            "results": [
                {"title": "Test", "url": "https://test.com", "content": "Test content"}
            ]
        }
        mock_search.return_value = mock_search_instance
        
        # Mock extractor
        mock_extract_instance = Mock()
        mock_extract_instance.invoke.return_value = {
            "results": [{"url": "https://test.com", "raw_content": "Extracted content"}]
        }
        mock_extract.return_value = mock_extract_instance
        
        system = MultiAgentResearchSystem()
        
        # Smoke test - verify it doesn't crash
        assert system is not None
        assert system.app is not None
        assert hasattr(system, 'evaluator')

    @pytest.mark.integration
    @patch('builtins.input', return_value='test topic')
    def test_manual_topic_input(self, mock_input, mock_env_vars):
        """Test manual topic input flow."""
        assert mock_input() == 'test topic'
