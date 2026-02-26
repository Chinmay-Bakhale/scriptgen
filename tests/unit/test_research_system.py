"""Unit tests for MultiAgentResearchSystem."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestMultiAgentResearchSystem:
    """Test cases for MultiAgentResearchSystem class."""

    @patch('scriptgen.core.workflow.KnowledgeBase')
    @patch('scriptgen.agents.researcher.TavilySearch')
    @patch('scriptgen.agents.base.ChatOpenAI')
    def test_system_initialization(self, mock_llm, mock_tavily, mock_kb, mock_env_vars):
        """Test system initializes with all components."""
        from scriptgen.core.workflow import MultiAgentResearchSystem

        system = MultiAgentResearchSystem()

        assert system is not None
        assert hasattr(system, 'planner')
        assert hasattr(system, 'searcher')
        assert hasattr(system, 'extractor')
        assert hasattr(system, 'filter')
        assert hasattr(system, 'retriever')
        assert hasattr(system, 'knowledge_store')
        assert hasattr(system, 'writer')
        assert hasattr(system, 'judge')
        assert hasattr(system, 'final_writer')
        assert hasattr(system, 'evaluator')
        assert hasattr(system, 'knowledge_base')
        assert hasattr(system, 'workflow')
        assert hasattr(system, 'app')

    @patch('scriptgen.core.workflow.KnowledgeBase')
    @patch('scriptgen.agents.researcher.TavilySearch')
    @patch('scriptgen.agents.base.ChatOpenAI')
    def test_planner_node_first_iteration(self, mock_llm, mock_tavily, mock_kb, mock_env_vars, sample_research_state):
        """Test planner node on first iteration."""
        from scriptgen.core.workflow import MultiAgentResearchSystem

        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = "Plan:\nResearch AI safety measures\n\nQueries:\n- AI safety 2026\n- Machine learning ethics"
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        system = MultiAgentResearchSystem()

        # Add prior_context to state since planner now expects it
        state = {**sample_research_state, "prior_context": ""}
        result = system._planner_node(state)

        assert 'plan' in result
        assert 'search_queries' in result
        assert isinstance(result['search_queries'], list)
        assert len(result['search_queries']) > 0

    @patch('scriptgen.core.workflow.KnowledgeBase')
    @patch('scriptgen.agents.researcher.TavilySearch')
    @patch('scriptgen.agents.base.ChatOpenAI')
    def test_searcher_node_executes_queries(self, mock_llm, mock_tavily, mock_kb, mock_env_vars, sample_research_state, mock_search_results):
        """Test searcher node executes search queries."""
        from scriptgen.core.workflow import MultiAgentResearchSystem

        mock_search = Mock()
        mock_search.invoke.return_value = mock_search_results
        mock_tavily.return_value = mock_search

        system = MultiAgentResearchSystem()
        state = {**sample_research_state, "search_queries": ["AI safety", "Machine learning"]}

        result = system._searcher_node(state)

        assert 'raw_search_results' in result
        assert isinstance(result['raw_search_results'], list)

    @patch('scriptgen.core.workflow.KnowledgeBase')
    @patch('scriptgen.agents.researcher.TavilySearch')
    @patch('scriptgen.agents.base.ChatOpenAI')
    def test_should_continue_logic(self, mock_llm, mock_tavily, mock_kb, mock_env_vars, sample_research_state):
        """Test workflow continuation logic."""
        from scriptgen.core.workflow import MultiAgentResearchSystem

        system = MultiAgentResearchSystem()

        state1 = {**sample_research_state, "iteration": 1}
        assert system._should_continue(state1) == "continue_to_judge"

        state2 = {**sample_research_state, "iteration": 3}
        assert system._should_continue(state2) == "end_workflow"
