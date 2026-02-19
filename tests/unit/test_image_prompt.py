"""Unit tests for ImagePromptGenerator."""
import pytest
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestImagePromptGenerator:
    """Test cases for ImagePromptGenerator class."""

    @patch('scriptgen.utils.image_prompt.ChatGoogleGenerativeAI')
    def test_initialization(self, mock_llm, mock_env_vars):
        """Test ImagePromptGenerator initializes correctly."""
        from scriptgen.utils.image_prompt import ImagePromptGenerator
        
        generator = ImagePromptGenerator()
        
        assert generator is not None
        assert hasattr(generator, 'llm')

    @patch('scriptgen.utils.image_prompt.ChatGoogleGenerativeAI')
    def test_create_prompts_for_source(self, mock_llm, mock_env_vars):
        """Test prompt generation for a source."""
        from scriptgen.utils.image_prompt import ImagePromptGenerator
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = "Prompt 1: Test prompt\nPrompt 2: Another prompt"
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance
        
        generator = ImagePromptGenerator()
        result = generator._create_prompts_for_source("Test analysis content")
        
        assert result is not None
        assert isinstance(result, str)
        assert "Prompt" in result

    @patch('scriptgen.utils.image_prompt.ChatGoogleGenerativeAI')
    def test_generate_and_save_prompts(self, mock_llm, mock_env_vars, tmp_path):
        """Test full prompt generation and saving workflow."""
        from scriptgen.utils.image_prompt import ImagePromptGenerator
        
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = "Prompt 1: Visual concept\nPrompt 2: Another concept"
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance
        
        generator = ImagePromptGenerator()
        output_file = tmp_path / "test_prompts.md"
        
        generator.generate_and_save_prompts("Test report content", str(output_file))
        
        assert output_file.exists()
        content = output_file.read_text()
        assert "AI-Generated Image Prompts" in content
        assert "Prompt" in content
