"""Base agent class for all workflow agents."""
from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI


class BaseAgent(ABC):
    """Abstract base class for all agents in the workflow."""
    
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.7):
        """
        Initialize base agent.
        
        Args:
            model: LLM model name
            temperature: Sampling temperature
        """
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.agent_name = self.__class__.__name__
    
    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent logic.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state dictionary
        """
        pass
    
    def log(self, message: str):
        """Simple logging helper."""
        print(f"[{self.agent_name}] {message}")
