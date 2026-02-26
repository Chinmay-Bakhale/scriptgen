from abc import ABC, abstractmethod
from typing import Dict, Any
import os
from langchain_openai import ChatOpenAI


class BaseAgent(ABC):
    def __init__(self, model: str = "sarvam-m", temperature: float = 0.7):
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise ValueError("SARVAM_API_KEY is not set")

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="https://api.sarvam.ai/v1",
            api_key=api_key,
            default_headers={"api-subscription-key": api_key},
            stream_usage=False,
            max_completion_tokens=8000
        )
        self.agent_name = self.__class__.__name__

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def log(self, message: str):
        print(f"[{self.agent_name}] {message}")
