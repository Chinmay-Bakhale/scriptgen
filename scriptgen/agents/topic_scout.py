"""Topic discovery agent."""
from typing import Optional
from langchain_tavily import TavilySearch
from .base import BaseAgent


class TopicScout(BaseAgent):
    """Agent to find trending and viral topics."""

    def __init__(self):
        super().__init__(model="sarvam-m", temperature=0.7)
        self.search_tool = TavilySearch(max_results=2)


    def execute(self, state: dict) -> dict:
        """Not used in current workflow, kept for compatibility."""
        return state
    
    def find_trending_topic(self) -> str:
        """Execute multi-step process to discover viral topic."""
        self.log("Hunting for trends...")
        
        # Step 1: Search across platforms
        self.log("Scanning Reddit and X/Twitter...")
        try:
            reddit_trends = self.search_tool.invoke({"query": "trending topics on reddit this week"})
            twitter_trends = self.search_tool.invoke({"query": "top viral discussions and hashtags on X this week"})
        except Exception as e:
            return f"Error: Could not fetch trends - {e}"
        
        # Step 2: Combine context
        combined_context = "Reddit Trends:\n" + "\n".join([str(t) for t in reddit_trends])
        combined_context += "\n\nTwitter/X Trends:\n" + "\n".join([str(t) for t in twitter_trends])
        
        # Step 3: Synthesize topic
        self.log("Analyzing trends to identify core topic...")
        synthesis_prompt = f"""
        You are a Viral Trend Analyst. Analyze this data to identify the most engaging,
        viral, and intellectually stimulating topic.
        
        Instructions:
        1. Identify recurring themes and keywords generating buzz
        2. Distill into a single compelling, research-worthy topic
        3. Format as a question or statement for deep investigation
        4. Output ONLY the final topic string
        
        Raw Data:
        ---
        {combined_context}
        ---
        
        Final Topic:
        """
        
        try:
            response = self.llm.invoke(synthesis_prompt)
            return response.content.strip()
        except Exception as e:
            return f"Error: Could not synthesize topic - {e}"
