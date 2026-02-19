"""Research planning agent."""
import re
from typing import Dict, Any, List
from .base import BaseAgent


class PlannerAgent(BaseAgent):
    """Agent responsible for creating research strategies and generating queries."""
    
    def __init__(self):
        super().__init__(model="gemini-2.5-flash", temperature=0.6)
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create research plan and generate search queries.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with plan and queries
        """
        iteration = state['iteration']
        self.log(f"Planning iteration {iteration}...")
        
        if iteration == 1:
            history_prompt = "This is the first iteration. Start a fresh research plan."
        else:
            history_prompt = (
                "Review the research history and judge's critique to refine the plan. "
                "Do NOT repeat previous searches. Focus on addressing critique and filling gaps."
            )
        
        prompt = f"""
        You are a master research planner. Devise a detailed research strategy.
        
        Topic: "{state['topic']}"
        Research History:
        {state['research_history']}
        
        Previous Critique:
        {state['critique']}
        
        Your Task ({history_prompt}):
        1. Create a concise, one-paragraph research plan
        2. Generate 2 precise and diverse search queries
        
        Output Format:
        Plan:
        [Your one-paragraph research plan]
        
        Queries:
        - [Query 1]
        - [Query 2]
        """
        
        response = self.llm.invoke(prompt)
        plan_match = re.search(r"Plan:\s*(.*?)\s*Queries:", response.content, re.DOTALL)
        queries_match = re.search(r"Queries:\s*(.*)", response.content, re.DOTALL)
        
        plan = plan_match.group(1).strip() if plan_match else "No plan generated."
        queries = [q.strip().replace('-', '').strip() 
                  for q in queries_match.group(1).strip().split('\n')] if queries_match else []
        
        self.log(f"Generated plan with {len(queries)} queries")
        
        return {
            "plan": plan,
            "search_queries": queries,
            "raw_search_results": []
        }
