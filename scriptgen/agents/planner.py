"""Research planning agent."""
import re
from typing import Dict, Any, List
from .base import BaseAgent


class PlannerAgent(BaseAgent):
    """Agent responsible for creating research strategies and generating queries."""
    
    def __init__(self):
        super().__init__(model="sarvam-m", temperature=0.7)

    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create research plan and generate search queries."""
        iteration = state['iteration']
        self.log(f"Planning iteration {iteration}...")

        prior_context = state.get("prior_context", "")

        if iteration == 1:
            history_prompt = "This is the first iteration. Start a fresh research plan."
        else:
            history_prompt = (
                "Review the research history and judge's critique to refine the plan. "
                "Do NOT repeat previous searches. Focus on addressing critique and filling gaps."
            )

        if prior_context:
            prior_context_block = f"Prior Knowledge (already researched — do NOT re-search this):\n{prior_context}"
        else:
            prior_context_block = ""

        prompt = f"""
        You are a master research planner. Devise a detailed research strategy.

        Topic: "{state['topic']}"

        {prior_context_block}

        Research History:
        {state['research_history']}

        Previous Critique:
        {state['critique']}

        Your Task ({history_prompt}):
        1. Create a concise, one-paragraph research plan
        2. Generate 2 precise and diverse search queries that explore
        angles NOT already covered in the prior knowledge above

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
        newline = "\n"
        queries = [
            q.strip().replace('-', '').strip()
            for q in queries_match.group(1).strip().split(newline)
        ] if queries_match else []

        self.log(f"Generated plan with {len(queries)} queries")

        return {
            "plan": plan,
            "search_queries": queries,
            "raw_search_results": []
        }
