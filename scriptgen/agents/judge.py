"""Critique and evaluation agent."""
from typing import Dict, Any
from time import sleep
from .base import BaseAgent


class JudgeAgent(BaseAgent):
    """Agent responsible for critiquing reports and suggesting improvements."""
    
    def __init__(self):
        super().__init__(model="gemini-2.0-flash", temperature=0.7)
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critique the draft report and suggest improvements.
        
        Args:
            state: Current research state with draft_report
            
        Returns:
            Updated state with critique and incremented iteration
        """
        self.log("Critiquing report...")
        sleep(20)
        
        prompt = f"""
        You are a highly critical and insightful judge. Evaluate a research report.
        
        Topic: "{state['topic']}"
        
        Your Persona: First, deduce the primary audience for this topic.
        Embody that persona for your critique.
        
        Report to Critique:
        {state['draft_report']}
        
        Your Task:
        1. Provide a concise, constructive critique from your persona's point of view
        2. Identify specific weaknesses, gaps in logic, or unanswered questions
        3. Suggest 2-3 new specific research angles or questions for the next iteration
        
        Output Format:
        Critique:
        [Your detailed critique]
        
        Suggestions for Next Iteration:
        - [Suggestion/Question 1]
        """
        
        response = self.llm.invoke(prompt)
        
        history_entry = (
            f"--- Iteration {state['iteration']} ---\n"
            f"Plan: {state['plan']}\n"
            f"Critique:\n{response.content}"
        )
        
        self.log(f"Critique complete, moving to iteration {state['iteration'] + 1}")
        
        return {
            "critique": response.content,
            "research_history": [history_entry],
            "iteration": state['iteration'] + 1
        }
