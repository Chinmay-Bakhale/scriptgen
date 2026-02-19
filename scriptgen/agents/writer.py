"""Writing agents for report generation."""
from typing import Dict, Any
from time import sleep
from .base import BaseAgent


class WriterAgent(BaseAgent):
    """Agent responsible for drafting research reports."""
    
    def __init__(self):
        super().__init__(model="gemini-2.5-flash-preview-04-17", temperature=0.5)
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Draft report from extracted content.
        
        Args:
            state: Current research state with extracted_pages
            
        Returns:
            Updated state with draft_report
        """
        self.log("Drafting report...")
        sleep(10)
        
        if not state["extracted_pages"]:
            return {"draft_report": "No extracted pages available for this iteration."}
        
        source_material = "\n\n---\n\n".join(
            page.get("raw_content", "") for page in state["extracted_pages"]
        )
        
        prompt = f"""
        You are an expert writer. Synthesize the following material into a detailed,
        well-structured report.
        
        Topic: "{state['topic']}"
        
        Instructions:
        - Analyze all provided source material
        - Write an in-depth report covering key findings, arguments, evidence, viewpoints
        - Structure the report properly
        - Integrate into a coherent narrative for a podcast
        - The report should be neutral
        
        Source Material:
        {source_material}
        """
        
        response = self.llm.invoke(prompt)
        self.log(f"Generated draft ({len(response.content)} chars)")
        
        return {"draft_report": response.content}


class FinalWriterAgent(BaseAgent):
    """Agent responsible for final report compilation."""
    
    def __init__(self):
        super().__init__(model="gemini-2.5-flash-preview-04-17", temperature=0.5)
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile final polished report.
        
        Args:
            state: Current research state with full context
            
        Returns:
            Updated state with final_report
        """
        self.log("Compiling final report...")
        sleep(10)
        
        full_context = (
            f"Topic: {state['topic']}\n\n"
            "Research History and Critiques:\n" + "\n".join(state['research_history']) +
            "\n\nFinal Draft Report to Polish:\n" + state['draft_report']
        )
        
        prompt = f"""
        You are the lead editor. Produce the final, polished version of a research report
        by integrating all context and the latest draft.
        
        Full Context:
        {full_context}
        
        Generate a final report based on this context, well structured as a discussion
        covering different viewpoints mentioned in it.
        """
        
        response = self.llm.invoke(prompt)
        self.log(f"Final report complete ({len(response.content)} chars)")
        
        return {"final_report": response.content}
