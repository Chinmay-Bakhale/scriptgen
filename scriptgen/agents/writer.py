"""Writing agents for report generation."""
import os
from typing import Dict, Any
from time import sleep
from langchain_google_genai import ChatGoogleGenerativeAI
from .base import BaseAgent


def _to_text(value: Any) -> str:
    """Normalize provider responses/state values into plain text."""

    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(value)

MAX_WRITER_PAGES = 3
MAX_CHARS_PER_PAGE = 7000
MAX_FINAL_DRAFT_CHARS = 12000
MAX_HISTORY_ITEMS = 3
MAX_HISTORY_CHARS = 1200

class WriterAgent(BaseAgent):
    """Agent responsible for drafting research reports."""
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.5,
            google_api_key=api_key,
            max_output_tokens=4096,
        )
        self.agent_name = self.__class__.__name__

    
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
        
        pages = state.get("extracted_pages", [])[:MAX_WRITER_PAGES]

        trimmed_chunks = []
        for page in pages:
            raw = _to_text(page.get("raw_content", "")).strip()
            if raw:
                trimmed_chunks.append(raw[:MAX_CHARS_PER_PAGE])

        if not trimmed_chunks:
            return {"draft_report": "No extracted page content available for this iteration."}

        source_material = "\n\n---\n\n".join(trimmed_chunks)
        self.log(
            f"Writer input: {len(trimmed_chunks)} page(s), "
            f"{sum(len(c) for c in trimmed_chunks)} chars"
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
        draft_text = _to_text(response.content)
        self.log(f"Generated draft ({len(draft_text)} chars)")
        
        return {"draft_report": draft_text}


class FinalWriterAgent(BaseAgent):
    """Agent responsible for final report compilation."""
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.4,
            google_api_key=api_key,
            max_output_tokens=4096,
        )
        self.agent_name = self.__class__.__name__

    
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
        
        history_items = state.get("research_history", [])[-MAX_HISTORY_ITEMS:]
        history_text = "\n\n".join(
            _to_text(item)[:MAX_HISTORY_CHARS] for item in history_items
        )
        draft_text = _to_text(state.get("draft_report", ""))[:MAX_FINAL_DRAFT_CHARS]


        full_context = (
            f"Topic: {state['topic']}\n\n"
            "Research History and Critiques:\n" + history_text +
            "\n\nFinal Draft Report to Polish:\n" + draft_text
        )

        
        prompt = f"""
        You are the lead editor. Produce the final, polished version of a research report
        by integrating all context and the latest draft.
        
        Full Context:
        {full_context}
        
        Generate a final report in 8000 words based on this context, well structured as a discussion
        covering different viewpoints mentioned in it.
        """
        
        response = self.llm.invoke(prompt)
        final_text = _to_text(response.content)
        self.log(f"Final report complete ({len(final_text)} chars)")

        return {"final_report": final_text}
