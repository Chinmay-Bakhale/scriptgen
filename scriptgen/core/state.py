"""State management for workflow."""
from typing import List, TypedDict, Annotated
import operator


class ResearchState(TypedDict):
    """State shared across all workflow agents."""
    topic: str
    iteration: int
    plan: str
    search_queries: List[str]
    raw_search_results: Annotated[List[dict], operator.add]
    extracted_pages: Annotated[List[dict], operator.add]
    draft_report: str
    critique: str
    research_history: Annotated[List[str], operator.add]
    final_report: str
