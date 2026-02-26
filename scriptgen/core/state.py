"""State management for workflow."""
from typing import List, TypedDict, Annotated, Optional
import operator


class ResearchState(TypedDict):
    """State shared across all workflow agents."""
    topic: str
    iteration: int
    plan: str
    search_queries: List[str]
    raw_search_results: List[dict]
    Annotated[List[dict], operator.add]
    draft_report: str
    critique: str
    research_history: Annotated[List[str], operator.add]
    final_report: str
    quality_summary: dict   # NEW: source quality scores per iteration
    search_latency_seconds: float
    prior_context: str
