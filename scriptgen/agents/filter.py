"""Source quality filter agent."""
from typing import Dict, Any
from .base import BaseAgent
from ..utils.scorer import SourceQualityScorer


class FilterAgent(BaseAgent):
    """
    Agent that scores and filters extracted sources
    before they reach the writer.
    """

    def __init__(self, min_score: float = 0.3):
        """
        Initialize filter agent.

        Args:
            min_score: Minimum quality threshold (0.0 - 1.0)
        """
        super().__init__(model="sarvam-m", temperature=0.7)
        self.scorer = SourceQualityScorer(min_score=min_score)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score and filter extracted pages.

        Args:
            state: Current research state with extracted_pages

        Returns:
            Updated state with scored + filtered extracted_pages
        """
        pages = state.get("extracted_pages", [])
        topic = state.get("topic", "")

        if not pages:
            self.log("No pages to filter")
            return {"extracted_pages": []}

        self.log(f"Scoring {len(pages)} source(s)...")

        # Score and filter
        filtered = self.scorer.filter_sources(pages, topic)
        summary = self.scorer.get_score_summary(
            self.scorer.score_sources(pages, topic)
        )

        # Log summary
        self.log(
            f"Kept {len(filtered)}/{len(pages)} sources | "
            f"Avg score: {summary.get('avg_score', 0):.2f} | "
            f"High quality: {summary.get('high_quality_count', 0)}"
        )

        # Print score table
        scored_all = self.scorer.score_sources(pages, topic)
        print("\n  📊 Source Quality Scores:")
        print(f"  {'URL':<55} {'Final':>6} {'Domain':>7} {'Content':>8} {'Relevance':>10}")
        print("  " + "-" * 90)
        for s in scored_all:
            q = s["quality_scores"]
            url_short = s.get("url", "")[:52] + "..." if len(s.get("url", "")) > 52 else s.get("url", "")
            kept = "✅" if s in filtered else "❌"
            print(f"  {kept} {url_short:<52} {q['final']:>6.2f} {q['domain']:>7.2f} {q['content']:>8.2f} {q['relevance']:>10.2f}")
        print()

        return {
            "extracted_pages": filtered,
            "quality_summary": summary
        }
