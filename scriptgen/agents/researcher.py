"""Research agents for search and content extraction."""
import asyncio
import time
from typing import Dict, Any, List
from langchain_tavily import TavilySearch, TavilyExtract
from .base import BaseAgent


class SearchAgent(BaseAgent):
    """
    Agent responsible for executing web searches.
    Runs multiple queries concurrently using asyncio
    for significantly reduced latency.
    """

    def __init__(self):
        super().__init__(model="sarvam-m", temperature=0.7)
        self.search_tool = TavilySearch(
            max_results=2,
            topic="general",
            search_depth="basic"
        )

    # ------------------------------------------------------------------
    # Public execute method (sync entry point for LangGraph)
    # ------------------------------------------------------------------

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search queries concurrently.

        Args:
            state: Current research state with search_queries

        Returns:
            Updated state with raw_search_results and search_latency_seconds
        """
        valid_queries = [q.strip("\"'") for q in state['search_queries'] if q.strip()]

        if not valid_queries:
            self.log("No valid queries to execute")
            return {"raw_search_results": [], "search_latency_seconds": 0.0}

        self.log(f"Executing {len(valid_queries)} queries concurrently...")
        start = time.time()

        # Run async search inside sync context (LangGraph is sync)
        results = asyncio.run(self._search_all(valid_queries))

        elapsed = round(time.time() - start, 2)
        self.log(f"All queries completed in {elapsed}s | "
                 f"Total results: {len(results)}")

        return {
            "raw_search_results": results,
            "search_latency_seconds": elapsed
        }

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _search_all(self, queries: List[str]) -> List[dict]:
        """
        Fire all queries concurrently and collect results.

        Args:
            queries: List of search query strings

        Returns:
            Flat list of all result dicts
        """
        tasks = [self._search_one(q) for q in queries]
        results_per_query = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for query, result in zip(queries, results_per_query):
            if isinstance(result, Exception):
                self.log(f"Query failed: '{query}' → {result}")
            elif result:
                all_results.extend(result)
                self.log(f"  ✓ '{query}' → {len(result)} result(s)")
            else:
                self.log(f"  ✗ '{query}' → no results")

        return all_results

    async def _search_one(self, query: str) -> List[dict]:
        """
        Execute a single search query asynchronously.

        Wraps the synchronous TavilySearch in a thread executor
        so it doesn't block the event loop.

        Args:
            query: Search query string

        Returns:
            List of result dicts for this query
        """
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.search_tool.invoke({"query": query})
            )
            if response and "results" in response:
                return response["results"]
            return []
        except Exception as e:
            raise RuntimeError(f"Search error for '{query}': {e}") from e


class ExtractorAgent(BaseAgent):
    """Agent responsible for extracting full page content."""

    def __init__(self):
        super().__init__(model="sarvam-m", temperature=0.5)

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract full content from URLs.

        Args:
            state: Current research state with raw_search_results

        Returns:
            Updated state with extracted_pages
        """
        urls = [result["url"] for result in state["raw_search_results"]]
        urls = urls[-4:]  # Take last 4 URLs

        self.log(f"Extracting {len(urls)} URL(s)")

        if not urls:
            self.log("No URLs to extract")
            return {"extracted_pages": []}

        extractor = TavilyExtract(
            extract_depth="advanced",
            format="markdown",
            inlude_images=False
        )

        try:
            extract_resp = extractor.invoke({"urls": urls})
            results = extract_resp.get("results", [])
            self.log(f"Successfully extracted {len(results)} page(s)")
            return {"extracted_pages": results}
        except Exception as e:
            self.log(f"Extraction failed: {e}")
            return {"extracted_pages": []}
