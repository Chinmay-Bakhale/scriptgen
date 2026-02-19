"""Research agents for search and content extraction."""
from typing import Dict, Any, List
from time import sleep
from langchain_tavily import TavilySearch, TavilyExtract
from .base import BaseAgent


class SearchAgent(BaseAgent):
    """Agent responsible for executing web searches."""
    
    def __init__(self):
        super().__init__(model="gemini-2.5-flash", temperature=0.5)
        self.search_tool = TavilySearch(max_results=2, topic="general", search_depth="basic")
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search queries from planner.
        
        Args:
            state: Current research state with search_queries
            
        Returns:
            Updated state with raw_search_results
        """
        self.log("Executing search queries...")
        all_results = []
        
        # Filter empty queries
        valid_queries = [q.strip("\"'") for q in state['search_queries'] if q]
        self.log(f"Processing {len(valid_queries)} queries")
        
        for query in valid_queries:
            self.log(f"Searching: {query}")
            sleep(2)  # Rate limiting
            
            try:
                results = self.search_tool.invoke({"query": query})
                if results and 'results' in results:
                    all_results.extend(results['results'])
                    self.log(f"Found {len(results['results'])} result(s)")
                else:
                    self.log("No results returned")
            except Exception as e:
                self.log(f"Search failed: {e}")
        
        self.log(f"Total results gathered: {len(all_results)}")
        return {"raw_search_results": all_results}


class ExtractorAgent(BaseAgent):
    """Agent responsible for extracting full page content."""
    
    def __init__(self):
        super().__init__(model="gemini-2.5-flash", temperature=0.5)
    
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
        
        self.log(f"Extracting {len(urls)} URLs")
        
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
