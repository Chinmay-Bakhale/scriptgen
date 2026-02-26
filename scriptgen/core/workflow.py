"""Main workflow orchestrator using modular agents."""
import re
import time
import json
from pathlib import Path
from typing import Dict, Any

from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv

from .state import ResearchState
from ..agents import (
    TopicScout,
    PlannerAgent,
    SearchAgent,
    ExtractorAgent,
    FilterAgent,
    RetrieverAgent,
    KnowledgeStoreAgent,
    WriterAgent,
    FinalWriterAgent,
    JudgeAgent
)
from ..metrics.evaluator import ReportEvaluator
from ..utils.knowledge_base import KnowledgeBase
from scriptgen.core import state

load_dotenv()


class MultiAgentResearchSystem:
    """Orchestrates multi-agent research workflow."""
    
    def __init__(self):
        """Initialize the research system with all agents."""
        # Initialize agents
        self.knowledge_base = KnowledgeBase()
        self.planner = PlannerAgent()
        self.searcher = SearchAgent()
        self.extractor = ExtractorAgent()
        self.filter = FilterAgent()
        self.retriever = RetrieverAgent(self.knowledge_base)
        self.knowledge_store = KnowledgeStoreAgent(self.knowledge_base)
        self.writer = WriterAgent()
        self.final_writer = FinalWriterAgent()
        self.judge = JudgeAgent()
        self.evaluator = ReportEvaluator()
        
        # Build and compile workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _planner_node(self, state: ResearchState) -> Dict[str, Any]:
        """Planner agent node."""
        return self.planner.execute(state)
    
    def _filter_node(self, state: ResearchState) -> Dict[str, Any]:
        """Filter agent node."""
        return self.filter.execute(state)
    
    def _retriever_node(self, state: ResearchState) -> Dict[str, Any]:
        """RAG retrieval node."""
        return self.retriever.execute(state)

    def _knowledge_store_node(self, state: ResearchState) -> Dict[str, Any]:
        """Knowledge store node."""
        return self.knowledge_store.execute(state)

    def _searcher_node(self, state: ResearchState) -> Dict[str, Any]:
        """Searcher agent node."""
        return self.searcher.execute(state)
    
    def _extractor_node(self, state: ResearchState) -> Dict[str, Any]:
        """Extractor agent node."""
        return self.extractor.execute(state)
    
    def _writer_node(self, state: ResearchState) -> Dict[str, Any]:
        """Writer agent node."""
        return self.writer.execute(state)
    
    def _judge_node(self, state: ResearchState) -> Dict[str, Any]:
        """Judge agent node."""
        return self.judge.execute(state)
    
    def _final_writer_node(self, state: ResearchState) -> Dict[str, Any]:
        """Final writer agent node."""
        return self.final_writer.execute(state)
    
    def _should_continue(self, state: ResearchState) -> str:
        """Determine if workflow should continue or end."""
        if state['iteration'] > 2:
            return "end_workflow"
        return "continue_to_judge"
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("searcher", self._searcher_node)
        workflow.add_node("extractor", self._extractor_node)
        workflow.add_node("knowledge_store", self._knowledge_store_node)
        workflow.add_node("filter", self._filter_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("judge", self._judge_node)
        workflow.add_node("final_writer", self._final_writer_node)
        
        # Add edges
        workflow.add_edge(START, "retriever")
        workflow.add_edge("retriever", "planner")
        workflow.add_edge("planner", "searcher")
        workflow.add_edge("searcher", "extractor")
        workflow.add_edge("extractor", "knowledge_store")
        workflow.add_edge("knowledge_store", "filter")
        workflow.add_edge("filter", "writer")
        workflow.add_conditional_edges(
            "writer",
            self._should_continue,
            {
                "continue_to_judge": "judge",
                "end_workflow": "final_writer"
            }
        )
        workflow.add_edge("judge", "retriever")  # Loop back for next iteration
        workflow.add_edge("final_writer", END)
        
        return workflow
    
    def run(self, topic: str = None):
        """
        Run the complete research workflow.
        
        Args:
            topic: Research topic (optional, will prompt user if not provided)
            
        Returns:
            Final report content
        """
        start_time = time.time()
        
        # Step 1: Determine topic
        if not topic:
            while True:
                choice = input(
                    "Choose an option:\n"
                    "1. Enter a topic manually\n"
                    "2. Let the AI find a trending topic\n"
                    "Your choice: "
                ).strip()
                if choice in ['1', '2']:
                    break
                print("Invalid choice. Please enter 1 or 2.")
            
            if choice == '1':
                topic = input("Please enter the research topic: ").strip()
            else:
                scout = TopicScout()
                topic = scout.find_trending_topic()
                if "Error:" in topic:
                    print(topic)
                    return
                print(f"\n🤖 AI has selected the topic: \"{topic}\"")
        
        # Step 2: Run workflow
        initial_state = {
            "topic": topic,
            "iteration": 1,
            "plan": "",
            "search_queries": [],
            "raw_search_results": [],
            "extracted_pages": [],
            "draft_report": "",
            "critique": "",
            "research_history": [],
            "final_report": "",
            "prior_context": "",
        }
        
        final_state = None
        for output in self.app.stream(initial_state):
            for node_name, state_after_node in output.items():
                print(f"\n--- Output from: {node_name} ---")
                final_state = state_after_node
                
                if node_name == "planner":
                    print(f"Plan: {state_after_node['plan']}")
                    print(f"Queries: {state_after_node['search_queries']}")
                elif node_name == "writer":
                    print("Draft Report Generated (showing first 200 chars):")
                    print(state_after_node['draft_report'][:200])
                elif node_name == "judge":
                    print("\n**Critique and Suggestions from the Judge:**")
                    print(state_after_node['critique'])
        
        execution_time = time.time() - start_time
        
        # Step 3: Save report
        final_report_content = final_state.get('final_report', "No report was generated.")
        report_filename = "final_research_report_" + re.sub(
            r'[^\w\s-]', '', topic.lower()
        ).replace(' ', '_')[:50] + ".md"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(final_report_content)
        
        print(f"\n✅ Workflow complete! Final report saved to '{report_filename}'")
        
        # Step 4: Evaluate and save metrics
        if final_report_content and "No report was generated." not in final_report_content:
            print("\n📊 Evaluating report quality...")
            
            sources = final_state.get('extracted_pages', [])
            quality_summary = final_state.get('quality_summary', {})
            metrics = self.evaluator.evaluate_report(
                report=final_report_content,
                topic=topic,
                sources=sources,
                execution_time=execution_time
            )

            # Merge quality summary into metrics
            metrics["source_quality"] = quality_summary
            
            print(self.evaluator.format_metrics_report(metrics))
            
            # Save metrics
            metrics_filename = report_filename.replace('.md', '_metrics.json')
            metrics_data = {
                "topic": topic,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "report_file": report_filename,
                "metrics": metrics,
                "search_latency_seconds": final_state.get('search_latency_seconds', 0)
            }
            
            with open(metrics_filename, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2)
            
            print(f"📈 Metrics saved to '{metrics_filename}'")
            
            # Append to history
            history_file = Path("metrics_history.json")
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            history.append(metrics_data)
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            print(f"📚 Metrics appended to 'metrics_history.json'")
        
        return final_report_content
