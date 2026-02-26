"""RAG retrieval agent using ChromaDB knowledge base."""
from typing import Dict, Any
from .base import BaseAgent
from ..utils.knowledge_base import KnowledgeBase


class RetrieverAgent(BaseAgent):
    """
    Agent that retrieves relevant prior knowledge before each
    research iteration. Enables the workflow to build on
    previous findings rather than starting from scratch.
    """

    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize retriever.

        Args:
            knowledge_base: Shared KnowledgeBase instance
        """
        super().__init__(model="sarvam-m", temperature=0.7)
        self.kb = knowledge_base

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant prior context for the current topic.

        Runs before the planner on iteration > 1 so the planner
        can avoid re-researching already-known facts.

        Args:
            state: Current research state

        Returns:
            Updated state with prior_context field
        """
        topic = state.get("topic", "")
        iteration = state.get("iteration", 1)

        stats = self.kb.get_stats()
        self.log(f"Knowledge base has {stats['total_documents']} doc(s)")

        if stats["total_documents"] == 0:
            self.log("Knowledge base empty — skipping retrieval")
            return {"prior_context": ""}

        # Retrieve top relevant docs for this topic
        retrieved = self.kb.retrieve_for_topic(topic, n_results=3)

        if not retrieved:
            self.log("No relevant prior knowledge found")
            return {"prior_context": ""}

        self.log(
            f"Retrieved {len(retrieved)} relevant doc(s) from past research "
            f"(best relevance: {retrieved[0]['relevance_score']:.2f})"
        )

        context = self.kb.format_context(retrieved)
        return {"prior_context": context}


class KnowledgeStoreAgent(BaseAgent):
    """
    Agent that stores newly extracted pages into the knowledge base
    after each extraction step.
    """

    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize store agent.

        Args:
            knowledge_base: Shared KnowledgeBase instance
        """
        super().__init__(model="sarvam-m", temperature=0.5)
        self.kb = knowledge_base

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store extracted pages into the knowledge base.

        Args:
            state: Current research state with extracted_pages

        Returns:
            State unchanged (side-effect only: writes to KB)
        """
        pages = state.get("extracted_pages", [])
        topic = state.get("topic", "")
        iteration = state.get("iteration", 1)

        if not pages:
            self.log("No pages to store")
            return {}

        self.log(f"Storing {len(pages)} page(s) into knowledge base...")
        self.kb.add_documents(pages, topic=topic, iteration=iteration)

        stats = self.kb.get_stats()
        self.log(f"Knowledge base now has {stats['total_documents']} doc(s)")

        return {}
