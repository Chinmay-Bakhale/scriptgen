"""
Persistent knowledge base using FAISS for RAG.

Uses sentence-transformers for local embeddings and FAISS
for fast similarity search. No external API, no pydantic
dependency — fully compatible with Python 3.14+.
"""
import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

import faiss
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384   # fixed output dim for all-MiniLM-L6-v2
DEFAULT_DB_PATH = "./knowledge_store"


class KnowledgeBase:
    """
    Persistent vector store backed by FAISS + sentence-transformers.

    Documents are embedded locally, stored in a FAISS IndexFlatIP
    (inner product = cosine similarity after L2-normalisation), and
    persisted to disk as two files:
        - faiss.index   : the FAISS index binary
        - metadata.pkl  : doc metadata + id set
    """

    def __init__(self, persist_directory: str = DEFAULT_DB_PATH):
        """
        Initialize knowledge base, loading existing data if present.

        Args:
            persist_directory: Directory to persist FAISS index and metadata.
        """
        self.persist_dir   = Path(persist_directory)
        self.index_path    = self.persist_dir / "faiss.index"
        self.metadata_path = self.persist_dir / "metadata.pkl"

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Load the embedding model once — cached on disk after first download
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)

        # Runtime state
        self.metadata: List[Dict] = []
        self.doc_ids:  set        = set()

        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "rb") as f:
                saved          = pickle.load(f)
                self.metadata  = saved["metadata"]
                self.doc_ids   = saved["doc_ids"]
            print(f"[KnowledgeBase] Loaded existing store — "
                  f"{len(self.metadata)} doc(s)")
        else:
            # IndexFlatIP: exact inner-product search (cosine after normalise)
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            print("[KnowledgeBase] Created new FAISS store")

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add_documents(
        self,
        pages: List[dict],
        topic: str,
        iteration: int = 1
    ) -> None:
        """
        Embed and store extracted pages.

        Args:
            pages:     List of page dicts with 'url' and 'raw_content'.
            topic:     Research topic these pages belong to.
            iteration: Workflow iteration number.
        """
        if not pages:
            return

        new_docs: List[str] = []
        new_meta: List[Dict] = []

        for page in pages:
            content = page.get("raw_content", "") or page.get("content", "")
            url     = page.get("url", "")

            if not content or not url:
                continue

            doc_id = hashlib.md5(url.encode()).hexdigest()
            if doc_id in self.doc_ids:
                continue

            chunk = content[:4000]
            new_docs.append(chunk)
            new_meta.append({
                "doc_id":          doc_id,
                "url":             url,
                "topic":           topic,
                "iteration":       iteration,
                "word_count":      len(content.split()),
                "content_preview": chunk,
            })

        if not new_docs:
            print("[KnowledgeBase] All documents already stored — skipping")
            return

        embeddings = self._embed(new_docs)
        self.index.add(embeddings)

        for meta in new_meta:
            self.metadata.append(meta)
            self.doc_ids.add(meta["doc_id"])

        self._save()
        print(f"[KnowledgeBase] Stored {len(new_docs)} new doc(s) "
              f"(total: {len(self.metadata)})")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        n_results: int = 3,
        topic_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Semantically retrieve the most relevant documents.

        Args:
            query:        Natural language query string.
            n_results:    Maximum number of results to return.
            topic_filter: If set, restrict results to this topic only.

        Returns:
            List of dicts with content, url, topic, relevance_score.
        """
        if not self.metadata:
            return []

        query_vec = self._embed([query])
        k = min(n_results * 3, len(self.metadata))   # over-fetch for filtering

        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            if topic_filter and meta.get("topic") != topic_filter:
                continue

            results.append({
                "content":         meta["content_preview"],
                "url":             meta["url"],
                "topic":           meta["topic"],
                "relevance_score": round(float(dist), 4),
            })

            if len(results) >= n_results:
                break

        return results

    def retrieve_for_topic(
        self,
        topic: str,
        n_results: int = 3
    ) -> List[Dict]:
        """
        Retrieve documents most relevant to a research topic.

        Args:
            topic:     Research topic string.
            n_results: Number of results to return.
        """
        return self.retrieve(query=topic, n_results=n_results)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return summary statistics about the knowledge base."""
        return {
            "total_documents":  len(self.metadata),
            "persist_directory": str(self.persist_dir),
            "embedding_model":  EMBEDDING_MODEL,
        }

    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into a context string for LLM prompts.

        Args:
            retrieved_docs: Output of retrieve() or retrieve_for_topic().

        Returns:
            Formatted string ready to inject into a prompt.
        """
        if not retrieved_docs:
            return "No prior knowledge available."

        lines = ["=== Prior Research Context (from Knowledge Base) ===\n"]
        for i, doc in enumerate(retrieved_docs, 1):
            lines.append(
                f"[Doc {i}] Source: {doc['url']} "
                f"(relevance: {doc['relevance_score']:.2f})\n"
                f"{doc['content'][:600]}...\n"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts and L2-normalise for cosine similarity via inner product.

        Args:
            texts: List of strings to embed.

        Returns:
            float32 numpy array of shape (len(texts), EMBEDDING_DIM).
        """
        vecs = self.encoder.encode(texts, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vecs)
        return vecs

    def _exists(self, doc_id: str) -> bool:
        """Check whether a document is already stored."""
        return doc_id in self.doc_ids

    def _save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(
                {"metadata": self.metadata, "doc_ids": self.doc_ids},
                f
            )
