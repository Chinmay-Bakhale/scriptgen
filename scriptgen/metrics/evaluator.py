"""Report evaluation metrics module."""
import re
from typing import Dict, List, Any
from collections import Counter


class ReportEvaluator:
    """Evaluates research report quality with multiple metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def evaluate_report(
        self, 
        report: str, 
        topic: str, 
        sources: List[dict],
        execution_time: float = None
    ) -> Dict[str, Any]:
        """
        Evaluate a research report comprehensively.
        
        Args:
            report: The generated report text
            topic: The research topic
            sources: List of source dictionaries with URLs
            execution_time: Time taken to generate report (seconds)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            # Basic metrics
            "word_count": self._count_words(report),
            "sentence_count": self._count_sentences(report),
            "paragraph_count": self._count_paragraphs(report),
            
            # Readability metrics
            "avg_sentence_length": self._avg_sentence_length(report),
            "avg_word_length": self._avg_word_length(report),
            
            # Content metrics
            "source_count": len(sources),
            "unique_domains": self._count_unique_domains(sources),
            "citation_count": self._count_citations(report),
            
            # Topic relevance (simple keyword-based)
            "topic_mentions": self._count_topic_mentions(report, topic),
            
            # Structure metrics
            "has_sections": self._has_sections(report),
            "section_count": self._count_sections(report),
        }
        
        # Add execution time if provided
        if execution_time:
            metrics["execution_time_seconds"] = round(execution_time, 2)
        
        # Derived metrics
        metrics["words_per_source"] = (
            round(metrics["word_count"] / metrics["source_count"], 2) 
            if metrics["source_count"] > 0 else 0
        )
        
        return metrics
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences (rough approximation)."""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs."""
        paragraphs = text.split('\n\n')
        return len([p for p in paragraphs if p.strip()])
    
    def _avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        total_words = sum(len(s.split()) for s in sentences)
        return round(total_words / len(sentences), 2)
    
    def _avg_word_length(self, text: str) -> float:
        """Calculate average word length in characters."""
        words = text.split()
        if not words:
            return 0.0
        total_chars = sum(len(word) for word in words)
        return round(total_chars / len(words), 2)
    
    def _count_citations(self, text: str) -> int:
        """Count citations (assumes [source] format)."""
        return text.count('[')
    
    def _count_unique_domains(self, sources: List[dict]) -> int:
        """Count unique domains in sources."""
        domains = set()
        for source in sources:
            url = source.get('url', '')
            # Extract domain from URL
            match = re.search(r'https?://([^/]+)', url)
            if match:
                domain = match.group(1)
                # Remove www. prefix
                domain = domain.replace('www.', '')
                domains.add(domain)
        return len(domains)
    
    def _count_topic_mentions(self, text: str, topic: str) -> int:
        """Count mentions of topic keywords (case-insensitive)."""
        # Extract important words from topic (> 3 chars)
        topic_words = [w.lower() for w in topic.split() if len(w) > 3]
        text_lower = text.lower()
        
        mentions = 0
        for word in topic_words:
            mentions += text_lower.count(word)
        return mentions
    
    def _has_sections(self, text: str) -> bool:
        """Check if report has markdown sections (## headers)."""
        return bool(re.search(r'^##\s+', text, re.MULTILINE))
    
    def _count_sections(self, text: str) -> int:
        """Count markdown sections (## headers)."""
        return len(re.findall(r'^##\s+', text, re.MULTILINE))
    
    def format_metrics_report(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics as a readable string.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Formatted string report
        """
        report = "📊 Report Evaluation Metrics\n"
        report += "=" * 50 + "\n\n"
        
        report += "📝 Content Metrics:\n"
        report += f"  • Word Count: {metrics.get('word_count', 0)}\n"
        report += f"  • Sentence Count: {metrics.get('sentence_count', 0)}\n"
        report += f"  • Paragraph Count: {metrics.get('paragraph_count', 0)}\n"
        report += f"  • Citation Count: {metrics.get('citation_count', 0)}\n\n"
        
        report += "📚 Source Metrics:\n"
        report += f"  • Total Sources: {metrics.get('source_count', 0)}\n"
        report += f"  • Unique Domains: {metrics.get('unique_domains', 0)}\n"
        report += f"  • Words per Source: {metrics.get('words_per_source', 0)}\n\n"
        
        report += "📖 Readability:\n"
        report += f"  • Avg Sentence Length: {metrics.get('avg_sentence_length', 0)} words\n"
        report += f"  • Avg Word Length: {metrics.get('avg_word_length', 0)} chars\n\n"
        
        report += "🔍 Structure:\n"
        report += f"  • Has Sections: {metrics.get('has_sections', False)}\n"
        report += f"  • Section Count: {metrics.get('section_count', 0)}\n"
        report += f"  • Topic Mentions: {metrics.get('topic_mentions', 0)}\n"
        
        if 'execution_time_seconds' in metrics:
            report += f"\n⏱️  Execution Time: {metrics['execution_time_seconds']}s\n"
        
        return report
