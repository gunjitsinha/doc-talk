"""
Relevance Checker Module
=======================
Evaluates if retrieved document chunks sufficiently answer a query.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from config.settings import settings

RELEVANCE_PROMPT = """You are an evaluator determining if retrieved document content sufficiently answers a user's question.

QUESTION: {question}

RETRIEVED CONTENT:
{content}

TASK: Analyze if the retrieved content provides enough information to fully answer the question.

Return your assessment in this exact format:
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASON: [Brief explanation, max 30 words]

HIGH: Content directly answers the question with specific details
MEDIUM: Content is related but may need supplementation
LOW: Content is insufficient or irrelevant to answer the question"""


class RelevanceChecker:
    """
    Evaluates the relevance and sufficiency of retrieved document chunks.
    """

    def __init__(self):
        """Initialize the relevance checker with Groq LLM."""
        self.llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=0.1,  # Low temperature for consistent evaluation
            api_key=settings.GROQ_API_KEY
        )

    def evaluate_relevance(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Evaluate if retrieved documents sufficiently answer the query.

        Args:
            query: User's question
            documents: Retrieved document chunks

        Returns:
            Dict with confidence level and reasoning
        """
        if not documents:
            return {
                "confidence": "LOW",
                "reason": "No documents retrieved",
                "score": 0.0
            }

        # Format content for evaluation
        content_parts = []
        for i, doc in enumerate(documents, 1):
            content_parts.append(f"[Chunk {i}]: {doc.page_content}")
        content = "\n\n".join(content_parts)

        try:
            # Get evaluation from LLM
            response = self.llm.invoke(RELEVANCE_PROMPT.format(
                question=query,
                content=content
            )).content.strip()

            # Parse response
            confidence = "MEDIUM"  # default
            reason = "Unable to determine"

            for line in response.split('\n'):
                if line.startswith('CONFIDENCE:'):
                    confidence = line.split(':', 1)[1].strip().upper()
                elif line.startswith('REASON:'):
                    reason = line.split(':', 1)[1].strip()

            # Convert to numeric score
            score_map = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.0}
            score = score_map.get(confidence, 0.5)

            return {
                "confidence": confidence,
                "reason": reason,
                "score": score
            }

        except Exception as e:
            return {
                "confidence": "MEDIUM",
                "reason": f"Evaluation failed: {str(e)}",
                "score": 0.5
            }

    def is_sufficient(self, evaluation: Dict[str, Any], threshold: float = 0.7) -> bool:
        """
        Determine if the evaluation indicates sufficient relevance.

        Args:
            evaluation: Result from evaluate_relevance
            threshold: Minimum score to consider sufficient

        Returns:
            True if content is sufficient, False otherwise
        """
        return evaluation["score"] >= threshold

    def should_augment_with_web(self, query: str, documents: List[Document],
                               threshold: float = 0.7) -> Dict[str, Any]:
        """
        Determine if web search should be used to augment local results.

        Args:
            query: User's question
            documents: Retrieved document chunks
            threshold: Sufficiency threshold

        Returns:
            Dict with decision and evaluation details
        """
        evaluation = self.evaluate_relevance(query, documents)
        sufficient = self.is_sufficient(evaluation, threshold)

        return {
            "should_augment": not sufficient,
            "evaluation": evaluation,
            "reason": evaluation["reason"]
        }