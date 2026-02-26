"""
Query Router Module
===================
This module handles dynamic query routing for hybrid RAG.

Classifies queries into:
- document: Local knowledge base only
- web: Web search only
- hybrid: Both local and web search
"""

from typing import Literal, Dict, Any, List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
from core.relevance_checker import RelevanceChecker

QueryType = Literal["document", "web", "hybrid"]

QueryType = Literal["document", "web", "hybrid"]

ROUTING_PROMPT = """You are a query classifier for a hybrid RAG system. Analyze the user's question and classify it into one of three categories:

1. **document**: Questions about specific documents, internal knowledge, or content that would be in uploaded files. Examples:
   - "What does the report say about X?"
   - "Explain the concept from the PDF"
   - "Summarize chapter 3"

2. **web**: Questions about current events, recent developments, or information not likely to be in local documents. Examples:
   - "What's the latest news about AI?"
   - "Current stock price of X"
   - "Recent developments in machine learning"

3. **hybrid**: Questions that combine local context with current information, or when you're unsure. Examples:
   - "How does our company policy compare to current industry standards?"
   - "Apply this concept to recent events"
   - Questions that reference both local docs and external knowledge

Question: {question}

Respond with ONLY the category name (document/web/hybrid) and a brief explanation (max 20 words).
Format: category | explanation"""


class QueryRouter:
    """
    Routes queries to appropriate search sources based on content analysis.
    """

    def __init__(self):
        """Initialize the query router with Groq LLM."""
        self.llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=0.1,  # Low temperature for consistent classification
            api_key=settings.GROQ_API_KEY
        )
        self.prompt = ChatPromptTemplate.from_template(ROUTING_PROMPT)
        self.relevance_checker = RelevanceChecker()

    def classify_query(self, query: str) -> Dict[str, str]:
        """
        Classify a query into document/web/hybrid categories.

        Args:
            query: User's question

        Returns:
            Dict with 'category' and 'reason'
        """
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"question": query}).content.strip()

            # Parse response: "category | explanation"
            if "|" in response:
                category, reason = response.split("|", 1)
                category = category.strip().lower()
                reason = reason.strip()
            else:
                # Fallback if format is unexpected
                category = response.lower().strip()
                reason = "Auto-classified"

            # Validate category
            if category not in ["document", "web", "hybrid"]:
                category = "hybrid"  # Default to hybrid if unclear
                reason = "Unclear query type, using hybrid search"

            return {
                "category": category,
                "reason": reason
            }

        except Exception as e:
            # Fallback on error
            return {
                "category": "hybrid",
                "reason": f"Classification failed: {str(e)}"
            }

    def route_with_relevance_check(self, query: str, documents: List[Document],
                                   web_search_enabled: bool) -> Dict[str, Any]:
        """
        Advanced routing that considers both query intent and local content relevance.

        Args:
            query: User's question
            documents: Retrieved local documents
            web_search_enabled: Whether web search toggle is on

        Returns:
            Dict with comprehensive routing decision
        """
        # First, classify query intent
        classification = self.classify_query(query)
        intent_category = classification["category"]

        # Local-only routing: never use web search. Evaluate local relevance for diagnostics.
        if documents:
            evaluation = self.relevance_checker.evaluate_relevance(query, documents)
        else:
            evaluation = None

        return {
            "use_web_search": False,
            "use_document_search": True if documents else False,
            "category": "document",
            "reason": classification.get("reason", "Local-only routing"),
            "relevance_check": evaluation
        }