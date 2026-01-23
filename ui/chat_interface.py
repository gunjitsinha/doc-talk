"""
Chat Interface Module
=====================
DAY 4: Main chat interface logic.

SOLID Principle: Single Responsibility Principle (SRP)
- This module handles chat orchestration

Topics to teach:
- Streamlit chat elements
- Streaming responses
- Error handling
- User experience
"""

import streamlit as st
from typing import Generator, Optional

from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.chain import RAGChain
from core.router import QueryRouter
from tools.tavily_search import TavilySearchTool, HybridSearchManager
from ui.components import add_message, save_uploaded_file


class ChatInterface:
    """
    Main chat interface orchestrator.
    
    Coordinates between:
    - Document processing
    - Vector store
    - RAG chain
    - Web search
    """
    
    def __init__(self):
        """Initialize chat interface components."""
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
        if self.vector_store.is_initialized:
            st.session_state.vector_store_initialized = True
        self.rag_chain: Optional[RAGChain] = None
        self.tavily_search = TavilySearchTool()
        self.hybrid_search: Optional[HybridSearchManager] = None
        self.query_router = QueryRouter()
    
    def process_uploaded_files(self, uploaded_files) -> int:
        """
        Process uploaded files and add to vector store.
        
        Args:
            uploaded_files: List of Streamlit UploadedFile objects
            
        Returns:
            Number of chunks processed
        """
        all_chunks = []
        
        for uploaded_file in uploaded_files:
            # Save file temporarily
            file_path = save_uploaded_file(uploaded_file)
            
            # Process the document
            chunks = self.doc_processor.process(file_path)
            
            # Add source metadata
            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.name
            
            all_chunks.extend(chunks)
            
            # Track uploaded files
            if uploaded_file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file.name)
        
        # Add to vector store
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            st.session_state.vector_store_initialized = True
        
        return len(all_chunks)
    
    def initialize_rag_chain(self):
        """Initialize the RAG chain after documents are loaded."""
        if self.vector_store.is_initialized:
            self.rag_chain = RAGChain(self.vector_store)
            self.hybrid_search = HybridSearchManager(
                self.vector_store,
                self.tavily_search
            )
    
    def get_response(
        self,
        query: str,
        use_web_search: bool = False
    ) -> Generator[str, None, None]:
        """
        Get a streaming response for a query with dynamic routing.
        
        Args:
            query: User's question
            use_web_search: Whether web search toggle is enabled
            
        Yields:
            Response chunks
        """
        # Initialize RAG chain if needed
        if self.rag_chain is None and self.vector_store.is_initialized:
            self.initialize_rag_chain()
        
        # Always do initial local search to check relevance
        doc_results = []
        if self.vector_store.is_initialized:
            doc_results = self.vector_store.search(query)
        
        # Use intelligent routing with relevance checking
        routing_decision = self.query_router.route_with_relevance_check(
            query, doc_results, use_web_search
        )
        
        use_doc_search = routing_decision["use_document_search"]
        use_web = routing_decision["use_web_search"]
        
        # If no sources available, provide helpful message
        if not use_doc_search and not use_web:
            yield "Please upload some documents first, or enable web search to get started!"
            return
        
        # Handle different routing scenarios
        if use_web and use_doc_search:
            # Hybrid search - may include relevance info
            yield from self._get_hybrid_response(query, routing_decision)
        elif use_web and not use_doc_search:
            # Web-only search
            yield from self._get_web_only_response(query)
        else:
            # Document-only search
            yield from self._get_document_only_response(query)
    
    def _get_hybrid_response(self, query: str, routing_decision: dict) -> Generator[str, None, None]:
        """Generate response using both document and web search."""
        # Get web search results
        web_results = self.tavily_search.search(query)
        
        # Use document results from routing (already retrieved)
        doc_results = []
        if self.vector_store.is_initialized:
            doc_results = self.vector_store.search(query)
        
        # Format context with citations
        context_parts = []
        
        if doc_results:
            context_parts.append("=== DOCUMENT SOURCES ===")
            for i, doc in enumerate(doc_results, 1):
                source = doc.metadata.get("source", "Unknown")
                context_parts.append(f"[Doc{i}] ({source}):\n{doc.page_content}")
        
        if web_results:
            context_parts.append("\n=== WEB SOURCES ===")
            context_parts.append(web_results)
        
        context = "\n\n".join(context_parts) if context_parts else "No context available."
        
        # Generate citation-aware response
        yield from self._generate_citation_response(query, context, "hybrid")
    
    def _get_web_only_response(self, query: str) -> Generator[str, None, None]:
        """Generate response using only web search."""
        web_results = self.tavily_search.search(query)
        context = f"=== WEB SOURCES ===\n{web_results}"
        yield from self._generate_citation_response(query, context, "web")
    
    def _get_document_only_response(self, query: str) -> Generator[str, None, None]:
        """Generate response using only document search."""
        if not self.rag_chain:
            yield "No documents available. Please upload documents first."
            return
        
        # Use the existing RAG chain but with citation formatting
        documents = self.vector_store.search(query)
        
        # Format context with citations
        context_parts = ["=== DOCUMENT SOURCES ==="]
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[Doc{i}] ({source}):\n{doc.page_content}")
        context = "\n\n".join(context_parts)
        
        yield from self._generate_citation_response(query, context, "document")
    
    def _generate_citation_response(self, query: str, context: str, search_type: str) -> Generator[str, None, None]:
        """Generate response with citation awareness."""
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        from config.settings import settings
        
        llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.GROQ_API_KEY
        )
        
        # Citation-aware prompt
        prompt_template = """You are a helpful AI assistant. Answer the question based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer accurately using only the provided context
- Include specific citations in your answer using [Doc1], [Web1], etc.
- If using multiple sources, cite all relevant ones
- Keep your answer concise but comprehensive
- If the context doesn't contain enough information, say so

ANSWER:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        
        for chunk in chain.stream({"context": context, "question": query}):
            yield chunk.content
    
    def get_sources(self, query: str, use_web_search: bool = False) -> dict:
        """
        Get detailed source information for a query.
        
        Args:
            query: User's question
            use_web_search: Whether web search toggle is enabled
            
        Returns:
            Dict with routing info and source details
        """
        # Get document results first for relevance checking
        doc_results = []
        if self.vector_store.is_initialized:
            doc_results = self.vector_store.search(query)
        
        routing_decision = self.query_router.route_with_relevance_check(query, doc_results, use_web_search)
        
        sources = {
            "routing": routing_decision,
            "document_sources": [],
            "web_sources": []
        }
        
        # Get document sources
        if routing_decision["use_document_search"] and self.vector_store.is_initialized:
            docs = self.vector_store.search(query)
            sources["document_sources"] = [
                {
                    "name": doc.metadata.get("source", "Unknown"),
                    "type": "document",
                    "content_preview": doc.page_content[:200] + "..."
                }
                for doc in docs
            ]
        
        # Get web sources
        if routing_decision["use_web_search"]:
            web_results = self.tavily_search.search_with_context(query)
            if web_results and isinstance(web_results, dict) and web_results.get("results"):
                web_items = []
                for result in web_results["results"]:
                    # Normalize each result to a dict in case the API returns plain strings
                    if isinstance(result, dict):
                        title = result.get("title", "No title")
                        url = result.get("url", "")
                        content = result.get("content", "")
                    else:
                        # result is a string -> treat as content snippet
                        title = "(no title)"
                        url = ""
                        content = str(result)

                    web_items.append({
                        "title": title,
                        "url": url,
                        "content_preview": content[:200] + "...",
                        "type": "web"
                    })

                sources["web_sources"] = web_items
        
        return sources