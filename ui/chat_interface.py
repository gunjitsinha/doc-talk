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
from pathlib import Path
import tempfile
import os

from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.chain import RAGChain
from core.router import QueryRouter
from ui.components import add_message


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
        # Admin-managed Documents folder: automatically index files at startup
        # if the vector store isn't already initialized.
        base_dir = Path(__file__).resolve().parents[1]
        docs_dir = base_dir / "Documents"
        docs_dir.mkdir(parents=True, exist_ok=True)

        if not self.vector_store.is_initialized:
            all_chunks = []
            for p in docs_dir.iterdir():
                if p.is_file() and p.suffix.lower() in (".pdf", ".txt"):
                    try:
                        chunks = self.doc_processor.process(str(p))
                        for chunk in chunks:
                            chunk.metadata["source"] = p.name
                        all_chunks.extend(chunks)
                    except Exception as e:
                        # continue indexing other files even if one fails
                        print(f"Failed to process {p}: {e}")

            if all_chunks:
                self.vector_store.create_from_documents(all_chunks)
                st.session_state.vector_store_initialized = True
        self.rag_chain: Optional[RAGChain] = None
        self.hybrid_search: Optional[object] = None
        self.query_router = QueryRouter()
    
    def process_uploaded_files(self, uploaded_files, save_to_documents: bool = False) -> int:
        """
        Process uploaded files and add to vector store.
        
        Args:
            uploaded_files: List of Streamlit UploadedFile objects
            
        Returns:
            Number of chunks processed
        """
        all_chunks = []

        base_dir = Path(__file__).resolve().parents[1]
        docs_dir = base_dir / "Documents"
        docs_dir.mkdir(parents=True, exist_ok=True)

        for uploaded_file in uploaded_files:
            if save_to_documents:
                # Persist uploaded file into Documents/ before processing
                dest = docs_dir / uploaded_file.name
                stem = dest.stem
                suffix = dest.suffix
                counter = 1
                while dest.exists():
                    dest = docs_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

                with open(dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                file_path = str(dest)

                # Ensure session state reflects persisted documents
                persisted_name = Path(file_path).name
                if not persisted_name.startswith('.git') and persisted_name not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(persisted_name)

            else:
                # Write to a temporary file (do NOT persist in the repo)
                suffix = Path(uploaded_file.name).suffix or ""
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    file_path = tmp.name

            try:
                chunks = self.doc_processor.process(file_path)
                for chunk in chunks:
                    chunk.metadata["source"] = uploaded_file.name
                all_chunks.extend(chunks)
            finally:
                # Remove the temporary file if we created one
                if not save_to_documents:
                    try:
                        os.unlink(file_path)
                    except Exception:
                        pass

        # Add to vector store (in-memory / persisted by vector store save)
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            st.session_state.vector_store_initialized = True

        # Return number of chunks processed
        return len(all_chunks)
    
    def initialize_rag_chain(self):
        """Initialize the RAG chain after documents are loaded."""
        if self.vector_store.is_initialized:
            self.rag_chain = RAGChain(self.vector_store)
            self.hybrid_search = None
    
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

        # Route (local-only): keep routing for relevance metadata but disable web
        routing_decision = self.query_router.route_with_relevance_check(
            query, doc_results, False
        )

        # If no local documents available, prompt user to upload
        if not doc_results:
            yield "Please upload some documents first to use this local-only assistant."
            return

        # Document-only response
        yield from self._get_document_only_response(query)
    
    def _get_hybrid_response(self, query: str, routing_decision: dict) -> Generator[str, None, None]:
        """Generate response using both document and web search."""
        # Hybrid path removed in local-only mode
        yield from ()
    
    def _get_web_only_response(self, query: str) -> Generator[str, None, None]:
        """Generate response using only web search."""
        # Web-only responses are disabled in local-only mode
        yield from ()
    
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
        
        # Web sources removed for local-only mode
        
        return sources