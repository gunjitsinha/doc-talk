"""
UI Components Module
====================
DAY 4: Reusable Streamlit components.

SOLID Principle: Single Responsibility Principle (SRP)
- Each component has ONE job

Topics to teach:
- Streamlit components
- Reusability
- Session state management
- File upload handling
"""

import streamlit as st
from typing import List, Optional
from pathlib import Path
import tempfile
import os


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store_initialized" not in st.session_state:
        st.session_state.vector_store_initialized = False
    if "uploaded_files" not in st.session_state:
        # Populate uploaded files from input_data directory on startup
        base_dir = Path(__file__).resolve().parents[1]
        input_dir = base_dir / "input_data"
        input_dir.mkdir(parents=True, exist_ok=True)
        files = [f.name for f in input_dir.iterdir() if f.is_file()]
        st.session_state.uploaded_files = files

    # Web search toggle persisted in session state
    if "use_web_search" not in st.session_state:
        st.session_state.use_web_search = False


def display_chat_history():
    """Display all messages in the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Minimal source display: only show local content relevance and confidence
            if message.get("sources") and isinstance(message["sources"], dict):
                routing = message["sources"].get("routing", {})
                relevance = routing.get("relevance_check")
                if relevance:
                    confidence = relevance.get("confidence", "UNKNOWN")
                    score = relevance.get("score", 0)
                    emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "⚪")
                    st.caption(f"Local content relevance: {emoji} {confidence} ({score:.1f})")


def add_message(role: str, content: str, sources: List[str] = None):
    """
    Add a message to chat history.
    
    Args:
        role: 'user' or 'assistant'
        content: Message content
        sources: Optional list of source documents
    """
    message = {"role": role, "content": content}
    if sources:
        message["sources"] = sources
    st.session_state.messages.append(message)


def clear_chat_history():
    """Clear all messages from chat history."""
    st.session_state.messages = []


# def save_uploaded_file(uploaded_file) -> str:
#     """
#     Save an uploaded file to a temporary location.
    
#     Args:
#         uploaded_file: Streamlit UploadedFile object
        
#     Returns:
#         Path to the saved file
#     """
#     # Create temp directory if it doesn't exist
#     temp_dir = tempfile.mkdtemp()
#     file_path = os.path.join(temp_dir, uploaded_file.name)
    
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     return file_path

def save_uploaded_file(uploaded_file) -> str:
    """
    Save an uploaded file permanently into the repo `input_data/` folder.

    Returns the absolute path to the saved file.
    """
    base_dir = Path(__file__).resolve().parents[1]  # project root
    target_dir = base_dir / "input_data"
    target_dir.mkdir(parents=True, exist_ok=True)

    dest = target_dir / uploaded_file.name
    stem = dest.stem
    suffix = dest.suffix
    counter = 1
    while dest.exists():
        dest = target_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(dest)


def display_sidebar_info():
    """Display information in the sidebar."""
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This is a **Hybrid RAG Chatbot** that can:
        - Answer questions from your documents
        - Search the web dynamically
        - Provide citation-aware answers
        
        **Smart Routing:**
        - 📄 Document queries → Local search only
        - 🌐 Current events → Web search only  
        - 🔀 Complex queries → Hybrid search
        """)
        
        st.divider()

        # Show upload status
        st.header("📁 Uploaded Files")
        if st.session_state.uploaded_files:
            for file in st.session_state.uploaded_files:
                st.write(f"✅ {file}")
        else:
            st.write("No files uploaded yet")

        st.divider()

        # Web search toggle (fixed in sidebar)
        st.checkbox(
            "🌐 Enable Web Search",
            value=st.session_state.get("use_web_search", False),
            key="use_web_search",
            help="When enabled, the chatbot will also search the web for answers"
        )

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            clear_chat_history()
            st.rerun()


def display_file_uploader():
    """Display file upload widget and return uploaded files."""
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload documents to chat with"
    )
    return uploaded_files


def display_processing_status(message: str, status: str = "info"):
    """
    Display a status message.
    
    Args:
        message: Status message
        status: Type - 'info', 'success', 'warning', 'error'
    """
    if status == "success":
        st.success(message)
    elif status == "warning":
        st.warning(message)
    elif status == "error":
        st.error(message)
    else:
        st.info(message)


def create_web_search_toggle() -> bool:
    """Create a toggle for web search."""
    return st.toggle(
        "🌐 Enable Web Search",
        value=False,
        help="When enabled, the chatbot will also search the web for answers"
    )
