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



def display_chat_history():
    """Display all messages in the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Minimal source display: show source info only (relevance removed)
            if message.get("sources") and isinstance(message["sources"], dict):
                routing = message["sources"].get("routing", {})
                # Previously showed a relevance score here; removed for local-only mode


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
        This is a **Local RAG Chatbot** that can:
        - Answer questions from your documents
        - Provide citation-aware answers
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
    """Web search toggle removed in local-only mode."""
    return False
