"""
Configuration Module - Centralized Settings Management
=======================================================
DAY 1: This module handles all configuration and environment variables.

SOLID Principle: Single Responsibility Principle (SRP)
- This class has ONE job: manage configuration settings

Topics to teach:
- Environment variables and security
- Pydantic-style configuration
- Singleton pattern for settings
"""

import os
import streamlit as st
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Optional

# Fix tokenizers parallelism warning (must be set before importing transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file (for local development)
load_dotenv()


def get_secret(key: str, default: Optional[str] = None) -> str:
    """
    Get a secret from multiple sources in priority order:
    1. Streamlit secrets (st.secrets) - Used in Streamlit Cloud
    2. Environment variables - Used in Docker, local with .env, etc.
    3. Default value if provided
    
    This ensures compatibility with:
    - Local development: .env file or .streamlit/secrets.toml
    - Streamlit Cloud: Dashboard secrets
    - Docker/Other platforms: Environment variables
    
    Args:
        key: The secret key name
        default: Default value if key not found (default: empty string)
    
    Returns:
        The secret value or default
    """
    # Priority 1: Try Streamlit secrets (Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except (ImportError, AttributeError):
        pass
    
    # Priority 2: Try environment variables (.env or platform-specific)
    env_value = os.getenv(key)
    if env_value is not None:
        return env_value
    
    # Priority 3: Use default
    return default if default is not None else ""


@dataclass
class Settings:
    """
    Centralized configuration settings for the RAG application.
    
    All API keys and configuration values are loaded from environment variables
    for security. Never hardcode API keys in your code!
    
    Supports multiple deployment scenarios:
    - Local: .env file or .streamlit/secrets.toml
    - Streamlit Cloud: Dashboard secrets
    - Docker/Other: Environment variables
    """
    
    # API Keys (loaded from Streamlit secrets or environment)
    # Uses get_secret() for dual compatibility
    GROQ_API_KEY: str = get_secret("GROQ_API_KEY", "")
    
    # LLM Configuration
    LLM_MODEL: str = "llama-3.1-8b-instant"  # Free Groq model
    LLM_TEMPERATURE: float = 0.0
    
    # Embedding Configuration  
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Free HuggingFace model
    
    # Document Processing Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Vector Store Configuration
    FAISS_INDEX_PATH: str = "data/faiss_index"
    
    # Retrieval Configuration
    TOP_K_RESULTS: int = 3
    
    def validate(self) -> bool:
        """Check if required API keys are set."""
        if not self.GROQ_API_KEY:
            raise ValueError(
                "❌ GROQ_API_KEY is not set!\n"
                "1. Get it from: https://console.groq.com/\n"
                "2. Add to .env: GROQ_API_KEY=your_actual_key\n"
                "3. Do NOT use quotes in .env file\n"
                "4. Restart streamlit"
            )
        
        # Check if keys look valid (should be longer than just the prefix)
        if self.GROQ_API_KEY.startswith("your_") or len(self.GROQ_API_KEY) < 10:
            raise ValueError(
                "❌ GROQ_API_KEY appears to be a placeholder!\n"
                "Please replace with your actual key from https://console.groq.com/"
            )
        # No web-search API required for local-only deployment
        
        return True


# Create a singleton instance
settings = Settings()
