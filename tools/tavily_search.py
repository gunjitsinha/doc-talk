"""Tavily web-search integration removed.

This module used to provide web search via Tavily. The project is configured
to be local-only; importing this module will raise an informative error to
prevent accidental use of web search functionality.
"""

def __getattr__(name):
    raise ImportError(
        "Tavily web-search integration has been removed. "
        "This project runs as a local-only chatbot."
    )
