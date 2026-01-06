"""
Helper utilities for LangGraph validator nodes.
"""

from typing import Any


def extract_text_from_response(response: Any) -> str:
    """Extract text from LLM response."""
    if isinstance(response, dict):
        return response.get("content", response.get("text", response.get("message", "")))
    return str(response) if response else ""

