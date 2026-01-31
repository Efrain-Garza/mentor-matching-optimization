"""
Parsing utilities for catalog ingestion.

CLEAN stage helpers:
- Text normalization
- Course code extraction
- Description cleanup
"""

def clean_text(text: str) -> str:
    """
    Normalize whitespace and basic formatting.
    """
    return " ".join(text.split())
