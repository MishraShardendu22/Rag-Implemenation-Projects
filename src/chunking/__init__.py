"""
Chunking strategies for document splitting
"""

from src.chunking.semantic import chunk_documents_semantic
from src.chunking.agentic import chunk_documents_agentic


def chunk_documents(documents, method="semantic", **kwargs):
    """
    Unified chunking function - choose your chunking strategy
    
    Args:
        documents: List of LangChain Document objects
        method: Chunking method - "semantic" or "agentic"
        **kwargs: Additional arguments for the chosen method
    
    Returns:
        list: List of chunked Document objects
    """
    if method in ["character", "semantic"]:
        # Character-based is deprecated, redirect to semantic
        if method == "character":
            print("ℹ️  Character-based chunking deprecated, using semantic chunking")
        return chunk_documents_semantic(documents, **kwargs)
    
    elif method == "agentic":
        return chunk_documents_agentic(documents, **kwargs)
    
    else:
        raise ValueError(f"Unknown chunking method: {method}. Choose 'semantic' or 'agentic'")


__all__ = [
    "chunk_documents",
    "chunk_documents_semantic",
    "chunk_documents_agentic",
]
