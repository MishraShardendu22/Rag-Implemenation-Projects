"""
Semantic chunking strategy - groups text by meaning
"""

from langchain_experimental.text_splitter import SemanticChunker
from src.embeddings.models import LightweightEmbeddings


def get_semantic_splitter(breakpoint_threshold_type="percentile", breakpoint_threshold=70):
    """
    Create a semantic chunker that groups text by meaning
    
    Args:
        breakpoint_threshold_type: "percentile" or "standard_deviation"
        breakpoint_threshold: Threshold value
    
    Returns:
        SemanticChunker: Configured semantic splitter
    """
    embeddings = LightweightEmbeddings(model="embeddinggemma-300m", batch_size=16)
    
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold
    )


def chunk_documents_semantic(documents, breakpoint_threshold_type="percentile", breakpoint_threshold=70):
    """
    Chunk documents using semantic splitting
    
    Args:
        documents: List of LangChain Document objects
        breakpoint_threshold_type: "percentile" or "standard_deviation"
        breakpoint_threshold: Threshold value
    
    Returns:
        list: List of chunked Document objects
    """
    splitter = get_semantic_splitter(breakpoint_threshold_type, breakpoint_threshold)
    chunks = splitter.split_documents(documents)
    print(f"📄 Semantic chunking: {len(chunks)} chunks (threshold: {breakpoint_threshold_type}={breakpoint_threshold})")
    return chunks
