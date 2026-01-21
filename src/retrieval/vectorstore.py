"""
Vector store operations
"""

from langchain_chroma import Chroma
from src.core.database import get_chromadb_client
from src.embeddings.models import get_embedding_model


def create_vectorstore(chunks, collection_name):
    """
    Create a new Chroma vector store from documents
    
    Args:
        chunks: List of LangChain Document objects
        collection_name: Name of the collection
    
    Returns:
        Chroma: The created vector store
    """
    client = get_chromadb_client()
    embedding_model = get_embedding_model()
    
    # Delete collection if exists
    try:
        client.delete_collection(name=collection_name)
        print(f"🗑️  Deleted existing collection '{collection_name}'")
    except Exception:
        pass
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=client,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return vectorstore


def get_vectorstore(collection_name):
    """
    Load an existing Chroma vector store
    
    Args:
        collection_name: Name of the collection
    
    Returns:
        Chroma: The loaded vector store
    """
    client = get_chromadb_client()
    embedding_model = get_embedding_model()
    
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return vectorstore
