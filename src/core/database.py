"""
ChromaDB Cloud client management
"""

import chromadb
from src.core.config import Config


def get_chromadb_client():
    """
    Get ChromaDB Cloud client
    
    Returns:
        chromadb.Client: Connected ChromaDB client
    """
    client = chromadb.CloudClient(
        database=Config.CHROMA_DATABASE,
        tenant=Config.CHROMA_TENANT,
        api_key=Config.CHROMA_API_KEY,
    )
    return client


def check_collection_exists(collection_name):
    """
    Check if a collection exists in ChromaDB Cloud
    
    Args:
        collection_name: Name of the collection to check
    
    Returns:
        bool: True if collection exists, False otherwise
    """
    try:
        client = get_chromadb_client()
        collections = client.list_collections()
        return any(c.name == collection_name for c in collections)
    except Exception as e:
        print(f"⚠️  Error connecting to ChromaDB Cloud: {e}")
        return False
