"""
Complete ingestion pipeline
"""

from src.core.config import Config
from src.ingestion.loader import load_documents
from src.chunking import chunk_documents
from src.retrieval.vectorstore import create_vectorstore
from src.utils.file_utils import save_last_chunking_method


def run_ingestion(docs_dir=None, collection_name=None, chunking_method=None):
    """
    Run the complete ingestion pipeline
    
    Args:
        docs_dir: Directory containing documents to ingest
        collection_name: Name of the ChromaDB Cloud collection
        chunking_method: Chunking strategy - "semantic" or "agentic"
    
    Returns:
        Chroma: The vector store with ingested documents
    """
    # Use defaults from config if not provided
    docs_dir = docs_dir or Config.DOCS_DIRECTORY
    collection_name = collection_name or Config.DEFAULT_COLLECTION
    chunking_method = chunking_method or Config.DEFAULT_CHUNKING_METHOD
    
    print(f"=== Starting Document Ingestion (LangChain + ChromaDB Cloud) ===")
    print(f"📊 Chunking method: {chunking_method.upper()}\n")
    
    # Step 1: Load documents
    docs = load_documents(docs_dir)
    
    if not docs:
        print("\n❌ No documents loaded. Please add .txt files to the docs/ directory.")
        return None
    
    # Step 2: Split documents into chunks
    chunks = chunk_documents(docs, method=chunking_method)
    
    # Step 3: Create vector store
    print("\n🔄 Creating vector store on ChromaDB Cloud...")
    vectorstore = create_vectorstore(chunks, collection_name)
    
    # Save the chunking method used
    save_last_chunking_method(chunking_method)
    print(f"💾 Saved chunking method: {chunking_method}")
    
    doc_count = vectorstore._collection.count()
    print(f"\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    print(f"📊 ChromaDB Cloud collection contains {doc_count} documents")
    
    return vectorstore
