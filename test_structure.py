#!/usr/bin/env python3
"""Test script to verify new structure"""

print("🔍 Testing new modular structure...\n")

try:
    from src.core.config import Config
    print("✅ Core config imported")
    
    from src.core.database import get_chromadb_client, check_collection_exists
    print("✅ Core database imported")
    
    from src.embeddings.models import get_google_embedding_model
    print("✅ Embeddings module imported")
    
    from src.chunking import chunk_documents
    print("✅ Chunking module imported")
    
    from src.ingestion.loader import load_documents
    from src.ingestion.pipeline import run_ingestion
    print("✅ Ingestion module imported")
    
    from src.retrieval.vectorstore import get_vectorstore, create_vectorstore
    from src.retrieval.search import run_retrieval
    print("✅ Retrieval module imported")
    
    from src.generation.llm import get_llm
    from src.generation.rag import run_generation
    print("✅ Generation module imported")
    
    from src.utils.display import display_search_results, display_rag_answer
    from src.utils.file_utils import save_last_chunking_method
    print("✅ Utils module imported")
    
    print("\n" + "="*60)
    print("🎉 All modules imported successfully!")
    print("="*60)
    print(f"\n📋 Configuration:")
    print(f"   • Default collection: {Config.DEFAULT_COLLECTION}")
    print(f"   • Docs directory: {Config.DOCS_DIRECTORY}")
    print(f"   • Default chunking: {Config.DEFAULT_CHUNKING_METHOD}")
    print(f"   • Default LLM: {Config.DEFAULT_LLM_MODEL}")
    print("\n✅ Your RAG pipeline is ready to use!")
    print("   Run: python3 main.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
