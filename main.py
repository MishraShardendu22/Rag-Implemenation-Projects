"""
RAG Implementation - Main Entry Point
A modular RAG pipeline using LangChain and ChromaDB Cloud
"""

from src.core.config import Config
from src.core.database import check_collection_exists
from src.retrieval.vectorstore import get_vectorstore
from src.ingestion.pipeline import run_ingestion
from src.retrieval.search import run_retrieval
from src.generation.rag import run_generation


def main():
    """Main entry point for the RAG pipeline"""
    print("=" * 60)
    print("🚀 RAG Document Pipeline (LangChain + ChromaDB Cloud)")
    print("=" * 60 + "\n")

    print("📡 Connecting to ChromaDB Cloud...")
    
    collection_name = Config.DEFAULT_COLLECTION
    vectorstore_exists = check_collection_exists(collection_name)
    
    if vectorstore_exists:
        print(f"✅ Connected to ChromaDB Cloud")
        
        try:
            vectorstore = get_vectorstore(collection_name)
            doc_count = vectorstore._collection.count()
            print(f"📊 Collection '{collection_name}' contains {doc_count} documents\n")
        except Exception as e:
            print(f"⚠️  Could not load collection: {e}")
            vectorstore_exists = False
    else:
        print(f"📊 Collection '{collection_name}' not found. Run ingestion first.\n")

    # Menu
    print("What would you like to do?")
    print("1. Ingest documents (character/semantic chunking)")
    print("2. Ingest documents (semantic chunking)")
    print("3. Ingest documents (agentic/LLM chunking)")
    print("4. Search documents (query the collection)")
    print("5. Interactive search mode")
    print("6. Ask a question (RAG with AI answer)")
    print("7. Interactive Q&A mode (RAG with AI)")
    print("8. Run full pipeline (ingest + Q&A)")
    print("9. Exit\n")

    try:
        choice = input("Enter choice (1-9): ").strip()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        return
    
    # Handle choices
    if choice == "1":
        run_ingestion(chunking_method="character")
    
    elif choice == "2":
        run_ingestion(chunking_method="semantic")
    
    elif choice == "3":
        run_ingestion(chunking_method="agentic")

    ###########################################################################################################################
    elif choice == "4":
        if not vectorstore_exists:
            print("❌ Collection not found on ChromaDB Cloud. Please run ingestion first (option 1, 2, or 3).")
            return
        query = input("\n🔎 Enter your search query: ").strip()
        if query:
            run_retrieval(query=query, k=5, collection_name=COLLECTION_NAME)
        else:
            print("❌ No query provided.")
    ###########################################################################################################################

    ###########################################################################################################################    
    elif choice == "5":
        if not vectorstore_exists:
            print("❌ Collection not found on ChromaDB Cloud. Please run ingestion first (option 1, 2, or 3).")
            return
        run_retrieval(interactive=True, collection_name=COLLECTION_NAME)
    ###########################################################################################################################

    ###########################################################################################################################
    elif choice == "6":
        if not vectorstore_exists:
            print("❌ Collection not found on ChromaDB Cloud. Please run ingestion first (option 1, 2, or 3).")
            return
        # Get last used chunking method
        last_method = get_last_chunking_method()
        print(f"ℹ️  Using chunking method from last ingestion: {last_method.upper()}")
        query = input("\n❓ Enter your question: ").strip()
        if query:
            run_generation(query=query, collection_name=COLLECTION_NAME)
        else:
            print("❌ No question provided.")
    ###########################################################################################################################

    
    elif choice == "4":
        if not vectorstore_exists:
            print("⚠️  No collection found. Please run ingestion first (option 1-3).")
            return
        query = input("\n🔎 Enter your search query: ").strip()
        if query:
            run_retrieval(query=query)
    
    elif choice == "5":
        if not vectorstore_exists:
            print("⚠️  No collection found. Please run ingestion first (option 1-3).")
            return
        run_retrieval(interactive=True)
    
    elif choice == "6":
        if not vectorstore_exists:
            print("⚠️  No collection found. Please run ingestion first (option 1-3).")
            return
        query = input("\n❓ Enter your question: ").strip()
        if query:
            run_generation(query=query)
    
    elif choice == "7":
        if not vectorstore_exists:
            print("⚠️  No collection found. Please run ingestion first (option 1-3).")
            return
        run_generation(interactive=True)
    
    elif choice == "8":
        vectorstore = run_ingestion()
        if vectorstore:
            print("\n" + "="*60)
            print("Now starting interactive Q&A...")
            print("="*60)
            run_generation(vectorstore=vectorstore, interactive=True)
    
    elif choice == "9":
        print("\n👋 Goodbye!")
    
    else:
        print("❌ Invalid choice. Please run again and select 1-9.")


if __name__ == "__main__":
    main()

