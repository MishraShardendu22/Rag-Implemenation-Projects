"""
Document search and retrieval
"""

from src.core.config import Config
from src.retrieval.vectorstore import get_vectorstore
from src.utils.display import display_search_results


def get_retriever(vectorstore, k=None, search_type="similarity"):
    """
    Create a LangChain retriever from the vector store
    
    Args:
        vectorstore: Chroma vector store
        k: Number of documents to retrieve
        search_type: Type of search
    
    Returns:
        Retriever: LangChain retriever object
    """
    k = k or Config.DEFAULT_TOP_K
    
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )
    return retriever


def search_documents(vectorstore, query, k=None):
    """
    Search for relevant documents based on a query
    
    Args:
        vectorstore: Chroma vector store
        query: Search query string
        k: Number of results to return
    
    Returns:
        list: List of (Document, score) tuples
    """
    k = k or Config.DEFAULT_TOP_K
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def format_search_results(results):
    """
    Format search results for display
    
    Args:
        results: List of (Document, score) tuples
    
    Returns:
        list: Formatted result dictionaries
    """
    formatted = []
    
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "Unknown")
        similarity = round(1 - score, 4) if score <= 1 else round(score, 4)
        
        formatted.append({
            "rank": i + 1,
            "content": doc.page_content,
            "source": source,
            "metadata": doc.metadata,
            "similarity_score": similarity
        })
    
    return formatted


def run_retrieval(vectorstore=None, query=None, k=None, interactive=False, collection_name=None):
    """
    Run retrieval - either interactive or single query
    
    Args:
        vectorstore: Optional Chroma vector store
        query: Optional query string
        k: Number of results to return
        interactive: Whether to run interactive mode
        collection_name: Name of the collection
    
    Returns:
        list or None: Formatted results if single query, None if interactive
    """
    collection_name = collection_name or Config.DEFAULT_COLLECTION
    k = k or Config.DEFAULT_TOP_K
    
    print("=== Document Retrieval (LangChain + ChromaDB Cloud) ===\n")
    
    # Load vectorstore if not provided
    if vectorstore is None:
        print("🔄 Loading vector store from ChromaDB Cloud...")
        vectorstore = get_vectorstore(collection_name)
    
    doc_count = vectorstore._collection.count()
    print(f"📊 ChromaDB Cloud collection contains {doc_count} documents\n")
    
    if interactive:
        interactive_search(vectorstore, k)
        return None
    
    if query:
        print(f"🔍 Searching for: '{query}'\n")
        results = search_documents(vectorstore, query, k)
        formatted = format_search_results(results)
        display_search_results(formatted)
        return formatted
    
    print("⚠️  No query provided. Use interactive=True for interactive mode.")
    return None


def interactive_search(vectorstore, k=None):
    """
    Run an interactive search session
    
    Args:
        vectorstore: Chroma vector store
        k: Number of results to return
    """
    k = k or Config.DEFAULT_TOP_K
    
    print("\n" + "="*60)
    print("🔍 Interactive Document Search")
    print("="*60)
    print("Type your query and press Enter to search.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            query = input("🔎 Enter query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                print("⚠️  Please enter a valid query.\n")
                continue
            
            results = search_documents(vectorstore, query, k)
            formatted = format_search_results(results)
            display_search_results(formatted)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
