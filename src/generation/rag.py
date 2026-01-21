"""
RAG answer generation
"""

from langchain_core.messages import HumanMessage, SystemMessage
from src.core.config import Config
from src.generation.llm import get_llm
from src.retrieval.search import get_retriever
from src.retrieval.vectorstore import get_vectorstore
from src.utils.display import display_rag_answer


def generate_answer(vectorstore, query, k=None, model=None):
    """
    Generate an answer using RAG
    
    Args:
        vectorstore: Chroma vector store
        query: User's question
        k: Number of context documents to retrieve
        model: OpenRouter model to use
    
    Returns:
        dict: Contains query, context documents, and generated answer
    """
    k = k or Config.DEFAULT_TOP_K
    model = model or Config.DEFAULT_LLM_MODEL
    
    # Step 1: Retrieve relevant documents
    print(f"🔍 Searching for relevant documents...")
    retriever = get_retriever(vectorstore, k=k)
    relevant_docs = retriever.invoke(query)
    
    if not relevant_docs:
        return {
            "query": query,
            "context": [],
            "answer": "I couldn't find any relevant documents to answer your question."
        }
    
    # Step 2: Build context
    context_parts = []
    for doc in relevant_docs:
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    
    context_text = "\n\n---\n\n".join(context_parts)
    
    # Step 3: Create messages
    system_content = """You are a helpful assistant that answers questions based on the provided context documents. 

Rules:
- Only use information from the provided documents to answer
- If the answer isn't in the documents, say "I don't have enough information to answer that question based on the provided documents."
- Be concise and accurate
- Cite the source when possible"""

    user_content = f"""Based on the following documents, please answer this question: {query}

Documents:
{context_text}

Please provide a clear, helpful answer using only the information from these documents."""

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]
    
    # Step 4: Generate answer
    print(f"🤖 Generating answer using {model}...")
    llm = get_llm(model)
    result = llm.invoke(messages)
    
    # Format context info
    context_info = []
    for doc in relevant_docs:
        context_info.append({
            "source": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata
        })
    
    return {
        "query": query,
        "context": context_info,
        "answer": result.content
    }


def run_generation(vectorstore=None, query=None, interactive=False, model=None, collection_name=None):
    """
    Run answer generation - either interactive or single query
    
    Args:
        vectorstore: Optional Chroma vector store
        query: Optional query string
        interactive: Whether to run interactive mode
        model: OpenRouter model to use
        collection_name: Name of the collection
    
    Returns:
        dict or None: Result dict if single query, None if interactive
    """
    collection_name = collection_name or Config.DEFAULT_COLLECTION
    model = model or Config.DEFAULT_LLM_MODEL
    
    print("=== RAG Answer Generation (LangChain + ChromaDB Cloud) ===\n")
    
    # Load vectorstore if not provided
    if vectorstore is None:
        print("🔄 Loading vector store from ChromaDB Cloud...")
        vectorstore = get_vectorstore(collection_name)
    
    doc_count = vectorstore._collection.count()
    print(f"📊 ChromaDB Cloud collection contains {doc_count} documents")
    print(f"🤖 Using model: {model}\n")
    
    if interactive:
        interactive_qa(vectorstore, model)
        return None
    
    if query:
        result = generate_answer(vectorstore, query, model=model)
        display_rag_answer(result)
        return result
    
    print("⚠️  No query provided. Use interactive=True for interactive mode.")
    return None


def interactive_qa(vectorstore, model=None):
    """
    Run an interactive Q&A session
    
    Args:
        vectorstore: Chroma vector store
        model: OpenRouter model to use
    """
    model = model or Config.DEFAULT_LLM_MODEL
    
    print("\n" + "=" * 60)
    print("🤖 Interactive RAG Q&A")
    print("=" * 60)
    print(f"Model: {model}")
    print("Type your question and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            query = input("❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                print("⚠️  Please enter a valid question.\n")
                continue
            
            result = generate_answer(vectorstore, query, model=model)
            display_rag_answer(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
