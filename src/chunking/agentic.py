"""
Agentic chunking strategy - uses LLM to decide split points
"""

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from src.core.config import Config


def get_agentic_chunker(model=None, chunk_size_guideline=200):
    """
    Create an agentic chunker that uses LLM to decide boundaries
    
    Args:
        model: OpenRouter model to use (defaults to config)
        chunk_size_guideline: Target chunk size guideline
    
    Returns:
        function: Agentic chunking function
    """
    if model is None:
        model = Config.DEFAULT_LLM_MODEL
    
    llm = ChatOpenAI(
        model=model,
        openai_api_key=Config.OPEN_ROUTER_API,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "RAG Pipeline - Agentic Chunking"
        }
    )
    
    def agentic_chunk(text):
        """Split text using LLM to determine boundaries"""
        prompt = f"""
You are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around {chunk_size_guideline} characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{text}

Return the text with <<<SPLIT>>> markers where you want to split:
"""
        
        response = llm.invoke(prompt)
        marked_text = response.content
        
        # Split and clean chunks
        chunks = [chunk.strip() for chunk in marked_text.split("<<<SPLIT>>>") if chunk.strip()]
        return chunks
    
    return agentic_chunk


def chunk_documents_agentic(documents, model=None, chunk_size_guideline=200):
    """
    Chunk documents using agentic splitting
    
    Args:
        documents: List of LangChain Document objects
        model: OpenRouter model to use
        chunk_size_guideline: Target chunk size guideline
    
    Returns:
        list: List of chunked Document objects
    """
    agentic_chunker = get_agentic_chunker(model, chunk_size_guideline)
    
    all_chunks = []
    for doc in documents:
        print(f"🤖 Agentic chunking: {doc.metadata.get('source', 'Unknown')}...")
        text_chunks = agentic_chunker(doc.page_content)
        
        for chunk_text in text_chunks:
            chunk_doc = Document(
                page_content=chunk_text,
                metadata=doc.metadata.copy()
            )
            all_chunks.append(chunk_doc)
    
    print(f"📄 Agentic chunking: {len(all_chunks)} chunks (guideline: ~{chunk_size_guideline} chars)")
    return all_chunks
