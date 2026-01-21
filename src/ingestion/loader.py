"""
Document loading utilities
"""

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_documents(docs_dir="docs"):
    """
    Load all text documents from the specified directory
    
    Args:
        docs_dir: Directory containing text documents
    
    Returns:
        list: List of LangChain Document objects
    """
    if not os.path.exists(docs_dir):
        print(f"❌ Directory {docs_dir} not found!")
        return []
    
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    documents = loader.load()
    
    # Print loaded documents info
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        print(f"✓ Loaded: {os.path.basename(source)} ({len(doc.page_content)} chars)")
    
    print(f"\n📚 Total documents loaded: {len(documents)}")
    return documents
