"""
Embedding models for RAG Pipeline
"""

from typing import List
import requests
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from src.core.config import Config


def get_google_embedding_model():
    """
    Get the Google Gemini embedding model
    
    Returns:
        GoogleGenerativeAIEmbeddings: Google embedding model
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=Config.GEMINI_API_KEY
    )


def get_openai_embedding_model():
    """
    Get the OpenAI embedding model
    
    Returns:
        OpenAIEmbeddings: OpenAI embedding model
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=Config.OPENAI_API_KEY
    )


def get_embedding_model():
    """
    Get the configured embedding model (OpenAI or Google)
    
    Returns:
        Embeddings: The configured embedding model
    """
    if Config.EMBEDDING_PROVIDER == "google":
        print("🔵 Using Google Gemini embeddings")
        return get_google_embedding_model()
    else:
        print("🟢 Using OpenAI embeddings")
        return get_openai_embedding_model()


class LightweightEmbeddings(Embeddings):
    """
    Free embedding API - OpenAI compatible, no API key required.
    Used for semantic chunking.
    """
    
    def __init__(self, model: str = "embeddinggemma-300m", batch_size: int = 16):
        self.model = model
        self.api_url = "https://lamhieu-lightweight-embeddings.hf.space/v1/embeddings"
        self.batch_size = batch_size
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        all_embeddings = []
        total_batches = (len(texts) - 1) // self.batch_size + 1
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            print(f"  🔄 Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)...", flush=True)
            
            response = requests.post(
                self.api_url,
                json={"input": batch, "model": self.model},
                timeout=300
            )
            response.raise_for_status()
            data = response.json()
            all_embeddings.extend([item["embedding"] for item in data["data"]])
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        response = requests.post(
            self.api_url,
            json={"input": [text], "model": self.model},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
