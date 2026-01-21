"""
Configuration management for RAG Pipeline
Loads and validates environment variables
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class for RAG pipeline"""
    
    # ChromaDB Cloud Configuration
    CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "rag-learn")
    CHROMA_TENANT = os.getenv("CHROMA_TENANT")
    CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
    
    # Google Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # OpenAI API
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # OpenRouter API
    OPEN_ROUTER_API = os.getenv("OPEN_ROUTER_API")
    
    # Embedding settings
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # "openai" or "google"
    
    # Default settings
    DEFAULT_COLLECTION = "rag-documents"
    DOCS_DIRECTORY = "docs"
    
    # Chunking defaults
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    DEFAULT_CHUNKING_METHOD = "semantic"
    
    # Retrieval defaults
    DEFAULT_TOP_K = 5
    
    # Generation defaults
    DEFAULT_LLM_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
    DEFAULT_TEMPERATURE = 0.7
    
    @classmethod
    def validate(cls):
        """Validate required environment variables"""
        required = {
            "CHROMA_TENANT": cls.CHROMA_TENANT,
            "CHROMA_API_KEY": cls.CHROMA_API_KEY,
            "OPEN_ROUTER_API": cls.OPEN_ROUTER_API,
        }
        
        # Validate embedding provider
        if cls.EMBEDDING_PROVIDER == "google" and not cls.GEMINI_API_KEY:
            required["GEMINI_API_KEY"] = cls.GEMINI_API_KEY
        elif cls.EMBEDDING_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            required["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        
        missing = [k for k, v in required.items() if not v]
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please check your .env file."
            )
        
        return True


# Validate on import
try:
    Config.validate()
except ValueError as e:
    print(f"⚠️  Configuration Error: {e}")
