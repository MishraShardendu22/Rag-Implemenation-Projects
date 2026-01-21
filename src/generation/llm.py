"""
LLM setup for answer generation
"""

from langchain_openai import ChatOpenAI
from src.core.config import Config


def get_llm(model=None, temperature=None):
    """
    Get the ChatOpenAI model configured for OpenRouter
    
    Args:
        model: OpenRouter model to use
        temperature: Temperature for generation
    
    Returns:
        ChatOpenAI: LangChain chat model
    """
    model = model or Config.DEFAULT_LLM_MODEL
    temperature = temperature if temperature is not None else Config.DEFAULT_TEMPERATURE
    
    llm = ChatOpenAI(
        model=model,
        openai_api_key=Config.OPEN_ROUTER_API,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=1024,
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "RAG Pipeline"
        }
    )
    
    return llm
