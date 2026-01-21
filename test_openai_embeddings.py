#!/usr/bin/env python3
"""Test OpenAI embeddings configuration"""

from src.core.config import Config
from src.embeddings.models import get_embedding_model

print("="*60)
print("Testing OpenAI Embeddings Configuration")
print("="*60)

print(f"\n✅ Config loaded successfully")
print(f"📊 Embedding provider: {Config.EMBEDDING_PROVIDER}")
print(f"🔑 OpenAI API key: {'Set' if Config.OPENAI_API_KEY else 'Not set'}")

try:
    model = get_embedding_model()
    print(f"✅ Embedding model loaded: {type(model).__name__}")
    
    # Test embedding a single text
    print(f"\n🧪 Testing embedding...")
    test_text = "This is a test sentence."
    embedding = model.embed_query(test_text)
    print(f"✅ Successfully generated embedding")
    print(f"   Embedding dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    
    print("\n" + "="*60)
    print("✅ All tests passed! OpenAI embeddings are working.")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
