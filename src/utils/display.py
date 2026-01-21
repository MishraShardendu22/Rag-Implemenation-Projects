"""
Display utilities for formatting output
"""


def display_search_results(formatted_results):
    """Display formatted search results"""
    if not formatted_results:
        print("❌ No results found.")
        return
    
    print(f"\n{'='*60}")
    print(f"📋 Found {len(formatted_results)} relevant documents")
    print(f"{'='*60}\n")
    
    for result in formatted_results:
        print(f"📌 Rank #{result['rank']} | Source: {result['source']} | "
              f"Score: {result['similarity_score']}")
        print(f"{'─'*60}")
        content_preview = result['content'][:300]
        if len(result['content']) > 300:
            content_preview += "..."
        print(f"{content_preview}")
        print(f"{'─'*60}\n")


def display_rag_answer(result):
    """Display RAG answer with context"""
    print("\n" + "=" * 60)
    print("📝 RAG Answer Generation")
    print("=" * 60)
    
    print(f"\n❓ Question: {result['query']}")
    
    print(f"\n📚 Context Sources ({len(result['context'])} documents):")
    for ctx in result['context']:
        print(f"   • {ctx['source']}")
    
    print("\n" + "-" * 60)
    print("💡 Answer:")
    print("-" * 60)
    print(result['answer'])
    print("=" * 60 + "\n")
