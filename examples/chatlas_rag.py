"""
Example: Integrating raghilda RAG with chatlas

This example demonstrates how to:
1. Build a RAG index from the chatlas documentation
2. Register a search tool with chatlas
3. Use the chat to answer questions about chatlas using RAG

Requirements:
    pip install chatlas raghilda

Usage:
    python examples/chatlas_rag.py
"""

from pathlib import Path

from dotenv import load_dotenv

from raghilda.store import DuckDBStore
from raghilda.embedding import EmbeddingOpenAI
from raghilda.scrape import find_links

load_dotenv()

# Path to store the RAG database
DB_PATH = Path(__file__).parent / "chatlas_docs.db"


def build_rag_index():
    """Build a RAG index from the chatlas documentation."""
    print("Finding links from chatlas documentation...")
    links = find_links(
        "https://posit-dev.github.io/chatlas/",
        depth=1,
        children_only=True,
        validate=True,
        progress=True,
    )
    # Filter out image and non-document files
    excluded_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp')
    links = [link for link in links if not link.lower().endswith(excluded_extensions)]
    print(f"Found {len(links)} pages to index.")

    print("Creating RAG store and ingesting documents...")
    store = DuckDBStore.create(
        location=str(DB_PATH),
        embed=EmbeddingOpenAI(),
        overwrite=True,
        name="chatlas_docs",
        title="Chatlas Documentation",
    )
    store.ingest(links, progress=True)

    # Build indexes for faster retrieval
    print("Building search indexes...")
    store.build_index()

    print(f"RAG index built successfully! Stored at: {DB_PATH}")
    print(f"Total documents: {store.size()}")
    return store


def create_chat_with_rag():
    """Create a chatlas chat with RAG tool registered."""
    from chatlas import ChatOpenAI  # type: ignore[reportMissingImports]

    # Connect to existing store or build if it doesn't exist
    if DB_PATH.exists():
        print(f"Connecting to existing RAG store at {DB_PATH}")
        store = DuckDBStore.connect(str(DB_PATH), read_only=True)
    else:
        print("RAG store not found. Building index first...")
        store = build_rag_index()

    # Define the RAG search tool
    def search_chatlas_docs(query: str, num_results: int = 5) -> str:
        """
        Search the chatlas documentation for relevant information.

        Use this tool when the user asks questions about chatlas, such as:
        - How to create a chat
        - How to use different models
        - How to register tools
        - How to stream responses
        - Any other chatlas-related questions

        Args:
            query: The search query describing what information to find.
            num_results: Number of relevant chunks to return (default: 5).

        Returns:
            Relevant excerpts from the chatlas documentation.
        """
        chunks = store.retrieve(query, top_k=num_results, deoverlap=True)

        if not chunks:
            return "No relevant documentation found."

        results = []
        for i, chunk in enumerate(chunks, 1):
            context = f" (from: {chunk.context})" if chunk.context else ""
            results.append(f"[{i}]{context}\n{chunk.text}")

        return "\n\n---\n\n".join(results)

    # Create the chat with system prompt
    chat = ChatOpenAI(
        model="gpt-4o-mini",
        system_prompt="""You are a helpful assistant that answers questions about the chatlas Python library.

You have access to a search tool that can find relevant information from the chatlas documentation.
Always use the search tool when answering questions about chatlas to ensure accurate information.

When providing code examples, make sure they are correct and follow chatlas best practices.
If you're not sure about something, say so rather than making things up.""",
    )

    # Register the RAG tool
    chat.register_tool(search_chatlas_docs)

    # Show tool requests but not results
    def handle_tool_request(request):
        print(f"🔍 Searching: {request.arguments.get('query', '')}")
        return True  # Allow the tool call to proceed

    chat.on_tool_request(handle_tool_request)

    return chat


def main():
    import sys

    # Check if we need to rebuild the index
    if "--rebuild" in sys.argv:
        build_rag_index()

    # Create the chat
    chat = create_chat_with_rag()

    print("\n" + "=" * 60)
    print("Chatlas RAG Assistant")
    print("=" * 60)
    print("Ask questions about the chatlas library.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=" * 60 + "\n")

    # Interactive chat loop
    while True:
        try:
            user_input = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        chat.chat(user_input, echo="text")


if __name__ == "__main__":
    main()
