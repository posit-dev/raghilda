# raghilda <img src="docs/raghilda-logo.png" align="right" width="140" alt="raghilda hex logo" />

RAG made simple.

raghilda is a Python package for implementing Retrieval-Augmented Generation (RAG) workflows. It provides a complete solution with sensible defaults while remaining transparent—not a black box.

## Installation

```bash
pip install raghilda
```

Or install from GitHub:

```bash
pip install git+https://github.com/dfalbel/py-ragnar.git
```

## Key Steps

raghilda handles the complete RAG pipeline:

1. **Document Processing** — Convert documents to Markdown using MarkItDown
2. **Text Chunking** — Split text at semantic boundaries (headings, paragraphs, sentences)
3. **Embedding** — Generate vector representations via OpenAI or other providers
4. **Storage** — Store chunks and embeddings in DuckDB or OpenAI Vector Stores
5. **Retrieval** — Find relevant chunks using similarity search or BM25

## Usage

```python
from raghilda.store import DuckDBStore
from raghilda.embedding import EmbeddingOpenAI
from raghilda.scrape import find_links

# Create a store with embeddings
store = DuckDBStore.create(
    location="chatlas.db",
    embed=EmbeddingOpenAI(),
)

# Find and ingest all pages from the chatlas documentation
links = find_links("https://posit-dev.github.io/chatlas/")
store.ingest(links)

# Retrieve relevant chunks
chunks = store.retrieve("How do I stream a response?", top_k=5)
for chunk in chunks:
    print(chunk.text)
```

## Links

- [Documentation](https://dfalbel.github.io/py-ragnar/)
- [Source Code](https://github.com/dfalbel/py-ragnar)
- [Report Issues](https://github.com/dfalbel/py-ragnar/issues)
