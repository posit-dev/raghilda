# raghilda <img src="assets/raghilda-logo.png" align="right" width="140" alt="raghilda hex logo" />

RAG made simple.

raghilda is a Python package for implementing Retrieval-Augmented Generation (RAG) workflows. It provides a complete solution with sensible defaults while remaining transparent (not a black box).

## What you can build

raghilda puts a question-answering layer over content you already have. Use it to chat over your documentation with cited sources, build internal knowledge assistants over runbooks, wikis, or policies (scoped per team or product with attribute filters), run research and Q&A across a directory of PDFs, notebooks, or reports, or power entity-scoped retrieval over customer records, product catalogs, and case timelines where every answer must stay within a single record.

## Installation

```bash
pip install raghilda
```

Or install from GitHub:

```bash
pip install git+https://github.com/posit-dev/raghilda.git
```

## Key Steps

raghilda handles the complete RAG pipeline:

1. document Processing: convert documents to Markdown using MarkItDown
2. text chunking: split text at semantic boundaries (headings, paragraphs, sentences)
3. embedding: generate vector representations via OpenAI or other providers
4. storage: store chunks and embeddings in DuckDB, ChromaDB, or OpenAI Vector Stores
5. retrieval: find relevant chunks using similarity search or BM25

Each step is an ordinary, inspectable Python object with sensible defaults, so you can run the whole pipeline as-is and customize any stage when you need to.

## Usage

```python
from raghilda.store import DuckDBStore
from raghilda.embedding import EmbeddingOpenAI
from raghilda.scrape import find_links
from raghilda.read import read_as_markdown
from raghilda.chunker import MarkdownChunker

# Create a store with embeddings
store = DuckDBStore.create(
    location="chatlas.db",
    embed=EmbeddingOpenAI(),
)

# Find and index pages from the chatlas documentation
links = find_links("https://posit-dev.github.io/chatlas/")
chunker = MarkdownChunker()

# Read, chunk, and store each page
for link in links:
    document = read_as_markdown(link)
    chunked_document = chunker.chunk(document)
    store.upsert(chunked_document)

# Build indexes before retrieval
store.build_index()

# Retrieve relevant chunks
chunks = store.retrieve("How do I stream a response?", top_k=5)
for chunk in chunks:
    print(chunk.text)
```

## Requirements

- **Python 3.11–3.13**
- An embedding provider (such as OpenAI) is optional: semantic search needs one, but keyword search works with no API key.

## Links

- [Documentation](https://posit-dev.github.io/raghilda/)
- [PyPI](https://pypi.org/project/raghilda/)
- [Report an issue](https://github.com/posit-dev/raghilda/issues)

## Contributing

Contributions are welcome! See [CONTRIBUTING](CONTRIBUTING.md) for setup and development guidelines.

## License

Released under the MIT License.

---

<p align="center">
Developed by <strong>Daniel&nbsp;Falbel</strong> and <strong>Tomasz&nbsp;Kalinowski</strong>.<br>
Supported by <a href="https://posit.co"><strong>Posit&nbsp;Software,&nbsp;PBC</strong></a>.
</p>
