from raghilda.chunk import Chunk, MarkdownChunk, RetrievedChunk, Metric
from raghilda._duckdb_store import (
    RetrievedDuckDBMarkdownChunk,
)  # internal implementation
from raghilda.document import (
    ChunkedDocument,
    ChunkedMarkdownDocument,
    Document,
    MarkdownDocument,
)


def test_retrieved_chunk():
    metrics = [Metric(name="similarity", value=0.95)]
    chunk = RetrievedDuckDBMarkdownChunk(
        text="Sample content", metrics=metrics, start_index=1, end_index=10
    )

    assert isinstance(chunk, RetrievedDuckDBMarkdownChunk)
    assert isinstance(chunk, RetrievedChunk)
    assert isinstance(chunk, MarkdownChunk)
    assert isinstance(chunk, Chunk)


def test_document_to_chunked_returns_chunked_document():
    doc = Document(content="hello world", origin="doc")
    chunks = [Chunk(text="hello", start_index=0, end_index=5, char_count=5)]

    chunked = doc.to_chunked(chunks)

    assert isinstance(chunked, ChunkedDocument)
    assert chunked.content == doc.content
    assert chunked.origin == doc.origin
    assert chunked.chunks == chunks


def test_markdown_document_to_chunked_returns_chunked_markdown_document():
    doc = MarkdownDocument(content="# Hello", origin="doc.md")
    chunks = [
        MarkdownChunk(
            text="# Hello",
            start_index=0,
            end_index=7,
            char_count=7,
        )
    ]

    chunked = doc.to_chunked(chunks)

    assert isinstance(chunked, ChunkedMarkdownDocument)
    assert list(chunked) == chunks
    assert len(chunked) == 1
    assert chunked[0] is chunks[0]
