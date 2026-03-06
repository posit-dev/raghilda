"""Tests for compatibility with chonkie types and chunkers."""
# ruff: noqa: E402

import pytest

# Skip all tests in this module if chonkie is not installed
chonkie = pytest.importorskip("chonkie")

from chonkie.types import (
    Chunk as ChonkieChunk,
    Document as ChonkieDocument,
)  # noqa: E402
from raghilda.chunk import Chunk  # noqa: E402
from raghilda.document import Document, MarkdownDocument  # noqa: E402
from raghilda.types import ChunkLike, DocumentLike  # noqa: E402
from raghilda.store import DuckDBStore  # noqa: E402

TokenChunker = getattr(chonkie, "TokenChunker")


class TestChonkieChunkCompatibility:
    def test_chonkie_chunk_satisfies_chunk_like(self):
        """Chonkie Chunk should satisfy the chunk-like protocol."""
        chunk = ChonkieChunk(
            text="hello world",
            start_index=0,
            end_index=11,
            token_count=2,
        )
        assert isinstance(chunk, ChunkLike)

    def test_from_any_converts_chonkie_chunk(self):
        """Chunk.from_any should convert a chonkie Chunk."""
        chonkie_chunk = ChonkieChunk(
            text="hello world",
            start_index=0,
            end_index=11,
            token_count=2,
        )
        result = Chunk.from_any(chonkie_chunk)

        assert isinstance(result, Chunk)
        assert result.text == "hello world"
        assert result.start_index == 0
        assert result.end_index == 11
        assert result.char_count == 11

    def test_from_any_preserves_chonkie_chunk_context(self):
        """Chunk.from_any should preserve context from chonkie Chunk."""
        chonkie_chunk = ChonkieChunk(
            text="hello world",
            start_index=0,
            end_index=11,
            token_count=2,
            context="# Header",
        )
        result = Chunk.from_any(chonkie_chunk)
        assert result.context == "# Header"


class TestChonkieDocumentCompatibility:
    def test_chonkie_document_satisfies_document_like(self):
        """Chonkie Document should satisfy our DocumentLike protocol."""
        doc = ChonkieDocument(content="hello world")
        assert isinstance(doc, DocumentLike)

    def test_from_any_converts_chonkie_document(self):
        """Document.from_any should convert a chonkie Document."""
        chonkie_doc = ChonkieDocument(content="hello world")
        result = Document.from_any(chonkie_doc)  # type: ignore[arg-type]

        assert isinstance(result, Document)
        assert result.content == "hello world"

    def test_from_any_converts_chonkie_document_with_chunks(self):
        """Document.from_any should convert a chonkie Document with chunks."""
        chonkie_doc = ChonkieDocument(content="hello world")
        chonkie_doc.chunks = [
            ChonkieChunk(text="hello", start_index=0, end_index=5, token_count=1),
            ChonkieChunk(text="world", start_index=6, end_index=11, token_count=1),
        ]
        result = Document.from_any(chonkie_doc)  # type: ignore[arg-type]

        assert isinstance(result, Document)
        assert result.chunks is not None
        assert len(result.chunks) == 2
        assert all(isinstance(c, Chunk) for c in result.chunks)
        assert result.chunks[0].text == "hello"
        assert result.chunks[1].text == "world"


class TestChonkieChunkerWithStore:
    def test_store_accepts_chonkie_chunked_document(self):
        """DuckDBStore should accept documents chunked by chonkie."""

        # Create a store without embeddings
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            name="test_db",
            title="Test Store",
        )

        # Use chonkie's TokenChunker
        chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
        text = (
            "This is a test document. It has multiple sentences. We want to chunk it."
        )

        # Chunk the text using chonkie
        chonkie_chunks = chunker.chunk(text)

        # Create a MarkdownDocument with chonkie chunks converted
        doc = MarkdownDocument(
            content=text,
            origin="test://chonkie",
            chunks=[Chunk.from_any(c) for c in chonkie_chunks],
        )

        # Insert into store
        store.upsert(doc)

        assert store.size() == 1

    def test_store_ingest_with_chonkie_prepare_function(self):
        """DuckDBStore.ingest should work with a chonkie-based prepare function."""
        import tempfile
        import os

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test Document\n\nThis is test content for chonkie chunking.")
            temp_path = f.name

        try:
            store = DuckDBStore.create(
                location=":memory:",
                embed=None,
                name="test_db",
                title="Test Store",
            )

            chunker = TokenChunker(chunk_size=50, chunk_overlap=10)

            def prepare_with_chonkie(uri: str) -> MarkdownDocument:
                with open(uri) as f:
                    content = f.read()
                chonkie_chunks = chunker.chunk(content)
                return MarkdownDocument(
                    content=content,
                    origin=uri,
                    chunks=[Chunk.from_any(c) for c in chonkie_chunks],
                )

            store.ingest([temp_path], prepare=prepare_with_chonkie)

            assert store.size() == 1
        finally:
            os.unlink(temp_path)
