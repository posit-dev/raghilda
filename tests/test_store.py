import os
import time
import pytest
from ragnar.store import DuckDBStore, OpenAIStore
from ragnar.document import (
    ChunkedDocument,
    MarkdownDocument,
    MarkdownChunk,
    RetrievedMarkdownChunk,
    RetrievedChunk,
)
from ragnar import EmbeddingOpenAI


class TestDuckDBStore:
    @pytest.fixture
    def embed(self, request):
        try:
            return request.param
        except AttributeError:
            return None

    @pytest.fixture
    def store(self, embed):
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="test_db",
            title="Test DuckDB Store",
        )
        return store

    @pytest.fixture
    def store_with_docs(self, store):
        doc = MarkdownDocument(origin="test", content="This is a test document.")
        chunked_doc = ChunkedDocument(
            document=doc,
            chunks=[
                _get_markdown_chunk(doc, chunk_id=1, start=0, end=4),
                _get_markdown_chunk(doc, chunk_id=2, start=5, end=7),
                _get_markdown_chunk(doc, chunk_id=3, start=8, end=9),
                _get_markdown_chunk(doc, chunk_id=4, start=10, end=14),
                _get_markdown_chunk(doc, chunk_id=5, start=15, end=23),
            ],
        )
        store.insert(chunked_doc)
        return store

    def test_create_store(self, store):
        assert isinstance(store, DuckDBStore)
        assert store.metadata.name == "test_db"
        assert store.metadata.title == "Test DuckDB Store"
        assert store.metadata.embed is None

    @pytest.mark.parametrize("embed", [None, EmbeddingOpenAI()], indirect=True)
    def test_insert(self, store_with_docs):
        assert store_with_docs.size() == 1

    @pytest.mark.parametrize("embed", [EmbeddingOpenAI()], indirect=True)
    def test_retrieve_vss(self, store_with_docs):
        results = store_with_docs.retrieve_vss("test", top_k=3)
        assert len(results) == 3
        for chunk in results:
            assert isinstance(chunk, RetrievedMarkdownChunk)
            assert chunk.content is not None

        results = store_with_docs.retrieve_vss("test", top_k=5)
        assert len(results) == 5

    @pytest.mark.parametrize("embed", [None, EmbeddingOpenAI()], indirect=True)
    def test_retrieve_bm25(self, store_with_docs):
        store_with_docs.build_index("bm25")
        results = store_with_docs.retrieve_bm25("document", top_k=3)
        assert len(results) == 3
        for chunk in results:
            assert isinstance(chunk, RetrievedMarkdownChunk)
            assert chunk.content is not None

    @pytest.mark.parametrize("embed", [EmbeddingOpenAI()], indirect=True)
    def test_retrieve(self, store_with_docs):
        store_with_docs.build_index()
        results = store_with_docs.retrieve("document", top_k=3, deoverlap=False)
        assert len(results) > 3
        for chunk in results:
            assert isinstance(chunk, RetrievedMarkdownChunk)
            assert chunk.content is not None


class TestOpenAIStore:
    @pytest.fixture(autouse=True)
    def setup(self):
        if "OPENAI_API_KEY" not in os.environ:
            pytest.skip("OPENAI_API_KEY not set in environment variables")

    @pytest.fixture
    def store(self):
        store = OpenAIStore.create()
        return store

    @pytest.fixture
    def store_with_docs(self, store):
        doc = MarkdownDocument(
            origin="test", content="hello world this is a document world world world"
        )
        store.insert(doc)
        return store

    def test_create_store(self, store):
        assert isinstance(store, OpenAIStore)
        assert isinstance(store.store_id, str)

    def test_insert(self, store_with_docs):
        assert store_with_docs.size() == 1

    def test_retrieve(self, store_with_docs):
        for _ in range(5):
            if store_with_docs.size() > 0:
                break
            else:
                # wait for a bit and retry
                time.sleep(0.2)
        results = store_with_docs.retrieve("world", top_k=3)
        assert len(results) == 1
        for chunk in results:
            assert isinstance(chunk, RetrievedChunk)
            assert chunk.content is not None


def _get_markdown_chunk(doc, chunk_id, start, end):
    return MarkdownChunk(
        chunk_id=chunk_id,
        start=start,
        end=end,
        content=doc.content[start:end],
    )
