import os
import pytest
from ragnar.store import DuckDBStore, OpenAIStore
from ragnar.scrape import find_links
from ragnar.document import (
    MarkdownDocument,
    RetrievedChunk,
)
from ragnar._chunker import MarkdownChunk
from ragnar._store import RetrievedDuckDBMarkdownChunk
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
        doc.chunks = [
            _get_markdown_chunk(doc, start=0, end=4),
            _get_markdown_chunk(doc, start=5, end=7),
            _get_markdown_chunk(doc, start=8, end=9),
            _get_markdown_chunk(doc, start=10, end=14),
            _get_markdown_chunk(doc, start=15, end=23),
        ]
        store.insert(doc)
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
            assert isinstance(chunk, RetrievedDuckDBMarkdownChunk)
            assert chunk.text is not None

        results = store_with_docs.retrieve_vss("test", top_k=5)
        assert len(results) == 5

    @pytest.mark.parametrize("embed", [None, EmbeddingOpenAI()], indirect=True)
    def test_retrieve_bm25(self, store_with_docs):
        store_with_docs.build_index("bm25")
        results = store_with_docs.retrieve_bm25("document", top_k=3)
        assert len(results) == 3
        for chunk in results:
            assert isinstance(chunk, RetrievedDuckDBMarkdownChunk)
            assert chunk.text is not None

    @pytest.mark.parametrize("embed", [EmbeddingOpenAI()], indirect=True)
    def test_retrieve(self, store_with_docs):
        store_with_docs.build_index()
        results = store_with_docs.retrieve("document", top_k=3, deoverlap=False)
        assert len(results) > 3
        for chunk in results:
            assert isinstance(chunk, RetrievedDuckDBMarkdownChunk)
            assert chunk.text is not None


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
        for _ in range(20):
            store_with_docs.insert(
                MarkdownDocument(
                    origin="test", content="hello world world world world world"
                )
            )
        results = store_with_docs.retrieve("world", top_k=3)
        assert len(results) > 0
        for chunk in results:
            assert isinstance(chunk, RetrievedChunk)
            assert chunk.text is not None


def _get_markdown_chunk(doc, start, end):
    return MarkdownChunk(
        start_index=start,
        end_index=end,
        text=doc.content[start:end],
        token_count=len(doc.content[start:end]),
    )


def test_ingest():
    links = find_links("https://r4ds.hadley.nz/base-R.html", validate=True)

    store = DuckDBStore.create(
        location=":memory:",
        embed=EmbeddingOpenAI(),
        overwrite=True,
        name="ingest_db",
        title="Ingest Test DuckDB Store",
    )

    store.ingest(links)
