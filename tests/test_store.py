import os
import pytest
from raghilda.store import DuckDBStore, OpenAIStore
from raghilda.scrape import find_links
from raghilda.document import MarkdownDocument
from raghilda.chunk import MarkdownChunk, RetrievedChunk
from raghilda._store import RetrievedDuckDBMarkdownChunk  # internal implementation
from raghilda.embedding import EmbeddingOpenAI


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

    def test_retrieve_with_deoverlap(self, store):
        # Create a document with overlapping chunks
        # "hello world test document" = 24 chars
        doc = MarkdownDocument(
            origin="test_deoverlap", content="hello world test document"
        )
        doc.chunks = [
            _get_markdown_chunk(doc, start=0, end=11),  # "hello world"
            _get_markdown_chunk(doc, start=6, end=16),  # "world test"
            _get_markdown_chunk(doc, start=12, end=25),  # "test document"
        ]
        store.insert(doc)
        store.build_index("bm25")

        # Without deoverlap, we may get multiple overlapping chunks
        results_no_deoverlap = store.retrieve("test", top_k=5, deoverlap=False)

        # With deoverlap, overlapping chunks should be merged
        results_deoverlap = store.retrieve("test", top_k=5, deoverlap=True)

        # Deoverlapped results should have fewer or equal chunks
        assert len(results_deoverlap) <= len(results_no_deoverlap)

        # Check that deoverlapped chunks don't have overlapping ranges
        for i, chunk1 in enumerate(results_deoverlap):
            for chunk2 in results_deoverlap[i + 1 :]:
                if chunk1.doc_id == chunk2.doc_id:
                    # Same document - ranges should not overlap
                    assert (
                        chunk1.end_index <= chunk2.start_index
                        or chunk2.end_index <= chunk1.start_index
                    ), (
                        f"Chunks overlap: [{chunk1.start_index}, {chunk1.end_index}) and [{chunk2.start_index}, {chunk2.end_index})"
                    )


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
    from raghilda.chunker import MarkdownChunker
    from raghilda.read import read_as_markdown

    links = find_links("https://r4ds.hadley.nz/base-R.html", validate=True)

    store = DuckDBStore.create(
        location=":memory:",
        embed=EmbeddingOpenAI(),
        overwrite=True,
        name="ingest_db",
        title="Ingest Test DuckDB Store",
    )

    # Use smaller chunks to avoid exceeding embedding token limits on code-heavy pages
    chunker = MarkdownChunker(chunk_size=800)

    def prepare(uri: str):
        return chunker.chunk_document(read_as_markdown(uri))

    store.ingest(links, prepare=prepare)


def test_ingest_with_generator():
    """Test that ingest works with a generator (iterable without __len__)."""
    from raghilda.chunker import MarkdownChunker

    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_generator_db",
    )

    chunker = MarkdownChunker()

    def make_docs():
        for i in range(3):
            doc = MarkdownDocument(origin=f"test_{i}", content=f"Document number {i}")
            yield doc

    # Use identity prepare since we're yielding MarkdownDocuments that need chunking
    store.ingest(
        make_docs(), prepare=lambda doc: chunker.chunk_document(doc), progress=False
    )
    assert store.size() == 3


def test_ingest_with_custom_prepare():
    """Test that ingest works with a custom prepare function."""
    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_prepare_db",
    )

    def custom_prepare(item: dict) -> MarkdownDocument:
        doc = MarkdownDocument(origin=item["id"], content=item["text"])
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(item["text"]),
                text=item["text"],
                token_count=len(item["text"]),
            )
        ]
        return doc

    items = [
        {"id": "doc1", "text": "First document content"},
        {"id": "doc2", "text": "Second document content"},
    ]

    store.ingest(items, prepare=custom_prepare, progress=False)
    assert store.size() == 2


def test_ingest_lazy_evaluation():
    """Test that ingest consumes the iterator lazily, not all at once."""
    import threading
    import time

    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_lazy_db",
    )

    consumed_count = 0
    max_concurrent_consumed = 0
    inserted_count = 0
    lock = threading.Lock()

    def tracking_generator():
        nonlocal consumed_count, max_concurrent_consumed
        for i in range(20):
            with lock:
                consumed_count += 1
                # Track max items consumed while inserts are still pending
                pending = consumed_count - inserted_count
                if pending > max_concurrent_consumed:
                    max_concurrent_consumed = pending
            yield {"id": f"doc_{i}", "text": f"Document content {i}"}

    def slow_prepare(item: dict) -> MarkdownDocument:
        nonlocal inserted_count
        # Simulate slow processing
        time.sleep(0.05)
        doc = MarkdownDocument(origin=item["id"], content=item["text"])
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(item["text"]),
                text=item["text"],
                token_count=len(item["text"]),
            )
        ]
        with lock:
            inserted_count += 1
        return doc

    # Use only 2 workers to make the test more deterministic
    store.ingest(
        tracking_generator(), prepare=slow_prepare, num_workers=2, progress=False
    )

    assert store.size() == 20
    # With lazy evaluation, max concurrent should be bounded by num_workers (plus some buffer)
    # If it consumed eagerly, all 20 would be consumed before any inserts complete
    assert max_concurrent_consumed <= 10, (
        f"Expected lazy consumption but {max_concurrent_consumed} items were consumed "
        f"concurrently, suggesting eager evaluation"
    )


def test_connect(tmp_path):
    db_path = tmp_path / "test.db"

    # Create a store with embeddings
    store = DuckDBStore.create(
        location=str(db_path),
        embed=EmbeddingOpenAI(model="text-embedding-3-small"),
        name="connect_test",
        title="Connect Test Store",
    )
    doc = MarkdownDocument(origin="test", content="hello world")
    doc.chunks = [_get_markdown_chunk(doc, start=0, end=5)]
    store.insert(doc)
    store.build_index()
    store.con.close()

    # Connect to existing store
    store2 = DuckDBStore.connect(str(db_path))
    assert store2.metadata.name == "connect_test"
    assert store2.metadata.title == "Connect Test Store"
    assert store2.size() == 1

    # Embedding provider should be restored
    assert store2.metadata.embed is not None
    assert isinstance(store2.metadata.embed, EmbeddingOpenAI)
    assert store2.metadata.embed.model == "text-embedding-3-small"

    # Retrieve should work (uses both BM25 and VSS)
    results = store2.retrieve("hello", top_k=1)
    assert len(results) >= 1
    assert results[0].text == "hello"
