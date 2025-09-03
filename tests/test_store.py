import pytest
from ragnar.store import DuckDBStore
from ragnar.document import ChunkedDocument, MarkdownDocument, MarkdownChunk
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

    def test_create_store(self, store):
        assert isinstance(store, DuckDBStore)
        assert store.metadata.name == "test_db"
        assert store.metadata.title == "Test DuckDB Store"
        assert store.metadata.embed is None

    @pytest.mark.parametrize("embed", [None, EmbeddingOpenAI()], indirect=True)
    def test_insert(self, store):
        doc = MarkdownDocument(origin="test", content="This is a test document.")
        chunked_doc = ChunkedDocument(
            document=doc,
            chunks=[
                MarkdownChunk(chunk_id=1, parent_doc=doc, start=0, end=4),
                MarkdownChunk(chunk_id=2, parent_doc=doc, start=5, end=7),
                MarkdownChunk(chunk_id=3, parent_doc=doc, start=8, end=9),
                MarkdownChunk(chunk_id=4, parent_doc=doc, start=10, end=14),
                MarkdownChunk(chunk_id=5, parent_doc=doc, start=15, end=23),
            ],
        )
        store.insert(chunked_doc)
