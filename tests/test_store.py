from ragnar.store import DuckDBStore
from ragnar.document import ChunkedDocument, MarkdownDocument, LazyMarkdownChunk


class TestDuckDBStore:
    def create_store(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            name="test_db",
            title="Test DuckDB Store",
        )
        return store

    def test_create_store(self):
        store = self.create_store()
        assert isinstance(store, DuckDBStore)
        assert store.metadata.name == "test_db"
        assert store.metadata.title == "Test DuckDB Store"
        assert store.metadata.embed is None

    def test_insert(self):
        store = self.create_store()
        doc = MarkdownDocument(origin="test", content="This is a test document.")
        chunked_doc = ChunkedDocument(
            document=doc,
            chunks=[
                LazyMarkdownChunk(chunk_id=1, parent_doc=doc, start=0, end=4),
                LazyMarkdownChunk(chunk_id=2, parent_doc=doc, start=5, end=7),
                LazyMarkdownChunk(chunk_id=3, parent_doc=doc, start=8, end=9),
                LazyMarkdownChunk(chunk_id=4, parent_doc=doc, start=10, end=14),
                LazyMarkdownChunk(chunk_id=5, parent_doc=doc, start=15, end=23),
            ],
        )
        store.insert(chunked_doc)
