from types import SimpleNamespace

import pytest

from raghilda.chunk import MarkdownChunk
from raghilda.document import Document, MarkdownDocument
import raghilda.store as store_module
from raghilda.store import ChromaDBStore, DuckDBStore, OpenAIStore, WriteResult


def test_document_uses_origin_field_not_id():
    doc = Document(content="hello")
    assert hasattr(doc, "origin")
    assert doc.origin is None
    assert not hasattr(doc, "id")


def test_store_api_uses_upsert_not_insert():
    assert hasattr(DuckDBStore, "upsert")
    assert hasattr(ChromaDBStore, "upsert")
    assert hasattr(OpenAIStore, "upsert")
    assert not hasattr(DuckDBStore, "insert")
    assert not hasattr(ChromaDBStore, "insert")
    assert not hasattr(OpenAIStore, "insert")


def test_store_exports_write_result_not_insert_result():
    assert WriteResult is store_module.WriteResult
    assert not hasattr(store_module, "InsertResult")


def test_openai_upsert_rejects_chunked_document():
    class _SinglePage:
        def __init__(self):
            self.data = []

        def has_next_page(self):
            return False

    class FakeVectorStoreFiles:
        def list(self, **kwargs):
            return _SinglePage()

        def upload_and_poll(self, **kwargs):
            raise AssertionError("upload_and_poll should not be called")

        def delete(self, **kwargs):
            raise AssertionError("delete should not be called")

    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=FakeVectorStoreFiles()),
        files=SimpleNamespace(content=lambda **kwargs: None),
    )
    store = OpenAIStore(client=fake_client, store_id="vs_test")

    doc = MarkdownDocument(origin="doc", content="hello")
    doc.chunks = [
        MarkdownChunk(
            text="hello",
            start_index=0,
            end_index=5,
            char_count=5,
        )
    ]

    with pytest.raises(ValueError, match="does not support chunked documents"):
        store.upsert(doc)
