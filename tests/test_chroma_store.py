import pytest
import threading
import time
import json
from typing import Annotated
from tests import helpers as test_helpers

pytest.importorskip("chromadb")

from raghilda.store import ChromaDBStore
from raghilda.document import MarkdownDocument
from raghilda.chunk import MarkdownChunk, RetrievedChunk


from chromadb import EmbeddingFunction, Embeddings, Documents


class DummyEmbeddingFunction(EmbeddingFunction):
    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "test_embedding_function"

    @staticmethod
    def build_from_config(config: dict) -> "DummyEmbeddingFunction":
        DummyEmbeddingFunction.validate_config(config)
        return DummyEmbeddingFunction()

    def get_config(self) -> dict:
        return {}

    def _embed(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            embeddings.append(
                [
                    float(len(text)),
                    float(sum(ord(c) for c in text) % 1000),
                    float(len(text.split())),
                ]
            )
        return embeddings

    def __call__(self, input: Documents) -> Embeddings:
        return self._embed(input)

    def embed_documents(self, input: Documents) -> Embeddings:
        return self._embed(input)

    def embed_query(self, input: Documents) -> Embeddings:
        return self._embed(input)


def _make_doc():
    doc = MarkdownDocument(origin="test", content="This is a test document.")
    doc.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=4,
            text=doc.content[0:4],
            token_count=len(doc.content[0:4]),
        ),
        MarkdownChunk(
            start_index=5,
            end_index=7,
            text=doc.content[5:7],
            token_count=len(doc.content[5:7]),
        ),
        MarkdownChunk(
            start_index=8,
            end_index=9,
            text=doc.content[8:9],
            token_count=len(doc.content[8:9]),
        ),
        MarkdownChunk(
            start_index=10,
            end_index=14,
            text=doc.content[10:14],
            token_count=len(doc.content[10:14]),
        ),
        MarkdownChunk(
            start_index=15,
            end_index=23,
            text=doc.content[15:23],
            token_count=len(doc.content[15:23]),
        ),
    ]
    return doc


def test_create_store():
    store = ChromaDBStore.create(
        location=":memory:",
        name="test_store",
        title="Test ChromaDB Store",
        overwrite=True,
    )
    assert isinstance(store, ChromaDBStore)
    assert store.metadata.name == "test_store"
    assert store.metadata.title == "Test ChromaDB Store"


def test_insert_and_retrieve():
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_store_insert",
        overwrite=True,
    )
    store.upsert(_make_doc())
    assert store.size() == 1

    results = store.retrieve("test", top_k=3)
    assert len(results) == 3
    for chunk in results:
        assert isinstance(chunk, RetrievedChunk)
        assert chunk.text is not None


def test_insert_same_content_but_different_chunking_updates():
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_store_chunk_update",
        overwrite=True,
        attributes={"tenant": str},
    )
    content = "hello world"
    doc1 = MarkdownDocument(
        origin="same-origin",
        content=content,
        attributes={"tenant": "docs"},
    )
    doc1.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=len(content),
            text=content,
            token_count=len(content),
        )
    ]
    first = store.upsert(doc1)
    assert first.action == "inserted"
    assert store.collection.count() == 1

    doc2 = MarkdownDocument(
        origin="same-origin",
        content=content,
        attributes={"tenant": "eng"},
    )
    doc2.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=5,
            text=content[:5],
            token_count=5,
        ),
        MarkdownChunk(
            start_index=6,
            end_index=len(content),
            text=content[6:],
            token_count=len(content[6:]),
        ),
    ]
    second = store.upsert(doc2)
    assert second.action == "replaced"
    assert second.replaced_document is not None
    assert second.replaced_document.attributes == {"tenant": "docs"}
    assert second.document.chunks is not None
    assert len(second.document.chunks) == 2
    assert store.collection.count() == 2


def test_insert_unchanged_preserves_document_attributes():
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_store_skip_attributes",
        overwrite=True,
        attributes={"tenant": str},
    )

    content = "hello world"
    doc = MarkdownDocument(
        origin="same-origin",
        content=content,
        attributes={"tenant": "docs"},
    )
    doc.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=len(content),
            text=content,
            token_count=len(content),
        )
    ]

    store.upsert(doc)
    second = store.upsert(doc)
    assert second.action == "skipped"
    assert second.document.attributes == {"tenant": "docs"}


def test_insert_same_content_skips_when_existing_chunks_returned_in_reverse_order(
    monkeypatch,
):
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_store_skip_reordered_existing",
        overwrite=True,
    )
    content = "hello world"
    doc = MarkdownDocument(origin="same-origin", content=content)
    doc.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=5,
            text=content[:5],
            token_count=5,
        ),
        MarkdownChunk(
            start_index=6,
            end_index=11,
            text=content[6:],
            token_count=5,
        ),
    ]
    store.upsert(doc)

    original_get = store.collection.get

    def reverse_origin_rows(*args, **kwargs):
        result = original_get(*args, **kwargs)
        if kwargs.get("where") != {"origin": "same-origin"}:
            return result

        ids = list(result.get("ids") or [])
        documents = list(result.get("documents") or [])
        metadatas = list(result.get("metadatas") or [])
        order = list(reversed(range(len(ids))))
        result["ids"] = [ids[idx] for idx in order]
        result["documents"] = [documents[idx] for idx in order]
        result["metadatas"] = [metadatas[idx] for idx in order]
        return result

    monkeypatch.setattr(store.collection, "get", reverse_origin_rows)

    skipped = store.upsert(doc)
    assert skipped.action == "skipped"
    assert [chunk.text for chunk in skipped.document.chunks or []] == ["hello", "world"]


def test_insert_result_document_includes_merged_chunk_attributes():
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_store_insert_result_merged_attrs",
        overwrite=True,
        attributes={"tenant": str},
    )

    original = MarkdownDocument(
        origin="same-origin",
        content="hello world",
    )
    original.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=len(original.content),
            text=original.content,
            token_count=len(original.content),
            attributes={"tenant": "docs"},
        )
    ]

    inserted = store.upsert(original)
    assert inserted.action == "inserted"
    assert inserted.document.attributes == {"tenant": "docs"}

    updated = MarkdownDocument(
        origin="same-origin",
        content="hello world updated",
    )
    updated.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=len(updated.content),
            text=updated.content,
            token_count=len(updated.content),
            attributes={"tenant": "docs"},
        )
    ]

    replaced = store.upsert(updated, skip_if_unchanged=False)
    assert replaced.action == "replaced"
    assert replaced.document.attributes == {"tenant": "docs"}
    assert replaced.replaced_document is not None
    assert replaced.replaced_document.attributes == {"tenant": "docs"}


def test_insert_keeps_existing_chunks_when_upsert_fails(monkeypatch):
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_store_upsert_failure",
        overwrite=True,
    )

    doc = MarkdownDocument(origin="same-origin", content="hello world")
    doc.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=len(doc.content),
            text=doc.content,
            token_count=len(doc.content),
        )
    ]
    store.upsert(doc)

    def fail_upsert(**kwargs):
        raise RuntimeError("upsert failed")

    monkeypatch.setattr(store.collection, "upsert", fail_upsert)

    updated = MarkdownDocument(origin="same-origin", content="goodbye world")
    updated.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=len(updated.content),
            text=updated.content,
            token_count=len(updated.content),
        )
    ]

    with pytest.raises(RuntimeError, match="upsert failed"):
        store.upsert(updated, skip_if_unchanged=False)

    existing = store.collection.get(
        where={"origin": "same-origin"},
        include=["documents"],
    )
    assert existing["ids"] == ["same-origin:0"]
    assert existing["documents"] == ["hello world"]


def test_insert_raises_when_stale_chunk_delete_fails(monkeypatch):
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_store_stale_delete_failure",
        overwrite=True,
    )

    content = "hello world"
    original = MarkdownDocument(origin="same-origin", content=content)
    original.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=5,
            text=content[:5],
            token_count=5,
        ),
        MarkdownChunk(
            start_index=6,
            end_index=len(content),
            text=content[6:],
            token_count=len(content[6:]),
        ),
    ]
    store.upsert(original)

    delete_calls = []

    def fail_delete(**kwargs):
        delete_calls.append(kwargs)
        raise RuntimeError("delete failed")

    monkeypatch.setattr(store.collection, "delete", fail_delete)

    updated = MarkdownDocument(origin="same-origin", content=content)
    updated.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=len(content),
            text=content,
            token_count=len(content),
        )
    ]

    with pytest.raises(RuntimeError, match="delete failed"):
        store.upsert(updated, skip_if_unchanged=False)
    assert len(delete_calls) == 1
    assert delete_calls[0]["ids"] == ["same-origin:1"]


def test_upsert_replaces_when_existing_metadata_missing_content_text(monkeypatch):
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_store_missing_content_text",
        overwrite=True,
    )

    original = MarkdownDocument(origin="same-origin", content="hello world")
    original.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=len(original.content),
            text=original.content,
            token_count=len(original.content),
        )
    ]
    store.upsert(original)

    original_get = store.collection.get

    def missing_content_text_get(*args, **kwargs):
        result = original_get(*args, **kwargs)
        metadatas = []
        for metadata in result.get("metadatas") or []:
            if metadata is None:
                metadatas.append(None)
                continue
            without_content = dict(metadata)
            without_content.pop("_raghilda_content_text", None)
            metadatas.append(without_content)
        result["metadatas"] = metadatas
        return result

    monkeypatch.setattr(store.collection, "get", missing_content_text_get)

    updated = MarkdownDocument(origin="same-origin", content="updated content")
    updated.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=len(updated.content),
            text=updated.content,
            token_count=len(updated.content),
        )
    ]

    result = store.upsert(updated, skip_if_unchanged=False)

    assert result.action == "replaced"
    assert result.replaced_document is None
    assert result.document.content == "updated content"

    existing = store.collection.get(
        where={"origin": "same-origin"},
        include=["documents"],
    )
    assert existing["documents"] == ["updated content"]


def test_upsert_accepts_existing_empty_content_text_metadata():
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_store_empty_content_text",
        overwrite=True,
    )

    original = MarkdownDocument(origin="same-origin", content="")
    original.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=0,
            text="",
            token_count=0,
        )
    ]

    store.upsert(original)
    result = store.upsert(original)

    assert result.action == "skipped"
    assert result.document.content == ""


def test_insert_same_origin_concurrent_updates_do_not_leave_stale_chunks(monkeypatch):
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_store_concurrent_same_origin",
        overwrite=True,
    )

    content = "hello world"
    doc_two_chunks = MarkdownDocument(origin="same-origin", content=content)
    doc_two_chunks.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=5,
            text=content[:5],
            token_count=5,
        ),
        MarkdownChunk(
            start_index=6,
            end_index=len(content),
            text=content[6:],
            token_count=len(content[6:]),
        ),
    ]
    doc_one_chunk = MarkdownDocument(origin="same-origin", content=content)
    doc_one_chunk.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=len(content),
            text=content,
            token_count=len(content),
        )
    ]

    original_get = store.collection.get
    original_upsert = store.collection.upsert
    get_calls = 0
    get_calls_lock = threading.Lock()
    both_reads_finished = threading.Event()

    def coordinated_get(*args, **kwargs):
        nonlocal get_calls
        result = original_get(*args, **kwargs)
        with get_calls_lock:
            get_calls += 1
            if get_calls >= 2:
                both_reads_finished.set()
        both_reads_finished.wait(timeout=0.2)
        return result

    def ordered_upsert(*args, **kwargs):
        ids = kwargs.get("ids", [])
        if len(ids) == 1:
            time.sleep(0.05)
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(store.collection, "get", coordinated_get)
    monkeypatch.setattr(store.collection, "upsert", ordered_upsert)

    t1 = threading.Thread(
        target=lambda: store.upsert(doc_two_chunks, skip_if_unchanged=False)
    )
    t2 = threading.Thread(
        target=lambda: store.upsert(doc_one_chunk, skip_if_unchanged=False)
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    existing = store.collection.get(
        where={"origin": "same-origin"},
        include=["documents"],
    )
    assert sorted(existing["ids"]) == ["same-origin:0"]
    assert existing["documents"] == [content]


def test_insert_releases_origin_locks_for_completed_origins():
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_store_origin_lock_cleanup",
        overwrite=True,
    )

    for idx in range(100):
        content = f"doc {idx}"
        doc = MarkdownDocument(origin=f"origin-{idx}", content=content)
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(content),
                text=content,
                token_count=len(content),
            )
        ]
        store.upsert(doc, skip_if_unchanged=False)

    assert store._origin_locks == {}


def test_insert_stores_document_content_once_in_metadata():
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_store_content_metadata_once",
        overwrite=True,
    )

    content = "hello world"
    doc = MarkdownDocument(origin="same-origin", content=content)
    doc.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=5,
            text=content[:5],
            token_count=5,
        ),
        MarkdownChunk(
            start_index=6,
            end_index=len(content),
            text=content[6:],
            token_count=len(content[6:]),
        ),
    ]
    store.upsert(doc)

    existing = store.collection.get(
        where={"origin": "same-origin"},
        include=["metadatas"],
    )
    metadatas = existing.get("metadatas") or []
    rows_with_content = [
        metadata
        for metadata in metadatas
        if metadata and metadata.get("_raghilda_content_text") is not None
    ]
    assert len(rows_with_content) == 1
    assert rows_with_content[0]["_raghilda_content_text"] == content


def test_connect_with_embed(tmp_path):
    location = tmp_path / "chroma_store"
    embed = DummyEmbeddingFunction()
    store = ChromaDBStore.create(
        location=str(location),
        embed=embed,
        name="connect_test",
        overwrite=True,
    )
    store.upsert(_make_doc())
    if hasattr(store.client, "persist"):
        store.client.persist()

    store2 = ChromaDBStore.connect(
        location=str(location),
        name="connect_test",
        embed=embed,
    )
    assert store2.size() == 1
    results = store2.retrieve("document", top_k=1)
    assert len(results) == 1


def test_connect_restores_attributes_schema(tmp_path):
    location = tmp_path / "chroma_store_with_attributes"
    store = ChromaDBStore.create(
        location=str(location),
        embed=DummyEmbeddingFunction(),
        name="connect_attributes_test",
        overwrite=True,
        attributes={"tenant": str, "priority": int},
    )
    doc = _make_doc()
    doc.attributes = {"tenant": "docs", "priority": 1}
    store.upsert(doc)
    if hasattr(store.client, "persist"):
        store.client.persist()

    store2 = ChromaDBStore.connect(
        location=str(location),
        name="connect_attributes_test",
        embed=DummyEmbeddingFunction(),
    )
    assert store2.metadata.attributes_schema == {"tenant": str, "priority": int}


def test_connect_rejects_internal_attribute_names_from_metadata():
    schema_json = json.dumps(
        {
            "_raghilda_content_hash": {
                "type": "str",
                "nullable": False,
                "required": True,
            },
        }
    )

    class FakeCollection:
        metadata = {"raghilda_attributes_schema_json": schema_json}

    class FakeClient:
        def get_collection(self, *, name, embedding_function=None):
            return FakeCollection()

    with pytest.raises(ValueError, match="_raghilda_content_hash"):
        ChromaDBStore.connect(
            name="connect_internal_attr_test",
            client=FakeClient(),
            embed=DummyEmbeddingFunction(),
        )


def _make_doc_with_overlapping_chunks():
    """Create a document with overlapping chunks for testing deoverlap."""
    content = "hello world hello"
    doc = MarkdownDocument(origin="test_overlap", content=content)
    doc.chunks = [
        MarkdownChunk(
            start_index=0,
            end_index=11,
            text=content[0:11],  # "hello world"
            token_count=11,
        ),
        MarkdownChunk(
            start_index=6,
            end_index=17,
            text=content[6:17],  # "world hello"
            token_count=11,
        ),
    ]
    return doc


def test_retrieve_with_deoverlap():
    """Test that overlapping chunks are merged when deoverlap=True."""
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_deoverlap",
        overwrite=True,
    )
    store.upsert(_make_doc_with_overlapping_chunks())

    # With deoverlap=True (default), overlapping chunks should be merged
    results_merged = store.retrieve("hello", top_k=2, deoverlap=True)
    assert len(results_merged) == 1
    assert results_merged[0].start_index == 0
    assert results_merged[0].end_index == 17
    assert results_merged[0].text == "hello world hello"

    # With deoverlap=False, both chunks should be returned
    results_separate = store.retrieve("hello", top_k=2, deoverlap=False)
    assert len(results_separate) == 2


def test_retrieve_with_deoverlap_aggregates_attributes():
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_deoverlap_attributes",
        overwrite=True,
        attributes={"topic": str},
    )
    doc = _make_doc_with_overlapping_chunks()
    doc.attributes = {"topic": "first"}
    assert doc.chunks is not None
    doc.chunks[0].context = "h1"
    doc.chunks[1].context = "h2"
    doc.chunks[1].attributes = {"topic": "second"}

    store.upsert(doc)
    results = store.retrieve("hello", top_k=2, deoverlap=True)

    assert len(results) == 1
    assert results[0].context == "h1"
    assert results[0].attributes == {"topic": ["first", "second"]}


def test_insert_and_retrieve_with_attributes_filter():
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_attributes_filter",
        overwrite=True,
        attributes={"tenant": str, "topic": str},
    )

    doc = _make_doc()
    doc.attributes = {"tenant": "docs", "topic": "general"}
    assert doc.chunks is not None
    doc.chunks[0].attributes = {"topic": "intro"}
    store.upsert(doc)

    intro = store.retrieve(
        "test",
        top_k=5,
        attributes_filter="tenant = 'docs' AND topic = 'intro'",
        deoverlap=False,
    )
    assert len(intro) >= 1
    for chunk in intro:
        assert chunk.attributes is not None
        assert chunk.attributes.get("tenant") == "docs"
        assert chunk.attributes.get("topic") == "intro"

    intro_dict = store.retrieve(
        "test",
        top_k=5,
        attributes_filter={
            "type": "and",
            "filters": [
                {"type": "eq", "key": "tenant", "value": "docs"},
                {"type": "in", "key": "topic", "value": ["intro", "other"]},
            ],
        },
        deoverlap=False,
    )
    assert len(intro_dict) >= 1
    for chunk in intro_dict:
        assert chunk.attributes is not None
        assert chunk.attributes.get("tenant") == "docs"
        assert chunk.attributes.get("topic") == "intro"


def test_create_with_attributes_schema_class_annotations():
    class AttributesSpec:
        tenant: str
        topic: str

    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="test_attributes_schema_class",
        overwrite=True,
        attributes=AttributesSpec,
    )

    assert store.metadata.attributes_schema == {
        "tenant": str,
        "topic": str,
    }


def test_create_rejects_vector_attributes_annotations():
    with pytest.raises(ValueError, match="Vector attribute types are not supported"):
        ChromaDBStore.create(
            location=":memory:",
            embed=DummyEmbeddingFunction(),
            name="test_attributes_schema_vector_reject",
            overwrite=True,
            attributes={"embedding25": Annotated[list[float], 25]},
        )


def test_create_rejects_object_attributes_annotations():
    with pytest.raises(ValueError, match="Object attribute types are not supported"):
        ChromaDBStore.create(
            location=":memory:",
            embed=DummyEmbeddingFunction(),
            name="test_attributes_schema_object_reject",
            overwrite=True,
            attributes={"details": {"source": str}},
        )


def test_create_rejects_optional_attributes_annotations():
    with pytest.raises(
        ValueError, match="Optional attribute values are not supported for 'topic'"
    ):
        ChromaDBStore.create(
            location=":memory:",
            embed=DummyEmbeddingFunction(),
            name="test_attributes_schema_optional_reject",
            overwrite=True,
            attributes={"tenant": str, "topic": str | None},
        )


def test_create_rejects_defaulted_attributes_annotations():
    with pytest.raises(
        ValueError, match="Optional attribute values are not supported for 'priority'"
    ):
        ChromaDBStore.create(
            location=":memory:",
            embed=DummyEmbeddingFunction(),
            name="test_attributes_schema_default_reject",
            overwrite=True,
            attributes={"tenant": str, "priority": (int, 0)},
        )


def test_create_rejects_invalid_attribute_names():
    with pytest.raises(ValueError, match="must match"):
        ChromaDBStore.create(
            location=":memory:",
            embed=DummyEmbeddingFunction(),
            name="test_attributes_schema_name_reject",
            overwrite=True,
            attributes={"tenant-id": str},
        )


@pytest.mark.parametrize(
    "attribute_name",
    ["_raghilda_content_hash", "_raghilda_content_text"],
)
def test_create_rejects_internal_attribute_names(attribute_name):
    with pytest.raises(ValueError, match=attribute_name):
        ChromaDBStore.create(
            location=":memory:",
            embed=DummyEmbeddingFunction(),
            name="test_attributes_schema_internal_reject",
            overwrite=True,
            attributes={attribute_name: str},
        )


def test_ingest_with_generator():
    """Test that ingest works with a generator (iterable without __len__)."""
    from raghilda.chunker import MarkdownChunker

    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="ingest_generator",
        overwrite=True,
    )

    chunker = MarkdownChunker()

    def make_docs():
        for i in range(3):
            doc = MarkdownDocument(origin=f"test_{i}", content=f"Document number {i}")
            yield doc

    # Use prepare to chunk the documents since we're yielding MarkdownDocuments
    store.ingest(
        make_docs(), prepare=lambda doc: chunker.chunk_document(doc), progress=False
    )
    assert store.size() == 3


def test_ingest_with_custom_prepare():
    """Test that ingest works with a custom prepare function."""
    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="ingest_prepare",
        overwrite=True,
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

    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        name="ingest_lazy",
        overwrite=True,
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


class TestChromaConvertible:
    """Tests for the ChromaConvertible protocol and to_chroma() conversion."""

    def test_embedding_openai_to_chroma_works(self):
        """EmbeddingOpenAI.to_chroma() should return a working ChromaDB function."""
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        from raghilda.embedding import EmbeddingOpenAI

        test_helpers.skip_if_no_openai()

        provider = EmbeddingOpenAI(model="text-embedding-3-small")
        chroma_func = provider.to_chroma()

        assert isinstance(chroma_func, OpenAIEmbeddingFunction)

        # Actually call the function to verify it works
        embeddings = chroma_func(["hello world", "test embedding"])
        assert len(embeddings) == 2
        assert all(len(emb) > 0 for emb in embeddings)

    def test_embedding_cohere_to_chroma_works(self):
        """EmbeddingCohere.to_chroma() should return a working ChromaDB function."""
        from chromadb.utils.embedding_functions import CohereEmbeddingFunction
        from raghilda.embedding import EmbeddingCohere

        test_helpers.skip_if_no_cohere()

        provider = EmbeddingCohere(model="embed-english-v3.0")
        chroma_func = provider.to_chroma()

        assert isinstance(chroma_func, CohereEmbeddingFunction)

        # Actually call the function to verify it works
        embeddings = chroma_func(["hello world", "test embedding"])
        assert len(embeddings) == 2
        assert all(len(emb) > 0 for emb in embeddings)

    def test_to_chroma_embedding_function_passthrough(self):
        """ChromaDB embedding functions should pass through unchanged."""
        from raghilda._chroma_store import _to_chroma_embedding_function

        dummy_ef = DummyEmbeddingFunction()
        result = _to_chroma_embedding_function(dummy_ef)

        assert result is dummy_ef

    def test_to_chroma_embedding_function_none(self):
        """None should return None."""
        from raghilda._chroma_store import _to_chroma_embedding_function

        result = _to_chroma_embedding_function(None)
        assert result is None

    def test_create_store_with_raghilda_provider_insert_retrieve(self):
        """ChromaDBStore should work with raghilda EmbeddingProvider for insert and retrieve."""
        from raghilda.embedding import EmbeddingOpenAI

        test_helpers.skip_if_no_openai()

        provider = EmbeddingOpenAI()
        store = ChromaDBStore.create(
            location=":memory:",
            embed=provider,
            name="test_raghilda_provider_e2e",
            overwrite=True,
        )

        # Insert a document
        store.upsert(_make_doc())
        assert store.size() == 1

        # Retrieve and verify it works
        results = store.retrieve("test document", top_k=2)
        assert len(results) > 0
        assert all(chunk.text is not None for chunk in results)


class TestChromaEmbeddingAdapter:
    """Tests for the ChromaEmbeddingAdapter fallback."""

    def test_adapter_wraps_provider(self):
        """Adapter should wrap an EmbeddingProvider."""
        from raghilda._chroma_store import ChromaEmbeddingAdapter
        from raghilda._embedding import EmbeddingProvider, EmbedInputType

        # Create a simple mock provider
        class MockProvider(EmbeddingProvider):
            def embed(self, x, input_type=EmbedInputType.DOCUMENT):
                return [[1.0, 2.0, 3.0] for _ in x]

            def get_config(self):
                return {"type": "MockProvider"}

            @classmethod
            def from_config(cls, config):
                return cls()

        provider = MockProvider()
        adapter = ChromaEmbeddingAdapter(provider)

        assert adapter._provider is provider

    def test_adapter_call_generates_embeddings(self):
        """Adapter.__call__ should generate document embeddings."""
        from raghilda._chroma_store import ChromaEmbeddingAdapter
        from raghilda._embedding import EmbeddingProvider, EmbedInputType
        import numpy as np

        class MockProvider(EmbeddingProvider):
            def __init__(self):
                self.last_input_type = None

            def embed(self, x, input_type=EmbedInputType.DOCUMENT):
                self.last_input_type = input_type
                return [[1.0, 2.0, 3.0] for _ in x]

            def get_config(self):
                return {"type": "MockProvider"}

            @classmethod
            def from_config(cls, config):
                return cls()

        provider = MockProvider()
        adapter = ChromaEmbeddingAdapter(provider)

        result = adapter(["hello", "world"])

        assert len(result) == 2
        assert all(isinstance(emb, np.ndarray) for emb in result)
        assert provider.last_input_type == EmbedInputType.DOCUMENT

    def test_adapter_embed_query_generates_query_embeddings(self):
        """Adapter.embed_query should generate query embeddings."""
        from raghilda._chroma_store import ChromaEmbeddingAdapter
        from raghilda._embedding import EmbeddingProvider, EmbedInputType
        import numpy as np

        class MockProvider(EmbeddingProvider):
            def __init__(self):
                self.last_input_type = None

            def embed(self, x, input_type=EmbedInputType.DOCUMENT):
                self.last_input_type = input_type
                return [[1.0, 2.0, 3.0] for _ in x]

            def get_config(self):
                return {"type": "MockProvider"}

            @classmethod
            def from_config(cls, config):
                return cls()

        provider = MockProvider()
        adapter = ChromaEmbeddingAdapter(provider)

        result = adapter.embed_query(["search query"])

        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)
        assert provider.last_input_type == EmbedInputType.QUERY

    def test_adapter_get_config(self):
        """Adapter.get_config should include provider config."""
        from raghilda._chroma_store import ChromaEmbeddingAdapter
        from raghilda._embedding import EmbeddingProvider, EmbedInputType

        class MockProvider(EmbeddingProvider):
            def embed(self, x, input_type=EmbedInputType.DOCUMENT):
                return [[1.0, 2.0, 3.0] for _ in x]

            def get_config(self):
                return {"type": "MockProvider", "model": "test-model"}

            @classmethod
            def from_config(cls, config):
                return cls()

        provider = MockProvider()
        adapter = ChromaEmbeddingAdapter(provider)

        config = adapter.get_config()

        assert "provider_config" in config
        assert config["provider_config"]["type"] == "MockProvider"
        assert config["provider_config"]["model"] == "test-model"

    def test_adapter_build_from_config(self):
        """Adapter.build_from_config should restore provider."""
        from raghilda._chroma_store import ChromaEmbeddingAdapter
        from raghilda.embedding import EmbeddingOpenAI

        test_helpers.skip_if_no_openai()

        # Create adapter with real provider
        original_provider = EmbeddingOpenAI(model="text-embedding-3-small")
        adapter = ChromaEmbeddingAdapter(original_provider)

        # Get config and rebuild
        config = adapter.get_config()
        restored = ChromaEmbeddingAdapter.build_from_config(config)

        assert isinstance(restored._provider, EmbeddingOpenAI)
        assert restored._provider.model == "text-embedding-3-small"

    def test_adapter_name(self):
        """Adapter.name() should return the registered name."""
        from raghilda._chroma_store import ChromaEmbeddingAdapter, _ADAPTER_NAME

        assert ChromaEmbeddingAdapter.name() == _ADAPTER_NAME

    def test_adapter_used_for_non_convertible_provider(self):
        """Adapter should be used for providers without to_chroma()."""
        from raghilda._chroma_store import (
            _to_chroma_embedding_function,
            ChromaEmbeddingAdapter,
        )
        from raghilda._embedding import EmbeddingProvider, EmbedInputType

        # Provider without to_chroma() method
        class CustomProvider(EmbeddingProvider):
            def embed(self, x, input_type=EmbedInputType.DOCUMENT):
                return [[1.0, 2.0, 3.0] for _ in x]

            def get_config(self):
                return {"type": "CustomProvider"}

            @classmethod
            def from_config(cls, config):
                return cls()

        provider = CustomProvider()
        result = _to_chroma_embedding_function(provider)

        assert isinstance(result, ChromaEmbeddingAdapter)
        assert result._provider is provider
