import pytest

pytest.importorskip("chromadb")

from raghilda.store import ChromaDBStore
from raghilda.document import MarkdownDocument
from raghilda.chunk import MarkdownChunk, RetrievedChunk


class TestEmbeddingFunction:
    def __call__(self, input):
        if isinstance(input, str):
            raise TypeError("Embedding function expects a sequence of strings")
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
        embedding_function=TestEmbeddingFunction(),
        name="test_store_insert",
        overwrite=True,
    )
    store.insert(_make_doc())
    assert store.size() == 1

    results = store.retrieve("test", top_k=3)
    assert len(results) == 3
    for chunk in results:
        assert isinstance(chunk, RetrievedChunk)
        assert chunk.text is not None


def test_connect_with_embedding_function(tmp_path):
    location = tmp_path / "chroma_store"
    embedding_function = TestEmbeddingFunction()
    store = ChromaDBStore.create(
        location=str(location),
        embedding_function=embedding_function,
        name="connect_test",
        overwrite=True,
    )
    store.insert(_make_doc())
    if hasattr(store.client, "persist"):
        store.client.persist()

    store2 = ChromaDBStore.connect(
        location=str(location),
        name="connect_test",
        embedding_function=embedding_function,
    )
    assert store2.size() == 1
    results = store2.retrieve("document", top_k=1)
    assert len(results) == 1
