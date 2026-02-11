import pytest

pytest.importorskip("chromadb")

from raghilda.store import ChromaDBStore
from raghilda.document import MarkdownDocument
from raghilda.chunk import MarkdownChunk, RetrievedChunk


from chromadb import EmbeddingFunction, Embeddings, Documents


class DummyEmbeddingFunction(EmbeddingFunction):
    @staticmethod
    def name() -> str:
        return "test_embedding_function"

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
    store.insert(_make_doc())
    assert store.size() == 1

    results = store.retrieve("test", top_k=3)
    assert len(results) == 3
    for chunk in results:
        assert isinstance(chunk, RetrievedChunk)
        assert chunk.text is not None


def test_connect_with_embed(tmp_path):
    location = tmp_path / "chroma_store"
    embed = DummyEmbeddingFunction()
    store = ChromaDBStore.create(
        location=str(location),
        embed=embed,
        name="connect_test",
        overwrite=True,
    )
    store.insert(_make_doc())
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
    store.insert(_make_doc_with_overlapping_chunks())

    # With deoverlap=True (default), overlapping chunks should be merged
    results_merged = store.retrieve("hello", top_k=2, deoverlap=True)
    assert len(results_merged) == 1
    assert results_merged[0].start_index == 0
    assert results_merged[0].end_index == 17
    assert results_merged[0].text == "hello world hello"

    # With deoverlap=False, both chunks should be returned
    results_separate = store.retrieve("hello", top_k=2, deoverlap=False)
    assert len(results_separate) == 2


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
    store.ingest(make_docs(), prepare=lambda doc: chunker.chunk_document(doc), progress=False)
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


class TestChromaConvertible:
    """Tests for the ChromaConvertible protocol and to_chroma() conversion."""

    def test_embedding_openai_to_chroma_works(self):
        """EmbeddingOpenAI.to_chroma() should return a working ChromaDB function."""
        import os
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        from raghilda.embedding import EmbeddingOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        provider = EmbeddingOpenAI(model="text-embedding-3-small")
        chroma_func = provider.to_chroma()

        assert isinstance(chroma_func, OpenAIEmbeddingFunction)

        # Actually call the function to verify it works
        embeddings = chroma_func(["hello world", "test embedding"])
        assert len(embeddings) == 2
        assert all(len(emb) > 0 for emb in embeddings)

    def test_embedding_cohere_to_chroma_works(self):
        """EmbeddingCohere.to_chroma() should return a working ChromaDB function."""
        import os
        from chromadb.utils.embedding_functions import CohereEmbeddingFunction
        from raghilda.embedding import EmbeddingCohere

        if not (
            os.getenv("CO_API_KEY")
            or os.getenv("COHERE_API_KEY")
            or os.getenv("CHROMA_COHERE_API_KEY")
        ):
            pytest.skip("No Cohere API key set")

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
        import os
        from raghilda.embedding import EmbeddingOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        provider = EmbeddingOpenAI()
        store = ChromaDBStore.create(
            location=":memory:",
            embed=provider,
            name="test_raghilda_provider_e2e",
            overwrite=True,
        )

        # Insert a document
        store.insert(_make_doc())
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
        import os
        from raghilda._chroma_store import ChromaEmbeddingAdapter
        from raghilda.embedding import EmbeddingOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

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
