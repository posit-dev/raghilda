import pytest
import helpers as test_helpers
from raghilda.embedding import EmbeddingCohere, EmbedInputType


def test_embedding_cohere_does_not_use_chroma_cohere_api_key_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cohere

    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("CHROMA_COHERE_API_KEY", "test-chroma-key")

    observed_api_key: list[str | None] = []

    class FakeClient:
        def __init__(self, *, api_key):
            observed_api_key.append(api_key)

    monkeypatch.setattr(cohere, "Client", FakeClient)

    provider = EmbeddingCohere()

    assert provider.api_key is None
    assert observed_api_key == [None]


class TestEmbeddingCohere:
    @pytest.fixture(autouse=True)
    def setup(self):
        test_helpers.skip_if_no_cohere()

    def test_embedding_cohere_init(self):
        provider = EmbeddingCohere()

        assert provider.model == "embed-english-v3.0"
        assert provider.api_key is None
        assert provider.batch_size == 96

    def test_embedding_cohere_init_with_params(self):
        provider = EmbeddingCohere(
            model="embed-multilingual-v3.0",
            api_key="test-key",
            batch_size=50,
        )

        assert provider.model == "embed-multilingual-v3.0"
        assert provider.api_key == "test-key"
        assert provider.batch_size == 50

    def test_embedding_cohere_embed_documents(self):
        provider = EmbeddingCohere()
        texts = ["hello world", "testing embeddings"]
        embeddings = provider.embed(texts, EmbedInputType.DOCUMENT)

        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        assert all(isinstance(val, float) for emb in embeddings for val in emb)

    def test_embedding_cohere_embed_query(self):
        provider = EmbeddingCohere()
        texts = ["what is the meaning of life?"]
        embeddings = provider.embed(texts, EmbedInputType.QUERY)

        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0
        assert all(isinstance(val, float) for val in embeddings[0])

    def test_embedding_cohere_default_is_document(self):
        provider = EmbeddingCohere()
        texts = ["test text"]

        # Default should be DOCUMENT
        embeddings_default = provider.embed(texts)
        embeddings_document = provider.embed(texts, EmbedInputType.DOCUMENT)

        assert len(embeddings_default) == len(embeddings_document)
        assert len(embeddings_default[0]) == len(embeddings_document[0])

    def test_embedding_cohere_query_vs_document_differ(self):
        provider = EmbeddingCohere()
        text = ["test text for comparison"]

        embeddings_query = provider.embed(text, EmbedInputType.QUERY)
        embeddings_document = provider.embed(text, EmbedInputType.DOCUMENT)

        # Query and document embeddings should be different
        assert embeddings_query[0] != embeddings_document[0]

    def test_embedding_cohere_embed_empty_input(self):
        provider = EmbeddingCohere()
        texts = []
        embeddings = provider.embed(texts)

        assert embeddings == []

    def test_embedding_cohere_batch_size_handling(self):
        provider = EmbeddingCohere(batch_size=2)
        texts = ["text1", "text2", "text3", "text4", "text5"]

        assert provider.batch_size == 2
        assert len(texts) > provider.batch_size
        embeddings = provider.embed(texts)
        assert len(embeddings) == len(texts)

    def test_embedding_with_str(self):
        provider = EmbeddingCohere()
        text = "single text input"
        with pytest.raises(TypeError):
            provider.embed(text)
