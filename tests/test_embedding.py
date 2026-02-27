import pytest
from tests import helpers as test_helpers
from raghilda.embedding import EmbeddingOpenAI


class TestEmbeddingOpenAI:
    @pytest.fixture(autouse=True)
    def setup(self):
        test_helpers.skip_if_no_openai()

    def test_embedding_openai_init(self):
        provider = EmbeddingOpenAI()

        assert provider.model == "text-embedding-3-small"
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.api_key is None
        assert provider.batch_size == 20

    def test_embedding_openai_init_with_params(self):
        provider = EmbeddingOpenAI(
            model="text-embedding-ada-002",
            base_url="https://custom.api.com/v1",
            api_key="test-key",
            batch_size=10,
        )

        assert provider.model == "text-embedding-ada-002"
        assert provider.base_url == "https://custom.api.com/v1"
        assert provider.api_key == "test-key"
        assert provider.batch_size == 10

    def test_embedding_openai_embed_integration(self):
        provider = EmbeddingOpenAI()
        texts = ["hello world", "testing embeddings"]
        embeddings = provider.embed(texts)

        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        assert all(isinstance(val, float) for emb in embeddings for val in emb)

    def test_embedding_openai_embed_empty_input(self):
        provider = EmbeddingOpenAI()
        texts = []
        embeddings = provider.embed(texts)

        assert embeddings == []

    def test_embedding_openai_batch_size_handling(self):
        provider = EmbeddingOpenAI(batch_size=2)
        texts = ["text1", "text2", "text3", "text4", "text5"]

        assert provider.batch_size == 2
        assert len(texts) > provider.batch_size
        embeddings = provider.embed(texts)
        assert len(embeddings) == len(texts)

    def test_embedding_with_str(self):
        provider = EmbeddingOpenAI()
        text = "single text input"
        with pytest.raises(TypeError):
            provider.embed(text)
