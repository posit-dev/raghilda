import pytest
from tests import helpers as test_helpers
from raghilda.embedding import EmbeddingNVIDIA, EmbedInputType, embedding_from_config


class TestEmbeddingNVIDIA:
    @pytest.fixture(autouse=True)
    def setup(self):
        test_helpers.skip_if_no_nvidia()

    def test_embedding_nvidia_init(self):
        provider = EmbeddingNVIDIA()

        assert provider.model == "nvidia/llama-nemotron-embed-1b-v2"
        assert provider.base_url == "https://integrate.api.nvidia.com/v1"
        assert provider.api_key is None
        assert provider.batch_size == 20
        assert provider.truncate == "NONE"

    def test_embedding_nvidia_init_with_params(self):
        provider = EmbeddingNVIDIA(
            model="nvidia/nv-embedqa-e5-v5",
            base_url="https://custom.nvidia.com/v1",
            api_key="test-key",
            batch_size=10,
            truncate="END",
        )

        assert provider.model == "nvidia/nv-embedqa-e5-v5"
        assert provider.base_url == "https://custom.nvidia.com/v1"
        assert provider.api_key == "test-key"
        assert provider.batch_size == 10
        assert provider.truncate == "END"

    def test_embedding_nvidia_embed_integration(self):
        provider = EmbeddingNVIDIA()
        texts = ["hello world", "testing embeddings"]
        embeddings = provider.embed(texts)

        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        assert all(isinstance(val, float) for emb in embeddings for val in emb)

    def test_embedding_nvidia_query_vs_document(self):
        provider = EmbeddingNVIDIA()
        texts = ["what is machine learning?"]

        doc_emb = provider.embed(texts, input_type=EmbedInputType.DOCUMENT)
        query_emb = provider.embed(texts, input_type=EmbedInputType.QUERY)

        assert len(doc_emb) == 1
        assert len(query_emb) == 1
        # Query and document embeddings should differ
        assert doc_emb[0] != query_emb[0]

    def test_embedding_nvidia_embed_empty_input(self):
        provider = EmbeddingNVIDIA()
        embeddings = provider.embed([])
        assert embeddings == []

    def test_embedding_nvidia_embed_string_raises(self):
        provider = EmbeddingNVIDIA()
        with pytest.raises(TypeError):
            provider.embed("single text input")

    def test_embedding_nvidia_embed_empty_string_raises(self):
        provider = EmbeddingNVIDIA()
        with pytest.raises(ValueError):
            provider.embed(["valid", "", "also valid"])

    def test_embedding_nvidia_batch_size_handling(self):
        provider = EmbeddingNVIDIA(batch_size=2)
        texts = ["text1", "text2", "text3", "text4", "text5"]

        assert provider.batch_size == 2
        embeddings = provider.embed(texts)
        assert len(embeddings) == len(texts)

    def test_embedding_nvidia_get_config(self):
        provider = EmbeddingNVIDIA(
            model="nvidia/nv-embedqa-e5-v5",
            api_key="secret-key",
            batch_size=10,
            truncate="END",
        )
        config = provider.get_config()

        assert config["type"] == "EmbeddingNVIDIA"
        assert config["model"] == "nvidia/nv-embedqa-e5-v5"
        assert config["base_url"] == "https://integrate.api.nvidia.com/v1"
        assert config["batch_size"] == 10
        assert config["truncate"] == "END"
        assert "api_key" not in config

    def test_embedding_nvidia_from_config(self):
        config = {
            "type": "EmbeddingNVIDIA",
            "model": "nvidia/nv-embedqa-e5-v5",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "batch_size": 10,
            "truncate": "END",
        }
        provider = EmbeddingNVIDIA.from_config(config)

        assert provider.model == "nvidia/nv-embedqa-e5-v5"
        assert provider.base_url == "https://integrate.api.nvidia.com/v1"
        assert provider.batch_size == 10
        assert provider.truncate == "END"

    def test_embedding_nvidia_config_roundtrip(self):
        original = EmbeddingNVIDIA(model="nvidia/nv-embedqa-e5-v5", truncate="END")
        config = original.get_config()
        restored = EmbeddingNVIDIA.from_config(config)

        assert restored.model == original.model
        assert restored.base_url == original.base_url
        assert restored.batch_size == original.batch_size
        assert restored.truncate == original.truncate

    def test_embedding_nvidia_registry_roundtrip(self):
        original = EmbeddingNVIDIA(model="nvidia/nv-embedqa-e5-v5")
        config = original.get_config()
        restored = embedding_from_config(config)

        assert isinstance(restored, EmbeddingNVIDIA)
        assert restored.model == original.model
