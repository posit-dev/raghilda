import pytest

from raghilda.embedding import (
    EmbeddingSentenceTransformers,
    EmbedInputType,
    embedding_from_config,
)

sentence_transformers = pytest.importorskip("sentence_transformers")


class TestEmbeddingSentenceTransformers:
    def test_init_defaults(self):
        provider = EmbeddingSentenceTransformers()

        assert provider.model == "all-MiniLM-L6-v2"
        assert provider.device is None
        assert provider.batch_size == 64

    def test_init_with_params(self):
        provider = EmbeddingSentenceTransformers(
            model="sentence-transformers/all-mpnet-base-v2",
            device="cpu",
            batch_size=32,
        )

        assert provider.model == "sentence-transformers/all-mpnet-base-v2"
        assert provider.device == "cpu"
        assert provider.batch_size == 32

    def test_embed(self):
        provider = EmbeddingSentenceTransformers()
        texts = ["hello world", "testing embeddings"]
        embeddings = provider.embed(texts)

        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        assert all(isinstance(val, float) for emb in embeddings for val in emb)

    def test_embed_query(self):
        provider = EmbeddingSentenceTransformers()
        texts = ["what is the meaning of life?"]
        embeddings = provider.embed(texts, EmbedInputType.QUERY)

        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0

    def test_embed_empty_input(self):
        provider = EmbeddingSentenceTransformers()
        embeddings = provider.embed([])

        assert embeddings == []

    def test_embed_single_string_raises(self):
        provider = EmbeddingSentenceTransformers()
        with pytest.raises(TypeError):
            provider.embed("single text input")

    def test_embed_empty_string_raises(self):
        provider = EmbeddingSentenceTransformers()
        with pytest.raises(ValueError, match="empty strings"):
            provider.embed(["hello", "", "world"])

    def test_batch_size_handling(self):
        provider = EmbeddingSentenceTransformers(batch_size=2)
        texts = ["text1", "text2", "text3", "text4", "text5"]

        assert provider.batch_size == 2
        embeddings = provider.embed(texts)
        assert len(embeddings) == len(texts)

    def test_get_config(self):
        provider = EmbeddingSentenceTransformers(
            model="all-MiniLM-L6-v2", device="cpu", batch_size=32
        )
        config = provider.get_config()

        assert config == {
            "type": "EmbeddingSentenceTransformers",
            "model": "all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
        }

    def test_from_config(self):
        config = {
            "type": "EmbeddingSentenceTransformers",
            "model": "all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
        }
        provider = EmbeddingSentenceTransformers.from_config(config)

        assert provider.model == "all-MiniLM-L6-v2"
        assert provider.device == "cpu"
        assert provider.batch_size == 32

    def test_prompts_prepend_prefix(self):
        prompts = {
            EmbedInputType.QUERY: "search_query: ",
            EmbedInputType.DOCUMENT: "search_document: ",
        }
        provider = EmbeddingSentenceTransformers(prompts=prompts)

        doc_emb = provider.embed(["hello world"], EmbedInputType.DOCUMENT)
        query_emb = provider.embed(["hello world"], EmbedInputType.QUERY)

        # Different prefixes should produce different embeddings
        assert doc_emb[0] != query_emb[0]

    def test_prompts_config_roundtrip(self):
        prompts = {
            EmbedInputType.QUERY: "search_query: ",
            EmbedInputType.DOCUMENT: "search_document: ",
        }
        provider = EmbeddingSentenceTransformers(prompts=prompts)
        config = provider.get_config()

        assert config["prompts"] == {
            "query": "search_query: ",
            "document": "search_document: ",
        }

        restored = EmbeddingSentenceTransformers.from_config(config)
        assert restored.prompts == prompts

    def test_no_prompts_omitted_from_config(self):
        provider = EmbeddingSentenceTransformers()
        config = provider.get_config()
        assert "prompts" not in config

    def test_registry_roundtrip(self):
        provider = EmbeddingSentenceTransformers(model="all-MiniLM-L6-v2")
        config = provider.get_config()
        restored = embedding_from_config(config)

        assert isinstance(restored, EmbeddingSentenceTransformers)
        assert restored.model == provider.model
        assert restored.batch_size == provider.batch_size
