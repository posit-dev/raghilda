import socket

import pytest

from raghilda.embedding import EmbeddingOllama, EmbeddingOpenAI, embedding_from_config
from raghilda.store import DuckDBStore


def _ollama_is_running() -> bool:
    try:
        with socket.create_connection(("localhost", 11434), timeout=1.0):
            return True
    except OSError:
        return False


class TestEmbeddingOllama:
    def test_embedding_ollama_init(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = EmbeddingOllama()

        assert isinstance(provider, EmbeddingOpenAI)
        assert provider.model == "embeddinggemma-2:latest"
        assert provider.base_url == "http://localhost:11434/v1"
        assert provider.api_key == "ollama"
        assert provider.batch_size == 20

    def test_embedding_ollama_init_with_params(self):
        provider = EmbeddingOllama(
            model="mxbai-embed-large",
            base_url="http://192.168.1.10:11434/v1",
            api_key="proxy-key",
            batch_size=10,
        )

        assert provider.model == "mxbai-embed-large"
        assert provider.base_url == "http://192.168.1.10:11434/v1"
        assert provider.api_key == "proxy-key"
        assert provider.batch_size == 10

    def test_embedding_ollama_get_config(self):
        provider = EmbeddingOllama(model="mxbai-embed-large", api_key="proxy-key")
        config = provider.get_config()

        assert config == {
            "type": "EmbeddingOllama",
            "model": "mxbai-embed-large",
            "base_url": "http://localhost:11434/v1",
            "batch_size": 20,
        }

    def test_embedding_ollama_registry_roundtrip(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        original = EmbeddingOllama(
            model="mxbai-embed-large",
            base_url="http://192.168.1.10:11434/v1",
            batch_size=10,
        )
        restored = embedding_from_config(original.get_config())

        assert isinstance(restored, EmbeddingOllama)
        assert restored.model == original.model
        assert restored.base_url == original.base_url
        assert restored.batch_size == original.batch_size

    def test_embedding_ollama_store_reconnect_without_openai_key(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # `create()` embeds a probe string to size the vector column
        monkeypatch.setattr(
            EmbeddingOllama, "embed", lambda self, x, input_type=None: [[0.0] * 8]
        )
        location = tmp_path / "ollama.db"

        DuckDBStore.create(location=location, embed=EmbeddingOllama())
        store = DuckDBStore.connect(location)

        assert isinstance(store.metadata.embed, EmbeddingOllama)
        assert store.metadata.embed.model == "embeddinggemma-2:latest"

    def test_embedding_ollama_embed_integration(self):
        if not _ollama_is_running():
            pytest.skip("Ollama is not running on localhost:11434")

        provider = EmbeddingOllama()
        try:
            embeddings = provider.embed(["hello world", "testing embeddings"])
        except Exception as e:
            if getattr(e, "status_code", None) == 404:
                pytest.skip("embeddinggemma-2:latest has not been pulled")
            raise

        assert len(embeddings) == 2
        assert all(len(emb) > 0 for emb in embeddings)
