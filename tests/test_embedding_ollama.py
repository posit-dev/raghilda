import json
import socket
from functools import partial

import httpx2
import pytest
from openai import OpenAI

from raghilda.chunker import MarkdownChunker
from raghilda.document import MarkdownDocument
from raghilda.embedding import (
    EmbeddingOllama,
    EmbeddingOpenAI,
    EmbedInputType,
    embedding_from_config,
)
from raghilda.store import DuckDBStore


def _ollama_is_running() -> bool:
    try:
        with socket.create_connection(("localhost", 11434), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.fixture
def embedding_requests(monkeypatch, request):
    requests = []
    expected_api_key = getattr(request, "param", None)

    def handle(request):
        if (
            expected_api_key is not None
            and request.headers.get("authorization") != f"Bearer {expected_api_key}"
        ):
            return httpx2.Response(401, json={"error": {"message": "Invalid API key"}})
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": payload["model"],
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.1] * 8}
                    for i in range(len(payload["input"]))
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    with httpx2.Client(transport=httpx2.MockTransport(handle)) as client:
        monkeypatch.setattr(
            "raghilda._embedding.OpenAI", partial(OpenAI, http_client=client)
        )
        yield requests


class TestEmbeddingOllama:
    def test_embedding_ollama_init(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
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

    @pytest.mark.parametrize("embedding_requests", ["proxy-key"], indirect=True)
    @pytest.mark.parametrize("api_key", [None, "proxy-key"])
    def test_proxy_authentication(self, api_key, embedding_requests, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
        monkeypatch.setenv(
            "OLLAMA_API_KEY", "proxy-key" if api_key is None else "unused"
        )
        provider = EmbeddingOllama(
            base_url="https://ollama-proxy.example/v1", api_key=api_key
        )

        assert provider.embed(["Text to embed."]) == [[0.1] * 8]
        assert len(embedding_requests) == 1

    @pytest.mark.parametrize("api_key", [None, "proxy-key"])
    def test_embedding_ollama_get_config(self, api_key, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "env-proxy-key")
        provider = EmbeddingOllama(model="mxbai-embed-large", api_key=api_key)
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

    def test_prompts_apply_to_each_batch(self, embedding_requests):
        provider = EmbeddingOllama(
            batch_size=2,
            prompts={
                EmbedInputType.DOCUMENT: "title: none | text: ",
                EmbedInputType.QUERY: "task: search result | query: ",
            },
        )
        texts = ["first", "second", "third"]

        documents = provider.embed(texts)
        queries = provider.embed(texts, EmbedInputType.QUERY)

        assert [request["input"] for request in embedding_requests] == [
            ["title: none | text: first", "title: none | text: second"],
            ["title: none | text: third"],
            [
                "task: search result | query: first",
                "task: search result | query: second",
            ],
            ["task: search result | query: third"],
        ]
        assert documents == queries == [[0.1] * 8] * 3
        assert texts == ["first", "second", "third"]

    @pytest.mark.parametrize("prompts", [None, {}, {EmbedInputType.QUERY: "query: "}])
    def test_documents_without_prompt_are_unchanged(self, prompts, embedding_requests):
        provider = EmbeddingOllama(prompts=prompts)

        provider.embed(["first", "second"])

        assert embedding_requests[0]["input"] == ["first", "second"]

    @pytest.mark.parametrize(
        ("texts", "error"),
        [("single text", TypeError), (["valid", ""], ValueError), (["  "], ValueError)],
    )
    def test_prompts_preserve_input_validation(self, texts, error, embedding_requests):
        provider = EmbeddingOllama(prompts={EmbedInputType.DOCUMENT: "document: "})

        with pytest.raises(error):
            provider.embed(texts)

        assert embedding_requests == []

    def test_empty_input_with_prompts(self, embedding_requests):
        provider = EmbeddingOllama(prompts={EmbedInputType.DOCUMENT: "document: "})

        assert provider.embed([]) == []
        assert embedding_requests == []

    def test_prompts_json_config_roundtrip(self, embedding_requests):
        prompts = {
            EmbedInputType.QUERY: "query: ",
            EmbedInputType.DOCUMENT: "document: ",
        }
        original = EmbeddingOllama(prompts=prompts)
        config = json.loads(json.dumps(original.get_config()))

        assert config["prompts"] == {"query": "query: ", "document": "document: "}
        restored = embedding_from_config(config)
        restored.embed(["stored text"])
        restored.embed(["search text"], EmbedInputType.QUERY)

        assert [request["input"] for request in embedding_requests] == [
            ["document: stored text"],
            ["query: search text"],
        ]

    def test_store_reconnect_preserves_prompts(
        self, embedding_requests, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        location = tmp_path / "ollama-prompts.db"
        store = DuckDBStore.create(
            location=location,
            embed=EmbeddingOllama(
                prompts={
                    EmbedInputType.DOCUMENT: "document: ",
                    EmbedInputType.QUERY: "query: ",
                }
            ),
        )
        store.con.close()
        restored = DuckDBStore.connect(location)
        embedding_requests.clear()

        document = MarkdownDocument(content="Text to search.", origin="test")
        restored.upsert(MarkdownChunker().chunk(document))
        results = restored.retrieve_vss("search text", top_k=1)

        assert len(results) == 1
        assert results[0].text == document.content
        assert embedding_requests[0]["input"][0].startswith("document: ")
        assert embedding_requests[1]["input"] == ["query: search text"]
        restored.con.close()

    def test_chroma_reconnect_preserves_prompts(
        self, embedding_requests, monkeypatch, tmp_path
    ):
        pytest.importorskip("chromadb")
        from raghilda.store import ChromaDBStore

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        location = tmp_path / "ollama-prompts-chroma"
        ChromaDBStore.create(
            location=location,
            name="ollama_prompts",
            embed=EmbeddingOllama(
                prompts={
                    EmbedInputType.DOCUMENT: "document: ",
                    EmbedInputType.QUERY: "query: ",
                }
            ),
        )
        restored = ChromaDBStore.connect(location=location, name="ollama_prompts")
        embedding_requests.clear()

        document = MarkdownDocument(content="Text to search.", origin="test")
        restored.upsert(MarkdownChunker().chunk(document))
        results = restored.retrieve("search text", top_k=1)

        assert len(results) == 1
        assert results[0].text == document.content
        assert embedding_requests[0]["input"][0].startswith("document: ")
        assert embedding_requests[1]["input"] == ["query: search text"]

    @pytest.mark.parametrize("embedding_requests", ["proxy-key"], indirect=True)
    def test_store_reconnect_authenticates_proxy(
        self, embedding_requests, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("OLLAMA_API_KEY", "unused-at-creation")
        location = tmp_path / "ollama-proxy.db"
        store = DuckDBStore.create(
            location=location,
            embed=EmbeddingOllama(
                base_url="https://ollama-proxy.example/v1", api_key="proxy-key"
            ),
        )
        document = MarkdownDocument(content="Text to search.", origin="test")
        store.upsert(MarkdownChunker().chunk(document))
        store.con.close()

        monkeypatch.setenv("OLLAMA_API_KEY", "proxy-key")
        restored = DuckDBStore.connect(location)
        try:
            results = restored.retrieve_vss("search text", top_k=1)
            assert len(results) == 1
            assert results[0].text == document.content
            assert embedding_requests[-1]["input"] == ["search text"]
        finally:
            restored.con.close()

    @pytest.mark.parametrize("embedding_requests", ["proxy-key"], indirect=True)
    def test_chroma_reconnect_authenticates_proxy(
        self, embedding_requests, monkeypatch, tmp_path
    ):
        pytest.importorskip("chromadb")
        from raghilda.store import ChromaDBStore

        monkeypatch.setenv("OLLAMA_API_KEY", "unused-at-creation")
        location = tmp_path / "ollama-proxy-chroma"
        store = ChromaDBStore.create(
            location=location,
            name="ollama_proxy",
            embed=EmbeddingOllama(
                base_url="https://ollama-proxy.example/v1", api_key="proxy-key"
            ),
        )
        document = MarkdownDocument(content="Text to search.", origin="test")
        store.upsert(MarkdownChunker().chunk(document))

        monkeypatch.setenv("OLLAMA_API_KEY", "proxy-key")
        restored = ChromaDBStore.connect(location=location, name="ollama_proxy")
        results = restored.retrieve("search text", top_k=1)

        assert len(results) == 1
        assert results[0].text == document.content
        assert embedding_requests[-1]["input"] == ["search text"]

    def test_embedding_ollama_store_reconnect_without_openai_key(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
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
