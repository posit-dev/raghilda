from typing import Any

from ._embedding import EmbeddingOpenAI, register_embedding_provider


@register_embedding_provider("EmbeddingOllama")
class EmbeddingOllama(EmbeddingOpenAI):
    """
    Creates an embedding function provider backed by a local Ollama server.
    Implements the [EmbeddingProvider](`raghilda.EmbeddingProvider`) interface.

    Ollama exposes an OpenAI-compatible API, so this provider is a thin wrapper
    around [EmbeddingOpenAI](`raghilda.embedding.EmbeddingOpenAI`) with defaults
    suited to a local Ollama install. Using it instead of `EmbeddingOpenAI` with a
    custom `base_url` means stores can be reconnected without an `OPENAI_API_KEY`
    in the environment.

    Pull an embedding model before using it (e.g., `ollama pull nomic-embed-text`).
    Browse available embedding models at
    [ollama.com/search](https://ollama.com/search?c=embedding).

    Parameters
    ----------
    model
        The Ollama embedding model to use. Default is "nomic-embed-text".
    base_url
        The base URL for the Ollama server's OpenAI-compatible API. Default is
        "http://localhost:11434/v1". Change this when Ollama runs on another
        machine or port.
    api_key
        The API key sent with each request. Ollama does not check it, so this is
        only needed when the server sits behind a proxy that requires one. If None,
        a placeholder value is used.
    batch_size
        The number of texts to process in each batch when calling the API.

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.embedding import EmbeddingOllama

    provider = EmbeddingOllama(model="nomic-embed-text")
    embeddings = provider.embed(["hello world", "testing embeddings"])
    print(len(embeddings))
    print(len(embeddings[0]))  # Dimension of the embedding
    ```
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        batch_size: int = 20,
    ) -> None:
        # The OpenAI client requires a non-empty key, but Ollama ignores it
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or "ollama",
            batch_size=batch_size,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "EmbeddingOllama",
            "model": self.model,
            "base_url": self.base_url,
            "batch_size": self.batch_size,
            # api_key intentionally omitted for security
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EmbeddingOllama":
        return cls(
            model=config.get("model", "nomic-embed-text"),
            base_url=config.get("base_url", "http://localhost:11434/v1"),
            batch_size=config.get("batch_size", 20),
        )
