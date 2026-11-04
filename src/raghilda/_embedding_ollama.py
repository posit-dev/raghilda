from collections.abc import Sequence
from typing import Any

from ._embedding import EmbeddingOpenAI, EmbedInputType, register_embedding_provider


@register_embedding_provider("EmbeddingOllama")
class EmbeddingOllama(EmbeddingOpenAI):
    """
    Creates an embedding function provider backed by a local Ollama server.
    Implements the [EmbeddingProvider](`raghilda.EmbeddingProvider`) interface.

    This provider uses Ollama's OpenAI-compatible API with defaults for a local
    server. Stores created with it can be reconnected without setting
    `OPENAI_API_KEY`.

    Pull an embedding model before using it (e.g., `ollama pull embeddinggemma-2`).
    Browse available embedding models at
    [ollama.com/search](https://ollama.com/search?c=embedding).

    Parameters
    ----------
    model
        The Ollama embedding model to use. Default is "embeddinggemma-2:latest".
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
    prompts
        Optional prefix strings keyed by [EmbedInputType](`raghilda.EmbedInputType`).
        The matching prefix is added to each text before embedding. Use the query
        and document prefixes recommended by your model. No prefixes are added
        by default. These settings are saved with the store.

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.embedding import EmbeddingOllama, EmbedInputType

    provider = EmbeddingOllama(
        model="embeddinggemma-2:latest",
        prompts={
            EmbedInputType.QUERY: "task: search result | query: ",
            EmbedInputType.DOCUMENT: "title: none | text: ",
        },
    )
    embeddings = provider.embed(["hello world", "testing embeddings"])
    print(len(embeddings))
    print(len(embeddings[0]))  # Dimension of the embedding
    ```
    """

    def __init__(
        self,
        model: str = "embeddinggemma-2:latest",
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        batch_size: int = 20,
        prompts: dict[EmbedInputType, str] | None = None,
    ) -> None:
        # The OpenAI client requires a non-empty key, but Ollama ignores it
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or "ollama",
            batch_size=batch_size,
        )
        self.prompts = prompts

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "type": "EmbeddingOllama",
            "model": self.model,
            "base_url": self.base_url,
            "batch_size": self.batch_size,
            # api_key intentionally omitted for security
        }
        if self.prompts is not None:
            config["prompts"] = {k.value: v for k, v in self.prompts.items()}
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EmbeddingOllama":
        prompts = None
        if "prompts" in config:
            prompts = {EmbedInputType(k): v for k, v in config["prompts"].items()}
        return cls(
            model=config.get("model", "embeddinggemma-2:latest"),
            base_url=config.get("base_url", "http://localhost:11434/v1"),
            batch_size=config.get("batch_size", 20),
            prompts=prompts,
        )

    def _prepare_inputs(
        self, x: Sequence[str], input_type: EmbedInputType
    ) -> Sequence[str]:
        if self.prompts and input_type in self.prompts:
            prefix = self.prompts[input_type]
            return [prefix + text for text in x]
        return x
