import os
from typing import Any, Optional, Sequence

from openai import OpenAI

from ._embedding import (
    EmbeddingProvider,
    EmbedInputType,
    register_embedding_provider,
)


@register_embedding_provider("EmbeddingNVIDIA")
class EmbeddingNVIDIA(EmbeddingProvider):
    """
    Creates an embedding function provider backed by NVIDIA's NIM embedding models.
    Implements the [EmbeddingProvider](`raghilda.EmbeddingProvider`) interface.

    NVIDIA's embedding API is compatible with the OpenAI client library. It supports
    differentiated query vs document embeddings via the `input_type` parameter.

    Browse available models and get API keys at
    [NVIDIA NIM](https://build.nvidia.com/nvidia).

    NVIDIA also provides Docker images for self-hosting NIM models. When running a
    model locally or on your own infrastructure, set the `base_url` parameter to
    point to your self-hosted endpoint (e.g., `"http://localhost:8000/v1"`).

    Parameters
    ----------
    model
        The NVIDIA embedding model to use. Default is "nvidia/llama-nemotron-embed-1b-v2".
    base_url
        The base URL for the NVIDIA API. Default is "https://integrate.api.nvidia.com/v1".
        Change this to point to a self-hosted NIM container
        (e.g., `"http://localhost:8000/v1"`).
    api_key
        The API key for authenticating with NVIDIA. If None, it will use the
        NVIDIA_API_KEY environment variable if set. Not required when using a
        self-hosted NIM container.
    batch_size
        The number of texts to process in each batch when calling the API.
    truncate
        Truncation strategy. Default is "NONE". Set to "END" to truncate inputs
        that exceed the model's max token length.

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.embedding import EmbeddingNVIDIA, EmbedInputType

    # Using NVIDIA's hosted API
    provider = EmbeddingNVIDIA(model="nvidia/llama-nemotron-embed-1b-v2")

    # Or using a self-hosted NIM container
    provider = EmbeddingNVIDIA(
        model="nvidia/llama-nemotron-embed-1b-v2",
        base_url="http://localhost:8000/v1",
    )

    # Embed documents for indexing
    doc_embeddings = provider.embed(
        ["Hello world", "Testing embeddings"],
        input_type=EmbedInputType.DOCUMENT,
    )

    # Embed a query for search
    query_embedding = provider.embed(
        ["How do I test embeddings?"],
        input_type=EmbedInputType.QUERY,
    )
    ```
    """

    def __init__(
        self,
        model: str = "nvidia/llama-nemotron-embed-1b-v2",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        api_key: Optional[str] = None,
        batch_size: int = 20,
        truncate: str = "NONE",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.truncate = truncate

        self.client = OpenAI(
            api_key=self.api_key or os.environ.get("NVIDIA_API_KEY"),
            base_url=self.base_url,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "EmbeddingNVIDIA",
            "model": self.model,
            "base_url": self.base_url,
            "batch_size": self.batch_size,
            "truncate": self.truncate,
            # api_key intentionally omitted for security
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EmbeddingNVIDIA":
        return cls(
            model=config.get("model", "nvidia/llama-nemotron-embed-1b-v2"),
            base_url=config.get("base_url", "https://integrate.api.nvidia.com/v1"),
            batch_size=config.get("batch_size", 20),
            truncate=config.get("truncate", "NONE"),
        )

    def embed(
        self,
        x: Sequence[str],
        input_type: EmbedInputType = EmbedInputType.DOCUMENT,
    ) -> Sequence[Sequence[float]]:
        if isinstance(x, str):
            raise TypeError("Input must be a sequence of strings, not a single string.")

        if len(x) == 0:
            return []

        # Check for empty strings
        empty_indices = [i for i, text in enumerate(x) if not text or not text.strip()]
        if empty_indices:
            raise ValueError(
                f"Input contains empty strings at indices: {empty_indices}. "
                "Empty strings cannot be embedded."
            )

        # Map our enum to NVIDIA's input_type values
        nvidia_input_type = (
            "query" if input_type == EmbedInputType.QUERY else "passage"
        )

        result: list[Sequence[float]] = []
        for i in range(0, len(x), self.batch_size):
            data = list(x[i : i + self.batch_size])
            embedding = self._embed_with_retry(data, nvidia_input_type)
            result.extend([res.embedding for res in embedding.data])

        return result

    def _embed_with_retry(
        self,
        data: list[str],
        nvidia_input_type: str,
        max_retries: int = 20,
        max_seconds: float = 180,
    ):
        """Call NVIDIA embeddings API with retry on rate limit errors."""
        import time

        start_time = time.time()
        last_error = None
        wait_time = 1.0

        for _attempt in range(max_retries):
            try:
                return self.client.embeddings.create(
                    input=data,
                    model=self.model,
                    encoding_format="float",
                    extra_body={
                        "input_type": nvidia_input_type,
                        "truncate": self.truncate,
                    },
                )
            except Exception as e:
                # Only retry on 429 rate limit errors
                status_code = getattr(e, "status_code", None)
                if status_code != 429:
                    raise

                last_error = e
                elapsed = time.time() - start_time
                if elapsed >= max_seconds:
                    break

                actual_wait = min(wait_time, max_seconds - elapsed)
                actual_wait = max(actual_wait, 0.1)
                time.sleep(actual_wait)
                wait_time = min(wait_time * 2, 60)

        if last_error:
            raise last_error

        raise RuntimeError("Unexpected state: no result and no error")
