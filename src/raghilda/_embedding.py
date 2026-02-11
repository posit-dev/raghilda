from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Optional, Protocol, Sequence, Type, TypeVar, runtime_checkable

from openai import OpenAI

# Global registry for embedding providers
_EMBEDDING_REGISTRY: dict[str, Type["EmbeddingProvider"]] = {}


@runtime_checkable
class ChromaConvertible(Protocol):
    """Protocol for embedding providers that can convert to ChromaDB functions.

    Implement this protocol to enable automatic conversion of raghilda embedding
    providers to ChromaDB's built-in embedding functions. This provides cross-language
    compatibility since ChromaDB's built-in functions work in both Python and TypeScript.

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.embedding import EmbeddingOpenAI
    from raghilda.store import ChromaDBStore

    provider = EmbeddingOpenAI(model="text-embedding-3-small")

    # Automatic conversion - provider.to_chroma() is called internally
    store = ChromaDBStore.create(location="my_store", embed=provider)
    ```
    """

    def to_chroma(self) -> Any:
        """Convert to a ChromaDB embedding function.

        Returns
        -------
        EmbeddingFunction
            A ChromaDB-compatible embedding function instance.
        """
        ...


T = TypeVar("T", bound="EmbeddingProvider")


def register_embedding_provider(name: str):
    """
    Decorator to register an embedding provider class.

    Registered providers can be automatically restored when connecting to a
    database that was created with that provider.

    Parameters
    ----------
    name
        The name to register the provider under. This should be unique.

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.embedding import EmbeddingProvider, register_embedding_provider

    @register_embedding_provider("MyCustomEmbedding")
    class MyCustomEmbedding(EmbeddingProvider):
        def __init__(self, model: str = "default"):
            self.model = model

        def embed(self, x, input_type=None):
            ...

        def get_config(self):
            return {"type": "MyCustomEmbedding", "model": self.model}

        @classmethod
        def from_config(cls, config):
            return cls(model=config.get("model", "default"))
    ```
    """

    def decorator(cls: Type[T]) -> Type[T]:
        _EMBEDDING_REGISTRY[name] = cls
        return cls

    return decorator


def embedding_from_config(config: dict[str, Any]) -> "EmbeddingProvider":
    """
    Create an embedding provider from a configuration dict.

    Parameters
    ----------
    config
        Configuration dict with a "type" key specifying the provider class name.

    Returns
    -------
    EmbeddingProvider
        An instance of the specified embedding provider.

    Raises
    ------
    ValueError
        If the provider type is not found in the registry.
    """
    provider_type = config.get("type")
    if provider_type is None:
        raise ValueError("Config must contain a 'type' key")

    if provider_type not in _EMBEDDING_REGISTRY:
        registered = ", ".join(_EMBEDDING_REGISTRY.keys())
        raise ValueError(
            f"Unknown embedding provider type: '{provider_type}'. "
            f"Registered providers: {registered}. "
            "You can pass `embed=` to connect() to provide the provider manually."
        )

    cls = _EMBEDDING_REGISTRY[provider_type]
    return cls.from_config(config)


class EmbedInputType(StrEnum):
    """
    Specifies the type of input being embedded.

    Some embedding models (e.g., Cohere) produce different embeddings for queries
    vs documents to optimize retrieval performance.
    """

    QUERY = "query"
    """Input is a search query."""

    DOCUMENT = "document"
    """Input is a document to be indexed."""


class EmbeddingProvider(ABC):
    """
    Interface for embedding function providers.

    To create a custom embedding provider:

    1. Subclass `EmbeddingProvider` and implement `embed()`, `get_config()`, and `from_config()`
    2. Register it with `@register_embedding_provider("MyProvider")`

    Registered providers are automatically restored when connecting to a database
    that was created with that provider.

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.embedding import EmbeddingProvider, register_embedding_provider

    @register_embedding_provider("MyCustomEmbedding")
    class MyCustomEmbedding(EmbeddingProvider):
        def __init__(self, model: str = "default", api_key: str | None = None):
            self.model = model
            self.api_key = api_key
            # Initialize your embedding client here

        def embed(self, x, input_type=None):
            # Return list of embedding vectors
            ...

        def get_config(self):
            # Return config dict (exclude sensitive values like api_key)
            return {"type": "MyCustomEmbedding", "model": self.model}

        @classmethod
        def from_config(cls, config):
            return cls(model=config.get("model", "default"))
    ```
    """

    @abstractmethod
    def embed(
        self,
        x: Sequence[str],
        input_type: EmbedInputType = EmbedInputType.DOCUMENT,
    ) -> Sequence[Sequence[float]]:
        """
        Generate embeddings for a sequence of texts.

        Parameters
        ----------
        x
            A sequence of texts to generate embeddings for.
        input_type
            The type of input being embedded. Some models (e.g., Cohere) produce
            different embeddings for queries vs documents. Default is DOCUMENT.

        Returns
        -------
        :
            A sequence of embeddings (the same length as `x`), where each embedding is
            a sequence of floats.
        """
        NotImplementedError("embed method is not implemented")

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Get the configuration dict for this provider.

        The config should contain all parameters needed to recreate the provider,
        except for sensitive values like API keys. It must include a "type" key
        with the registered name of the provider.

        Returns
        -------
        dict
            Configuration dict that can be passed to `from_config()`.
        """
        NotImplementedError("get_config method is not implemented")

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "EmbeddingProvider":
        """
        Create a provider instance from a configuration dict.

        Parameters
        ----------
        config
            Configuration dict from `get_config()`.

        Returns
        -------
        EmbeddingProvider
            A new instance of the provider.
        """
        NotImplementedError("from_config method is not implemented")


@register_embedding_provider("EmbeddingOpenAI")
class EmbeddingOpenAI(EmbeddingProvider):
    """
    Creates an embedding function provider backed by OpenAI's embedding models
    Implements the [EmbeddingProvider](`raghilda.EmbeddingProvider`) interface.

    Parameters
    ----------
    model
        The OpenAI embedding model to use. Default is "text-embedding-3-small"
    base_url
        The base URL for the OpenAI API. Default is "https://api.openai.com/v1".
    api_key
        The API key for authenticating with OpenAI. If None, it will use the
        OPENAI_API_KEY environment variable if set.
    batch_size
        The number of texts to process in each batch when calling the API.

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.embedding import EmbeddingOpenAI

    provider = EmbeddingOpenAI(model="text-embedding-3-small")
    embeddings = provider.embed(["hello world", "testing embeddings"])
    print(len(embeddings))
    print(len(embeddings[0]))  # Dimension of the embedding
    print(embeddings[0][:10])  # The embedding vector
    ```
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        batch_size: int = 20,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.batch_size = batch_size

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "EmbeddingOpenAI",
            "model": self.model,
            "base_url": self.base_url,
            "batch_size": self.batch_size,
            # api_key intentionally omitted for security
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EmbeddingOpenAI":
        return cls(
            model=config.get("model", "text-embedding-3-small"),
            base_url=config.get("base_url", "https://api.openai.com/v1"),
            batch_size=config.get("batch_size", 20),
        )

    def embed(
        self,
        x: Sequence[str],
        input_type: EmbedInputType = EmbedInputType.DOCUMENT,
    ) -> Sequence[Sequence[float]]:
        # Note: OpenAI doesn't differentiate between query and document embeddings,
        # so input_type is accepted but ignored for API compatibility.
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

        result: list[Sequence[float]] = []
        for i in range(0, len(x), self.batch_size):
            data = x[i : i + self.batch_size]
            embedding = self._embed_with_retry(data)
            result.extend([res.embedding for res in embedding.data])

        return result

    def _embed_with_retry(
        self, data: Sequence[str], max_retries: int = 20, max_seconds: float = 180
    ):
        """Call embeddings API with retry on rate limit errors."""
        import time

        start_time = time.time()
        last_error = None

        for attempt in range(max_retries):
            try:
                return self.client.embeddings.create(input=list(data), model=self.model)
            except Exception as e:
                # Only retry on 429 rate limit errors
                status_code = getattr(e, "status_code", None)
                if status_code != 429:
                    raise

                last_error = e
                elapsed = time.time() - start_time
                if elapsed >= max_seconds:
                    break

                wait_time = self._get_retry_after(e)
                wait_time = min(wait_time, max_seconds - elapsed)
                wait_time = max(wait_time, 0.1)  # At least 100ms
                time.sleep(wait_time)

        if last_error:
            raise last_error

    def _get_retry_after(self, error) -> float:
        """Extract retry wait time from rate limit error headers."""
        import re

        # Try to get headers from the response
        headers = {}
        if hasattr(error, "response") and error.response is not None:
            headers = dict(error.response.headers)

        # Check for reset times in headers
        reset_tokens = headers.get("x-ratelimit-reset-tokens")
        reset_requests = headers.get("x-ratelimit-reset-requests")

        wait_times = []
        if reset_tokens:
            wait_times.append(self._parse_duration(reset_tokens))
        if reset_requests:
            wait_times.append(self._parse_duration(reset_requests))

        if wait_times:
            # Use the longer wait time, divided by 4 to retry earlier
            return max(wait_times) / 4

        # Fallback: parse from error message
        match = re.search(r"try again in (\d+)(ms|s)", str(error))
        if match:
            value, unit = match.groups()
            wait_time = float(value) / 1000 if unit == "ms" else float(value)
            return max(wait_time, 0.1)

        # Default fallback
        return 1.0

    def _parse_duration(self, duration_str: str) -> float:
        """Parse duration string like '1m0.612s' or '500ms' to seconds."""
        import re

        total = 0.0

        # Hours
        match = re.search(r"(\d+)h", duration_str)
        if match:
            total += int(match.group(1)) * 3600

        # Minutes
        match = re.search(r"(\d+)m(?!s)", duration_str)
        if match:
            total += int(match.group(1)) * 60

        # Seconds (including decimal)
        match = re.search(r"([\d.]+)s", duration_str)
        if match:
            total += float(match.group(1))

        # Milliseconds
        match = re.search(r"(\d+)ms", duration_str)
        if match:
            total += int(match.group(1)) / 1000

        return total if total > 0 else 1.0

    def to_chroma(self) -> Any:
        """Convert to a ChromaDB OpenAIEmbeddingFunction.

        Returns
        -------
        OpenAIEmbeddingFunction
            A ChromaDB-compatible embedding function using the same model and settings.
        """
        import os

        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        # Check if we can use environment variables for persistence
        if os.getenv("CHROMA_OPENAI_API_KEY"):
            # ChromaDB's preferred env var is set
            return OpenAIEmbeddingFunction(
                model_name=self.model,
                api_base=self.base_url,
            )
        elif self.api_key is None or self.api_key == os.getenv("OPENAI_API_KEY"):
            # Using standard OpenAI env var
            return OpenAIEmbeddingFunction(
                model_name=self.model,
                api_base=self.base_url,
                api_key_env_var="OPENAI_API_KEY",
            )
        else:
            # Custom api_key passed - won't persist but will work
            return OpenAIEmbeddingFunction(
                api_key=self.api_key,
                model_name=self.model,
                api_base=self.base_url,
            )


@register_embedding_provider("EmbeddingCohere")
class EmbeddingCohere(EmbeddingProvider):
    """
    Creates an embedding function provider backed by Cohere's embedding models.
    Implements the [EmbeddingProvider](`raghilda.EmbeddingProvider`) interface.

    Cohere's embedding models produce different embeddings for queries vs documents
    to optimize retrieval performance. Use `input_type=EmbedInputType.QUERY` when
    embedding search queries and `input_type=EmbedInputType.DOCUMENT` (default)
    when embedding documents for indexing.

    Parameters
    ----------
    model
        The Cohere embedding model to use. Default is "embed-english-v3.0".
    api_key
        The API key for authenticating with Cohere. If None, it will use the
        CO_API_KEY environment variable if set.
    batch_size
        The number of texts to process in each batch when calling the API.
        Cohere supports up to 96 texts per request.

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.embedding import EmbeddingCohere, EmbedInputType

    provider = EmbeddingCohere(model="embed-english-v3.0")

    # Embed documents for indexing
    doc_embeddings = provider.embed(
        ["Hello world", "Testing embeddings"],
        input_type=EmbedInputType.DOCUMENT
    )

    # Embed a query for search
    query_embedding = provider.embed(
        ["How do I test embeddings?"],
        input_type=EmbedInputType.QUERY
    )
    ```
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        batch_size: int = 96,
    ) -> None:
        import cohere

        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size

        self.client = cohere.Client(api_key=self.api_key)

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "EmbeddingCohere",
            "model": self.model,
            "batch_size": self.batch_size,
            # api_key intentionally omitted for security
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EmbeddingCohere":
        return cls(
            model=config.get("model", "embed-english-v3.0"),
            batch_size=config.get("batch_size", 96),
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

        # Map our enum to Cohere's input_type values
        cohere_input_type = (
            "search_query" if input_type == EmbedInputType.QUERY else "search_document"
        )

        result: list[Sequence[float]] = []
        for i in range(0, len(x), self.batch_size):
            data = list(x[i : i + self.batch_size])
            response = self.client.embed(
                texts=data,
                model=self.model,
                input_type=cohere_input_type,
                embedding_types=["float"],
            )
            embeddings = response.embeddings
            # Cohere SDK type stubs incorrectly type embeddings as List[List[float]]
            # but it's actually EmbedByTypeResponseEmbeddings when embedding_types is used
            if hasattr(embeddings, "float_") and embeddings.float_ is not None:  # type: ignore[union-attr]
                result.extend(embeddings.float_)  # type: ignore[union-attr]

        return result

    def to_chroma(self) -> Any:
        """Convert to a ChromaDB CohereEmbeddingFunction.

        Returns
        -------
        CohereEmbeddingFunction
            A ChromaDB-compatible embedding function using the same model and settings.
        """
        import os

        from chromadb.utils.embedding_functions import CohereEmbeddingFunction

        # Check if we can use environment variables for persistence
        if os.getenv("CHROMA_COHERE_API_KEY"):
            # ChromaDB's preferred env var is set
            return CohereEmbeddingFunction(
                model_name=self.model,
            )
        elif os.getenv("COHERE_API_KEY"):
            # ChromaDB also checks COHERE_API_KEY
            return CohereEmbeddingFunction(
                model_name=self.model,
                api_key_env_var="COHERE_API_KEY",
            )
        elif self.api_key is None or self.api_key == os.getenv("CO_API_KEY"):
            # Using raghilda's default env var
            return CohereEmbeddingFunction(
                model_name=self.model,
                api_key_env_var="CO_API_KEY",
            )
        else:
            # Custom api_key passed - won't persist but will work
            return CohereEmbeddingFunction(
                api_key=self.api_key,
                model_name=self.model,
            )
