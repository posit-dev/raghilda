from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Optional, Sequence

from openai import OpenAI


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
            embedding = self.client.embeddings.create(
                input=list(data), model=self.model
            )
            result.extend([res.embedding for res in embedding.data])

        return result


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
