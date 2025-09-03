from abc import ABC, abstractmethod
from typing import Optional, Sequence

from openai import OpenAI


class EmbeddingProvider(ABC):
    """
    Interface for embedding function providers.
    """

    @abstractmethod
    def embed(self, x: Sequence[str]) -> Sequence[Sequence[float]]:
        """
        Generate embeddings for a sequence of texts.

        Parameters
        ----------
        x
            A sequence of texts to generate embeddings for.

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
    Implements the [EmbeddingProvider](`ragnar.EmbeddingProvider`) interface.

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
    import os
    from ragnar import EmbeddingOpenAI
    if "OPENAI_API_KEY" in os.environ:
        provider = EmbeddingOpenAI(model="text-embedding-3-small")
        embeddings = provider.embed(["hello world", "testing embeddings"])
        print(embeddings)
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

    def embed(self, x: Sequence[str]):
        if isinstance(x, str):
            raise TypeError("Input must be a sequence of strings, not a single string.")

        result = []
        for i in range(0, len(x), self.batch_size):
            data = x[i : i + self.batch_size]
            embedding = self.client.embeddings.create(
                input=list(data), model=self.model
            )
            result.extend([res.embedding for res in embedding.data])

        return result
