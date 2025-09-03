from abc import ABC, abstractmethod
from typing import Optional, Sequence

from openai import OpenAI


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, x: Sequence[str]) -> Sequence[Sequence[float]]:
        NotImplementedError("embed method is not implemented")


class EmbeddingOpenAI(EmbeddingProvider):
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
        result = []
        for i in range(0, len(x), self.batch_size):
            data = x[i : i + self.batch_size]
            embedding = self.client.embeddings.create(
                input=list(data), model=self.model
            )
            result.extend([res.embedding for res in embedding.data])

        return result
