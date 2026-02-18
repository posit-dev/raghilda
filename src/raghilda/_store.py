from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from .chunk import RetrievedChunk
from .document import Document


class BaseStore(ABC):
    """Abstract base class for vector stores.

    A store is responsible for storing documents and their embeddings,
    and retrieving relevant chunks based on similarity search.

    Subclasses must implement all abstract methods to provide a concrete
    storage backend (e.g., DuckDB, OpenAI Vector Store).
    """

    @staticmethod
    @abstractmethod
    def connect(*args, **kwargs) -> "BaseStore":
        """Connect to an existing store.

        Returns
        -------
        BaseStore
            A connected store instance.
        """
        pass

    @staticmethod
    @abstractmethod
    def create(*args, **kwargs) -> "BaseStore":
        """Create a new store.

        Returns
        -------
        BaseStore
            A newly created store instance.
        """
        pass

    @abstractmethod
    def insert(self, document: Document) -> None:
        """Insert a document into the store.

        The document will be chunked and embedded before storage.

        Parameters
        ----------
        document
            The document to insert.
        """
        pass

    @abstractmethod
    def retrieve(
        self, text: str, top_k: int, *args, **kwargs
    ) -> Sequence[RetrievedChunk]:
        """Retrieve the most similar chunks to the given text.

        Parameters
        ----------
        text
            The query text to search for.
        top_k
            The maximum number of chunks to return.

        Returns
        -------
        Sequence[RetrievedChunk]
            The most similar chunks, ordered by relevance.
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Count the number of documents in the store.

        Returns
        -------
        int
            The number of documents (not chunks) in the store.
        """
        pass
