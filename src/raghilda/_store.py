from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Sequence

from .chunk import RetrievedChunk
from .document import Document, MarkdownDocument


@dataclass(frozen=True)
class WriteResult:
    action: Literal["inserted", "replaced", "skipped"]
    document: MarkdownDocument
    replaced_document: MarkdownDocument | None = None


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
    def upsert(
        self,
        document: Document,
        *,
        skip_if_unchanged: bool = True,
    ) -> WriteResult:
        """Upsert a document into the store.

        Insert or replace a document in the store.

        Parameters
        ----------
        document
            The document to upsert.
        skip_if_unchanged
            If True (default), skip the write when the existing document
            for the same identity key already has identical content and
            chunk metadata. This helps avoid unnecessary embedding work.
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
