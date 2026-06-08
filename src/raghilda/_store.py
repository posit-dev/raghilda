from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, CancelledError, ThreadPoolExecutor, wait
from dataclasses import dataclass
import threading
from typing import Any, Callable, Generic, Iterable, Literal, Sequence, TypeVar

from .chunk import RetrievedChunk
from .document import Document

TDocument = TypeVar("TDocument", bound=Document, covariant=True)
_RECENT_INGEST_ORIGIN_WINDOW = 10_000


@dataclass(frozen=True)
class WriteResult(Generic[TDocument]):
    action: Literal["inserted", "replaced", "skipped"]
    document: TDocument
    replaced_document: TDocument | None = None


@dataclass(frozen=True)
class IngestSummary:
    inserted: int
    replaced: int
    skipped: int


class BaseStore(ABC):
    """Abstract base class for vector stores.

    A store is responsible for storing documents and their embeddings,
    and retrieving relevant chunks based on similarity search.

    Subclasses must implement all abstract methods to provide a concrete
    storage backend:

    - :py:class:`raghilda.store.DuckDBStore`: local storage with embedding and BM25 search.
    - :py:class:`raghilda.store.ChromaDBStore`: local storage using ChromaDB.
    - :py:class:`raghilda.store.OpenAIStore`: hosted storage using OpenAI's Vector Store API.
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
    ) -> WriteResult[Document]:
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

    def _ingest_upsert(self, document: Document) -> WriteResult[Document]:
        return self.upsert(document)

    def ingest(
        self,
        documents: Iterable[Any],
        *,
        prepare: Callable[[Any], Document] | None = None,
        max_workers: int = 1,
    ) -> IngestSummary:
        """Prepare and upsert a stream of documents.

        Inputs are consumed lazily and submitted incrementally. After
        ``prepare`` is applied, recent non-empty string origins are checked for
        duplicates as the stream is consumed. Duplicate detection is best
        effort: a duplicate raises ``ValueError`` when encountered, after any
        writes already in flight complete. No rollback is attempted.

        Returns
        -------
        IngestSummary
            Aggregate counts for inserted, replaced, and skipped documents.
            Call ``upsert()`` directly when per-document ``WriteResult`` values
            are needed.
        """
        assert max_workers >= 1
        stop_event = threading.Event()
        recent_origins: dict[str, None] = {}
        recent_origins_lock = threading.Lock()

        def remember_origin(origin: str | None) -> None:
            if not isinstance(origin, str) or not origin:
                return
            with recent_origins_lock:
                if origin in recent_origins:
                    raise ValueError(f"Duplicate origin during ingest: {origin}")
                recent_origins[origin] = None
                if len(recent_origins) > _RECENT_INGEST_ORIGIN_WINDOW:
                    # dict preserves insertion order, so the first key is the oldest.
                    recent_origins.pop(next(iter(recent_origins)))

        def process_document(item: Any) -> WriteResult[Document]:
            if stop_event.is_set():
                raise CancelledError()
            document = prepare(item) if prepare is not None else item
            if stop_event.is_set():
                raise CancelledError()
            remember_origin(document.origin)
            if stop_event.is_set():
                raise CancelledError()
            return self._ingest_upsert(document)

        iterator = iter(documents)
        pending = set()
        inserted = 0
        replaced = 0
        skipped = 0
        exhausted = False
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            while not exhausted and len(pending) < max_workers:
                try:
                    document = next(iterator)
                except StopIteration:
                    exhausted = True
                    continue
                pending.add(executor.submit(process_document, document))

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                results = []
                cancelled_errors = []
                errors = []
                for future in done:
                    try:
                        results.append(future.result())
                    except CancelledError as exc:
                        cancelled_errors.append(exc)
                    except Exception as exc:
                        errors.append(exc)
                if errors:
                    raise errors[0]
                if cancelled_errors and not stop_event.is_set():
                    raise cancelled_errors[0]
                for result in results:
                    if result.action == "inserted":
                        inserted += 1
                    elif result.action == "replaced":
                        replaced += 1
                    elif result.action == "skipped":
                        skipped += 1
                    else:
                        raise ValueError(f"Unknown write action: {result.action}")

                while not exhausted and len(pending) < max_workers:
                    try:
                        document = next(iterator)
                    except StopIteration:
                        exhausted = True
                        continue
                    pending.add(executor.submit(process_document, document))
        except Exception:
            stop_event.set()
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise

        executor.shutdown(wait=True, cancel_futures=False)
        return IngestSummary(
            inserted=inserted,
            replaced=replaced,
            skipped=skipped,
        )

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
