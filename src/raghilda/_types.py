"""Protocol types for raghilda."""

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from .chunk import Chunk
    from .document import ChunkedDocument, Document


@runtime_checkable
class ChunkLike(Protocol):
    """Structural type for any chunk raghilda can consume.

    A `ChunkLike` is any object that exposes a chunk's essential fields, whether
    it comes from raghilda, [chonkie](https://github.com/chonkie-inc/chonkie), or
    your own code. raghilda accepts such objects wherever a chunk is expected and
    normalizes them with `Chunk.from_any()`. The required fields are `text`,
    `start_index`, and `end_index`; `char_count`, `context`, `origin`, and
    `attributes` are read when present.
    """

    text: str
    start_index: int
    end_index: int


@runtime_checkable
class IntoChunk(Protocol):
    """Structural type for objects that convert themselves into a `Chunk`.

    Implement `to_chunk()` returning a raghilda `Chunk` to let raghilda accept
    your type directly; `Chunk.from_any()` calls it when given an `IntoChunk`.
    This is the explicit alternative to `ChunkLike`: rather than exposing chunk
    fields, the object knows how to build a `Chunk` itself.
    """

    def to_chunk(self) -> "Chunk":
        """Convert this object into a `Chunk`."""
        ...


@runtime_checkable
class DocumentLike(Protocol):
    """Structural type for any unchunked document raghilda can consume.

    A `DocumentLike` exposes a document's `content` (and optionally `origin` and
    `attributes`). raghilda normalizes such objects with `Document.from_any()`.
    Use `ChunkedDocumentLike` instead for objects that already carry chunks.
    """

    content: str


@runtime_checkable
class ChunkedDocumentLike(Protocol):
    """Any chunked document-like object."""

    content: str
    chunks: Sequence[ChunkLike]


@runtime_checkable
class IntoDocument(Protocol):
    """Any object that can be converted into a Document via to_document()."""

    def to_document(self) -> "Document":
        """Convert this object into a `Document`."""
        ...


@runtime_checkable
class ChunkerLike(Protocol):
    """Any chunker-like object (chonkie, raghilda, or custom)."""

    def chunk(self, document: "Document") -> "ChunkedDocument":
        """Chunk a document into a `ChunkedDocument`."""
        ...

    def chunk_text(self, text: str) -> Sequence["Chunk"]:
        """Chunk raw text into a sequence of `Chunk` objects."""
        ...
