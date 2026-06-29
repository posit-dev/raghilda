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
    """Structural type for a document that already carries its chunks.

    Like `DocumentLike`, but it also exposes a `chunks` sequence of `ChunkLike`
    objects. `ChunkedDocument.from_any()` normalizes such objects into a raghilda
    `ChunkedDocument`.
    """

    content: str
    chunks: Sequence[ChunkLike]


@runtime_checkable
class IntoDocument(Protocol):
    """Structural type for objects that convert themselves into a `Document`.

    Implement `to_document()` returning a raghilda `Document` (or
    `ChunkedDocument`) so raghilda can accept your type directly;
    `Document.from_any()` and `ChunkedDocument.from_any()` call it.
    """

    def to_document(self) -> "Document":
        """Convert this object into a `Document`."""
        ...


@runtime_checkable
class ChunkerLike(Protocol):
    """Structural type for any chunker raghilda can use.

    A `ChunkerLike` provides `chunk(document)` returning a `ChunkedDocument` and
    `chunk_text(text)` returning a sequence of `Chunk`s. raghilda's own
    `MarkdownChunker` satisfies this, as do third-party chunkers such as
    [chonkie](https://github.com/chonkie-inc/chonkie)'s, so they can be used
    wherever raghilda expects a chunker.
    """

    def chunk(self, document: "Document") -> "ChunkedDocument":
        """Chunk a document into a `ChunkedDocument`."""
        ...

    def chunk_text(self, text: str) -> Sequence["Chunk"]:
        """Chunk raw text into a sequence of `Chunk` objects."""
        ...
