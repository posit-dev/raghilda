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
    """Any object that can be converted into a Chunk via to_chunk()."""

    def to_chunk(self) -> "Chunk":
        """Convert this object into a `Chunk`."""
        ...


@runtime_checkable
class DocumentLike(Protocol):
    """Any document-like object."""

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
