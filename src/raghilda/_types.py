"""Protocol types for raghilda."""

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from .chunk import Chunk
    from .document import ChunkedDocument, Document


@runtime_checkable
class ChunkLike(Protocol):
    """Any chunk-like object (chonkie, raghilda, or custom)."""

    text: str
    start_index: int
    end_index: int


@runtime_checkable
class IntoChunk(Protocol):
    """Any object that can be converted into a Chunk via to_chunk()."""

    def to_chunk(self) -> "Chunk": ...


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

    def to_document(self) -> "Document": ...


@runtime_checkable
class ChunkerLike(Protocol):
    """Any chunker-like object (chonkie, raghilda, or custom)."""

    def chunk(self, document: "Document") -> "ChunkedDocument": ...

    def chunk_text(self, text: str) -> Sequence["Chunk"]: ...
