"""Protocol types for raghilda.

These protocols define the interfaces for type checking compatibility
with chunks, documents, and chunkers.
"""

from typing import TYPE_CHECKING, Protocol, Optional, Sequence, runtime_checkable

if TYPE_CHECKING:
    from .chunk import Chunk
    from .document import Document


@runtime_checkable
class ChunkLike(Protocol):
    """Any chunk-like object (chonkie, raghilda, or custom)."""

    text: str
    start_index: int
    end_index: int
    token_count: int


@runtime_checkable
class IntoChunk(Protocol):
    """Any object that can be converted into a Chunk via to_chunk()."""

    def to_chunk(self) -> "Chunk": ...


@runtime_checkable
class DocumentLike(Protocol):
    """Any document-like object."""

    content: str
    chunks: Optional[Sequence[ChunkLike]]


@runtime_checkable
class IntoDocument(Protocol):
    """Any object that can be converted into a Document via to_document()."""

    def to_document(self) -> "Document": ...


@runtime_checkable
class ChunkerLike(Protocol):
    """Any chunker-like object (chonkie, raghilda, or custom)."""

    def chunk(self, text: str) -> Sequence["Chunk"]: ...
