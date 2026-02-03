from dataclasses import dataclass, field
from typing import Protocol, Optional, Sequence, runtime_checkable, Union
import uuid


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


@dataclass
class Chunk:
    """Base chunk type for raghilda."""

    text: str
    start_index: int
    end_index: int
    token_count: int
    context: Optional[str] = None

    @classmethod
    def from_any(cls, chunk: Union[ChunkLike, IntoChunk]) -> "Chunk":
        """Convert any chunk-like or IntoChunk object to a raghilda Chunk."""
        if isinstance(chunk, IntoChunk):
            if not callable(chunk.to_chunk):
                raise TypeError(
                    f"{type(chunk).__name__}.to_chunk must be a method, not {type(chunk.to_chunk).__name__}"
                )
            result = chunk.to_chunk()
            if not isinstance(result, Chunk):
                raise TypeError(
                    f"{type(chunk).__name__}.to_chunk() must return a Chunk, got {type(result).__name__}"
                )
            return result
        elif isinstance(chunk, ChunkLike):
            return cls(
                text=chunk.text,
                start_index=chunk.start_index,
                end_index=chunk.end_index,
                token_count=chunk.token_count,
                context=getattr(chunk, "context", None),
            )
        raise TypeError(f"Cannot convert {type(chunk).__name__} to Chunk")


def _generate_doc_id() -> str:
    return f"doc_{uuid.uuid4().hex}"


@dataclass
class Document:
    """Base document type for raghilda."""

    content: str
    id: str = field(default_factory=_generate_doc_id)
    chunks: Optional[list[Chunk]] = None

    @classmethod
    def from_any(cls, doc: Union[DocumentLike, IntoDocument]) -> "Document":
        """Convert any document-like or IntoDocument object to a raghilda Document."""
        if isinstance(doc, IntoDocument):
            if not callable(doc.to_document):
                raise TypeError(
                    f"{type(doc).__name__}.to_document must be a method, not {type(doc.to_document).__name__}"
                )
            result = doc.to_document()
            if not isinstance(result, Document):
                raise TypeError(
                    f"{type(doc).__name__}.to_document() must return a Document, got {type(result).__name__}"
                )
            return result
        elif isinstance(doc, DocumentLike):
            chunks = None
            if doc.chunks is not None:
                chunks = [Chunk.from_any(c) for c in doc.chunks]
            doc_id = getattr(doc, "id", None) or _generate_doc_id()
            return cls(content=doc.content, id=doc_id, chunks=chunks)
        raise TypeError(f"Cannot convert {type(doc).__name__} to Document")


@runtime_checkable
class ChunkerLike(Protocol):
    """Any chunker-like object (chonkie, raghilda, or custom)."""

    def chunk(self, text: str) -> list[Chunk]: ...


class BaseChunker:
    """Base class for chunkers."""

    def chunk(self, text: str) -> list[Chunk]:
        raise NotImplementedError

    def chunk_document(self, doc: Document) -> Document:
        """Chunk a document and return it with chunks attached."""
        doc.chunks = self.chunk(doc.content)
        return doc

    def __call__(self, text: str) -> list[Chunk]:
        return self.chunk(text)
