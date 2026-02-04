from dataclasses import dataclass, field
from typing import Optional, Union
import uuid

from .types import DocumentLike, IntoDocument
from .chunk import Chunk

__all__ = ["Document", "MarkdownDocument"]


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


@dataclass
class MarkdownDocument(Document):
    origin: Optional[str] = None

    @classmethod
    def from_any(
        cls, doc: Union[DocumentLike, IntoDocument], origin: Optional[str] = None
    ) -> "MarkdownDocument":
        """Convert any document-like or IntoDocument object to a MarkdownDocument."""
        base = Document.from_any(doc)
        return cls(
            content=base.content,
            id=base.id,
            chunks=base.chunks,
            origin=getattr(doc, "origin", origin),
        )
