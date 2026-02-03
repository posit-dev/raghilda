from dataclasses import dataclass, field
from typing import Optional, Union
from ._types import Chunk, Document, DocumentLike, IntoDocument


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


@dataclass
class Metric:
    name: str
    value: float


@dataclass
class RetrievedChunk(Chunk):
    metrics: list[Metric] = field(default_factory=list)
