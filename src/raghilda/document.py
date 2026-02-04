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
    """A document containing text content to be chunked and indexed.

    Documents are the primary input for RAG stores. Each document has
    text content and a unique identifier. After chunking, the document
    holds references to its chunks.

    Attributes
    ----------
    content
        The full text content of the document.
    id
        Unique identifier for the document. Auto-generated if not provided.
    chunks
        List of chunks after the document has been processed by a chunker.
        None if the document hasn't been chunked yet.
    """

    content: str
    id: str = field(default_factory=_generate_doc_id)
    chunks: Optional[list[Chunk]] = None

    @classmethod
    def from_any(cls, doc: Union[DocumentLike, IntoDocument]) -> "Document":
        """Convert any document-like or IntoDocument object to a raghilda Document.

        Parameters
        ----------
        doc
            An object that implements the DocumentLike protocol or has a
            `to_document()` method.

        Returns
        -------
        Document
            A raghilda Document instance.
        """
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
    """A Markdown document with source tracking.

    MarkdownDocument extends Document with an `origin` field that tracks
    where the document came from (e.g., a URL or file path). This is useful
    for citation and provenance tracking in RAG applications.

    Attributes
    ----------
    origin
        The source location of the document (URL, file path, etc.).
        Used for tracking provenance and generating citations.

    Examples
    --------
    ```{python}
    from raghilda.document import MarkdownDocument

    # Create from content directly
    doc = MarkdownDocument(
        content="# Hello World\\n\\nThis is a test document.",
        origin="https://example.com/hello.md",
    )
    print(f"Document from: {doc.origin}")
    print(f"Content length: {len(doc.content)} characters")
    ```
    """

    origin: Optional[str] = None

    @classmethod
    def from_any(
        cls, doc: Union[DocumentLike, IntoDocument], origin: Optional[str] = None
    ) -> "MarkdownDocument":
        """Convert any document-like or IntoDocument object to a MarkdownDocument.

        Parameters
        ----------
        doc
            An object that implements the DocumentLike protocol or has a
            `to_document()` method.
        origin
            Optional origin to set if the source object doesn't have one.

        Returns
        -------
        MarkdownDocument
            A raghilda MarkdownDocument instance.
        """
        base = Document.from_any(doc)
        return cls(
            content=base.content,
            id=base.id,
            chunks=base.chunks,
            origin=getattr(doc, "origin", origin),
        )
