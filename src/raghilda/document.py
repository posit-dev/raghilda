from dataclasses import dataclass
from typing import Any, Optional, Union

from .types import DocumentLike, IntoDocument
from .chunk import Chunk

__all__ = ["Document", "MarkdownDocument"]


@dataclass
class Document:
    """A document containing text content to be chunked and indexed.

    Documents are the primary input for RAG stores. Each document has
    text content and an optional origin identifier. After chunking, the document
    holds references to its chunks.

    Attributes
    ----------
    content
        The full text content of the document.
    origin
        Unique origin identifier for the document. This can be None or an empty
        string while preparing a document object, but stores require a populated
        origin for upsert operations.
    chunks
        List of chunks after the document has been processed by a chunker.
        None if the document hasn't been chunked yet.
    attributes
        Optional user-defined attributes applied at document insertion time.
        Document-level attributes can be inherited by chunks and returned
        during retrieval for filtering and downstream prompt/context use.
    """

    content: str
    origin: Optional[str] = None
    chunks: Optional[list[Chunk]] = None
    attributes: Optional[dict[str, Any]] = None

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
            raw_attributes = getattr(doc, "attributes", None)
            return cls(
                content=doc.content,
                origin=getattr(doc, "origin", None),
                chunks=chunks,
                attributes=dict(raw_attributes or {}),
            )
        raise TypeError(f"Cannot convert {type(doc).__name__} to Document")


@dataclass
class MarkdownDocument(Document):
    """A Markdown document with source tracking.

    MarkdownDocument extends Document with markdown-specific semantics for
    content that comes from a source origin (e.g., URL or file path). This is useful
    for citation and provenance tracking in RAG applications.

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
            origin=base.origin if base.origin is not None else origin,
            chunks=base.chunks,
            attributes=base.attributes,
        )
