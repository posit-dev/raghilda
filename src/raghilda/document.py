from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

from .types import ChunkedDocumentLike, DocumentLike, IntoDocument
from .chunk import Chunk

__all__ = [
    "Document",
    "ChunkedDocument",
    "MarkdownDocument",
    "ChunkedMarkdownDocument",
]


@dataclass
class Document:
    """A document containing text content to be chunked and indexed.

    Documents are the primary input for RAG stores. Each document has
    text content and an optional origin identifier.

    Attributes
    ----------
    content
        The full text content of the document.
    origin
        Unique origin identifier for the document. This can be None or an empty
        string while preparing a document object, but stores require a populated
        origin for upsert operations.
    attributes
        Optional user-defined attributes applied at document insertion time.
        Document-level attributes can be inherited by chunks and returned
        during retrieval for filtering and downstream prompt/context use.
    """

    content: str
    origin: Optional[str] = None
    attributes: Optional[dict[str, Any]] = None

    @classmethod
    def from_any(cls, doc: Union[DocumentLike, IntoDocument]) -> "Document":
        """Convert any document-like or IntoDocument object to a raghilda Document.

        This conversion only accepts unchunked inputs. If the source object
        already carries chunks, use `ChunkedDocument.from_any()` instead.

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
            if isinstance(result, ChunkedDocument):
                raise TypeError(
                    f"{type(doc).__name__}.to_document() must return an unchunked Document, got {type(result).__name__}"
                )
            if not isinstance(result, Document):
                raise TypeError(
                    f"{type(doc).__name__}.to_document() must return a Document, got {type(result).__name__}"
                )
            return result
        elif isinstance(doc, DocumentLike):
            incoming_chunks = getattr(doc, "chunks", None)
            if incoming_chunks is not None and len(incoming_chunks) > 0:
                raise TypeError(
                    f"Cannot convert chunked {type(doc).__name__} to Document; use ChunkedDocument.from_any()"
                )
            raw_attributes = getattr(doc, "attributes", None)
            return cls(
                content=doc.content,
                origin=getattr(doc, "origin", None),
                attributes=dict(raw_attributes or {}),
            )
        raise TypeError(f"Cannot convert {type(doc).__name__} to Document")

    def to_chunked(self, chunks: Sequence[Chunk]) -> "ChunkedDocument":
        """Return a ChunkedDocument with the same fields and supplied chunks."""
        return ChunkedDocument(
            content=self.content,
            origin=self.origin,
            attributes=self.attributes,
            chunks=list(chunks),
        )


@dataclass(kw_only=True)
class ChunkedDocument(Document):
    """A document with an attached sequence of chunks.

    This is the explicit chunked variant of `Document`, used by stores and
    chunkers that operate on pre-segmented content.
    """

    chunks: list[Chunk]

    @classmethod
    def from_any(
        cls, doc: Union[DocumentLike, ChunkedDocumentLike, IntoDocument]
    ) -> "ChunkedDocument":
        """Convert any chunked document-like object to a raghilda ChunkedDocument.

        Use `Document.from_any()` for unchunked inputs.
        """
        if isinstance(doc, cls):
            return doc
        if isinstance(doc, IntoDocument):
            if not callable(doc.to_document):
                raise TypeError(
                    f"{type(doc).__name__}.to_document must be a method, not {type(doc.to_document).__name__}"
                )
            result = doc.to_document()
            if not isinstance(result, ChunkedDocument):
                raise TypeError(
                    f"{type(doc).__name__}.to_document() must return a ChunkedDocument, got {type(result).__name__}"
                )
            return cls.from_any(result)
        if isinstance(doc, ChunkedDocumentLike):
            raw_attributes = getattr(doc, "attributes", None)
            return cls(
                content=doc.content,
                origin=getattr(doc, "origin", None),
                attributes=dict(raw_attributes or {}),
                chunks=[Chunk.from_any(c) for c in doc.chunks],
            )
        if isinstance(doc, DocumentLike):
            raise TypeError(
                f"Cannot convert unchunked {type(doc).__name__} to ChunkedDocument; use Document.from_any()"
            )
        raise TypeError(f"Cannot convert {type(doc).__name__} to ChunkedDocument")

    def __iter__(self):
        return iter(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index):
        return self.chunks[index]


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

        This conversion only accepts unchunked inputs. If the source object
        already carries chunks, use `ChunkedMarkdownDocument.from_any()`
        instead.

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
            attributes=base.attributes,
        )

    def to_chunked(self, chunks: Sequence[Chunk]) -> "ChunkedMarkdownDocument":
        """Return a ChunkedMarkdownDocument with the same fields and chunks."""
        return ChunkedMarkdownDocument(
            content=self.content,
            origin=self.origin,
            attributes=self.attributes,
            chunks=list(chunks),
        )


@dataclass(kw_only=True)
class ChunkedMarkdownDocument(MarkdownDocument, ChunkedDocument):
    """A Markdown document with an attached sequence of chunks."""

    @classmethod
    def from_any(
        cls,
        doc: Union[DocumentLike, ChunkedDocumentLike, IntoDocument],
        origin: Optional[str] = None,
    ) -> "ChunkedMarkdownDocument":
        """Convert any chunked document-like object to a ChunkedMarkdownDocument.

        Use `MarkdownDocument.from_any()` for unchunked inputs.
        """
        base = ChunkedDocument.from_any(doc)
        return cls(
            content=base.content,
            origin=base.origin if base.origin is not None else origin,
            attributes=base.attributes,
            chunks=base.chunks,
        )
