"""Protocol types for raghilda.

This module exports the protocol types for type checking compatibility
with chunks, documents, and chunkers.
"""

from ._types import (
    ChunkedDocumentLike,
    ChunkerLike,
    ChunkLike,
    DocumentLike,
    IntoChunk,
    IntoDocument,
)

__all__ = [
    "ChunkLike",
    "ChunkedDocumentLike",
    "ChunkerLike",
    "DocumentLike",
    "IntoChunk",
    "IntoDocument",
]
