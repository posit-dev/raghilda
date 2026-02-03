"""Public types for raghilda.

This module exports the core types and protocols for working with chunks,
documents, and chunkers in raghilda.
"""

from ._types import (
    # Concrete types
    Chunk,
    Document,
    BaseChunker,
    # Protocols for accepting compatible types
    ChunkLike,
    DocumentLike,
    ChunkerLike,
    # Protocols for custom conversion
    IntoChunk,
    IntoDocument,
)

__all__ = [
    # Concrete types
    "Chunk",
    "Document",
    "BaseChunker",
    # Protocols
    "ChunkLike",
    "DocumentLike",
    "ChunkerLike",
    "IntoChunk",
    "IntoDocument",
]
