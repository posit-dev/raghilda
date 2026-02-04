from dataclasses import dataclass, field
from typing import Optional, Union
from .types import ChunkLike, IntoChunk

__all__ = ["Chunk", "MarkdownChunk", "Metric", "RetrievedChunk"]


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


@dataclass
class MarkdownChunk(Chunk):
    pass


@dataclass
class Metric:
    name: str
    value: float


@dataclass
class RetrievedChunk(Chunk):
    metrics: list[Metric] = field(default_factory=list)
