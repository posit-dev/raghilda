from dataclasses import dataclass, field
from typing import Optional, Union
from .types import ChunkLike, IntoChunk

__all__ = ["Chunk", "MarkdownChunk", "Metric", "RetrievedChunk"]


@dataclass
class Chunk:
    """A segment of text extracted from a document.

    Chunks are the fundamental unit for retrieval in RAG applications.
    Each chunk contains the text content along with positional information
    that allows mapping back to the original document.

    Attributes
    ----------
    text
        The actual text content of the chunk.
    start_index
        Character position where this chunk begins in the source document.
    end_index
        Character position where this chunk ends in the source document.
    token_count
        Approximate number of tokens in this chunk.
    context
        Optional heading context showing the document hierarchy at this
        chunk's position (e.g., the Markdown headings that apply).
    """

    text: str
    start_index: int
    end_index: int
    token_count: int
    context: Optional[str] = None

    @classmethod
    def from_any(cls, chunk: Union[ChunkLike, IntoChunk]) -> "Chunk":
        """Convert any chunk-like or IntoChunk object to a raghilda Chunk.

        Parameters
        ----------
        chunk
            An object that implements the ChunkLike protocol or has a
            `to_chunk()` method.

        Returns
        -------
        Chunk
            A raghilda Chunk instance.
        """
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
    """A chunk extracted from a Markdown document.

    MarkdownChunk extends Chunk for use with Markdown content.
    It typically preserves heading context from the source document,
    allowing retrieval results to show where in the document hierarchy
    each chunk originated.
    """

    pass


@dataclass
class Metric:
    """A named metric value associated with a retrieved chunk.

    Metrics are used to store retrieval scores and other measurements
    that describe how well a chunk matches a query.

    Attributes
    ----------
    name
        The name of the metric (e.g., "similarity", "bm25_score").
    value
        The numeric value of the metric.

    Examples
    --------
    ```{python}
    from raghilda.chunk import Metric

    similarity = Metric(name="similarity", value=0.95)
    print(f"{similarity.name}: {similarity.value}")
    ```
    """

    name: str
    value: float


@dataclass
class RetrievedChunk(Chunk):
    """A chunk returned from a retrieval operation with associated metrics.

    RetrievedChunk extends Chunk with retrieval metrics that indicate
    how well the chunk matched the query. Common metrics include
    similarity scores and BM25 scores.

    Attributes
    ----------
    metrics
        List of Metric objects containing retrieval scores.

    Examples
    --------
    ```{python}
    from raghilda.chunk import RetrievedChunk, Metric

    chunk = RetrievedChunk(
        text="This is relevant content.",
        start_index=0,
        end_index=25,
        token_count=5,
        metrics=[
            Metric(name="similarity", value=0.92),
            Metric(name="bm25_score", value=15.3),
        ],
    )

    for metric in chunk.metrics:
        print(f"{metric.name}: {metric.value}")
    ```
    """

    metrics: list[Metric] = field(default_factory=list)
