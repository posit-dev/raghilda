from ragnar.document import (
    RetrievedChunk,
    Metric,
)

from ragnar._chunker import Chunk, MarkdownChunk
from ragnar._store import RetrievedDuckDBMarkdownChunk


def test_retrieved_chunk():
    metrics = [Metric(name="similarity", value=0.95)]
    chunk = RetrievedDuckDBMarkdownChunk(
        text="Sample content", metrics=metrics, start_index=1, end_index=10
    )

    assert isinstance(chunk, RetrievedDuckDBMarkdownChunk)
    assert isinstance(chunk, RetrievedChunk)
    assert isinstance(chunk, MarkdownChunk)
    assert isinstance(chunk, Chunk)
