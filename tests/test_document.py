from ragnar.document import (
    RetrievedChunk,
    RetrievedMarkdownChunk,
    Metric,
    MarkdownChunk,
    Chunk,
)


def test_retrieved_chunk():
    metrics = [Metric(name="similarity", value=0.95)]
    chunk = RetrievedMarkdownChunk(
        content="Sample content", metrics=metrics, start=1, end=10
    )

    assert isinstance(chunk, RetrievedMarkdownChunk)
    assert isinstance(chunk, RetrievedChunk)
    assert isinstance(chunk, MarkdownChunk)
    assert isinstance(chunk, Chunk)
