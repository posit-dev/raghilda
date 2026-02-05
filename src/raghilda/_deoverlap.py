"""
Deoverlap functionality for merging overlapping retrieved chunks.
"""

from __future__ import annotations
from typing import Any, Callable, TypeVar
from .chunk import RetrievedChunk

T = TypeVar("T", bound=RetrievedChunk)


def default_merge(target: RetrievedChunk, source: RetrievedChunk) -> None:
    """
    Default merge function for overlapping RetrievedChunk instances.

    Merges the source chunk into the target chunk by:
    - Extending the text to cover the union of both ranges
    - Updating end_index to the maximum of both
    - Combining metrics from both chunks
    - Updating token_count based on the new text length

    Parameters
    ----------
    target
        The chunk to merge into (modified in place).
    source
        The overlapping chunk to merge from.
    """
    new_end = max(target.end_index, source.end_index)
    if source.end_index > target.end_index:
        # Extend text with the non-overlapping part of the source chunk
        overlap_len = target.end_index - source.start_index
        target.text = target.text + source.text[overlap_len:]
    target.end_index = new_end
    target.metrics.extend(source.metrics or [])
    target.token_count = len(target.text)


def deoverlap_chunks(
    chunks: list[T],
    key: Callable[[T], Any],
    merge: Callable[[T, T], None] = default_merge,
) -> list[T]:
    """
    Merge overlapping chunks from the same document.

    Chunks are considered overlapping if they share any character positions
    (based on start_index and end_index). When chunks overlap, they are merged
    using the provided merge function.

    Parameters
    ----------
    chunks
        List of retrieved chunks, potentially with overlapping ranges within
        the same document.
    key
        Function to extract a grouping key from each chunk. Chunks with the same
        key are considered to be from the same document and may be merged.
    merge
        Function to merge two overlapping chunks. Takes (target, source) and
        modifies target in place to incorporate source. Defaults to `default_merge`
        which extends text, updates end_index, combines metrics, and updates
        token_count.

    Returns
    -------
    list[T]
        List of chunks with overlaps merged. Chunks with different keys
        are never merged together.
    """
    if not chunks:
        return []

    # Group chunks by key (overlaps only occur within the same document)
    by_doc: dict[Any, list[T]] = {}
    for chunk in chunks:
        k = key(chunk)
        if k not in by_doc:
            by_doc[k] = []
        by_doc[k].append(chunk)

    result: list[T] = []

    for _, doc_chunks in by_doc.items():
        # Sort by start_index for interval merging
        doc_chunks.sort(key=lambda c: c.start_index)

        merged: list[T] = []
        for chunk in doc_chunks:
            if not merged:
                merged.append(chunk)
                continue

            last = merged[-1]
            # Check if current chunk overlaps with the last merged chunk
            # Chunks overlap if current.start_index < last.end_index
            if chunk.start_index < last.end_index:
                merge(last, chunk)
            else:
                # No overlap, add as a new chunk
                merged.append(chunk)

        result.extend(merged)

    return result


# Tests for deoverlap_chunks
def test__deoverlap_chunks_overlapping_same_document():
    """Overlapping chunks from the same document should be merged."""
    from ._store import RetrievedDuckDBMarkdownChunk
    from .chunk import Metric

    chunks = [
        RetrievedDuckDBMarkdownChunk(
            text="hello world",
            start_index=0,
            end_index=11,
            doc_id=1,
            chunk_id=1,
            metrics=[Metric("bm25", 0.8)],
        ),
        RetrievedDuckDBMarkdownChunk(
            text="world hello",
            start_index=6,
            end_index=17,
            doc_id=1,
            chunk_id=2,
            metrics=[Metric("bm25", 0.7)],
        ),
    ]
    result = deoverlap_chunks(chunks, key=lambda c: c.doc_id)
    assert len(result) == 1
    assert result[0].start_index == 0
    assert result[0].end_index == 17
    assert result[0].text == "hello world hello"
    assert len(result[0].metrics) == 2


def test__deoverlap_chunks_different_documents():
    """Overlapping ranges from different documents should not be merged."""
    from ._store import RetrievedDuckDBMarkdownChunk
    from .chunk import Metric

    chunks = [
        RetrievedDuckDBMarkdownChunk(
            text="hello",
            start_index=0,
            end_index=5,
            doc_id=1,
            chunk_id=1,
            metrics=[Metric("a", 1)],
        ),
        RetrievedDuckDBMarkdownChunk(
            text="world",
            start_index=0,
            end_index=5,
            doc_id=2,
            chunk_id=2,
            metrics=[Metric("b", 2)],
        ),
    ]
    result = deoverlap_chunks(chunks, key=lambda c: c.doc_id)
    assert len(result) == 2


def test__deoverlap_chunks_non_overlapping():
    """Non-overlapping chunks from the same document should not be merged."""
    from ._store import RetrievedDuckDBMarkdownChunk
    from .chunk import Metric

    chunks = [
        RetrievedDuckDBMarkdownChunk(
            text="hello",
            start_index=0,
            end_index=5,
            doc_id=1,
            chunk_id=1,
            metrics=[Metric("a", 1)],
        ),
        RetrievedDuckDBMarkdownChunk(
            text="world",
            start_index=10,
            end_index=15,
            doc_id=1,
            chunk_id=2,
            metrics=[Metric("b", 2)],
        ),
    ]
    result = deoverlap_chunks(chunks, key=lambda c: c.doc_id)
    assert len(result) == 2


def test__deoverlap_chunks_adjacent_not_merged():
    """Chunks that are adjacent (end == start) should not be merged."""
    from ._store import RetrievedDuckDBMarkdownChunk

    chunks = [
        RetrievedDuckDBMarkdownChunk(
            text="hello",
            start_index=0,
            end_index=5,
            doc_id=1,
            chunk_id=1,
            metrics=[],
        ),
        RetrievedDuckDBMarkdownChunk(
            text="world",
            start_index=5,
            end_index=10,
            doc_id=1,
            chunk_id=2,
            metrics=[],
        ),
    ]
    result = deoverlap_chunks(chunks, key=lambda c: c.doc_id)
    assert len(result) == 2


def test__deoverlap_chunks_empty():
    """Empty input should return empty output."""
    result = deoverlap_chunks([], key=lambda c: c.doc_id)
    assert len(result) == 0


def test__deoverlap_chunks_single():
    """Single chunk should be returned unchanged."""
    from ._store import RetrievedDuckDBMarkdownChunk
    from .chunk import Metric

    chunks = [
        RetrievedDuckDBMarkdownChunk(
            text="solo",
            start_index=0,
            end_index=4,
            doc_id=1,
            chunk_id=1,
            metrics=[Metric("a", 1)],
        )
    ]
    result = deoverlap_chunks(chunks, key=lambda c: c.doc_id)
    assert len(result) == 1
    assert result[0].text == "solo"


def test__deoverlap_chunks_chain():
    """Multiple chunks that chain together should all be merged."""
    from ._store import RetrievedDuckDBMarkdownChunk
    from .chunk import Metric

    chunks = [
        RetrievedDuckDBMarkdownChunk(
            text="01234",
            start_index=0,
            end_index=5,
            doc_id=1,
            chunk_id=1,
            metrics=[Metric("a", 1)],
        ),
        RetrievedDuckDBMarkdownChunk(
            text="34567",
            start_index=3,
            end_index=8,
            doc_id=1,
            chunk_id=2,
            metrics=[Metric("b", 2)],
        ),
        RetrievedDuckDBMarkdownChunk(
            text="6789X",
            start_index=6,
            end_index=11,
            doc_id=1,
            chunk_id=3,
            metrics=[Metric("c", 3)],
        ),
    ]
    result = deoverlap_chunks(chunks, key=lambda c: c.doc_id)
    assert len(result) == 1
    assert result[0].start_index == 0
    assert result[0].end_index == 11
    assert result[0].text == "0123456789X"
    assert len(result[0].metrics) == 3


def test__deoverlap_chunks_reverse_order():
    """Chunks should be sorted before processing."""
    from ._store import RetrievedDuckDBMarkdownChunk
    from .chunk import Metric

    chunks = [
        RetrievedDuckDBMarkdownChunk(
            text="world",
            start_index=6,
            end_index=11,
            doc_id=1,
            chunk_id=2,
            metrics=[Metric("b", 2)],
        ),
        RetrievedDuckDBMarkdownChunk(
            text="hello ",
            start_index=0,
            end_index=6,
            doc_id=1,
            chunk_id=1,
            metrics=[Metric("a", 1)],
        ),
    ]
    result = deoverlap_chunks(chunks, key=lambda c: c.doc_id)
    # Adjacent chunks (end_index=6, start_index=6) should NOT be merged
    assert len(result) == 2
    # First chunk should be "hello " (sorted by start_index)
    assert result[0].text == "hello "
    assert result[1].text == "world"


def test__deoverlap_chunks_fully_contained():
    """A chunk fully contained within another should be merged."""
    from ._store import RetrievedDuckDBMarkdownChunk
    from .chunk import Metric

    chunks = [
        RetrievedDuckDBMarkdownChunk(
            text="hello world",
            start_index=0,
            end_index=11,
            doc_id=1,
            chunk_id=1,
            metrics=[Metric("a", 1)],
        ),
        RetrievedDuckDBMarkdownChunk(
            text="lo wo",
            start_index=3,
            end_index=8,
            doc_id=1,
            chunk_id=2,
            metrics=[Metric("b", 2)],
        ),
    ]
    result = deoverlap_chunks(chunks, key=lambda c: c.doc_id)
    assert len(result) == 1
    assert result[0].text == "hello world"
    assert result[0].start_index == 0
    assert result[0].end_index == 11
    assert len(result[0].metrics) == 2
