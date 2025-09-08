from ragnar.chunker import RagnarMarkdownChunker


def test_markdown_chunker_basic() -> None:
    md = """# Title

## Section 1

Some text that is long enough to be chunked.

A second paragraph to make the text even longer.

## Section 2

More text here.

### Section 2.1

Some text under a level three heading.

#### Section 2.1.1

Some text under a level four heading.

## Section 3

Even more text here."""

    chunker = RagnarMarkdownChunker(chunk_size=40)
    chunks = chunker.chunk(md)
    assert len(chunks) >= 3
    first = chunks[0]
    assert first.text.startswith("# Title")
    ctx = chunks[2].context
    assert ctx is not None
    assert "Section 1" in ctx.text


def test_chunker_overlap() -> None:
    text = "abcdefghij" "klmnopqrst" "uvwxyz0123"
    chunker = RagnarMarkdownChunker(
        chunk_size=10, target_overlap=0.5, max_snap_distance=0
    )
    chunks = chunker.chunk(text)
    assert len(chunks) == 5
    for c in chunks:
        assert c.end_index - c.start_index == 10
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.end_index - nxt.start_index == 5
        assert prev.text[-5:] == nxt.text[:5]


def test_chunker_heading_context() -> None:
    md = (
        "# Title\n\n## Section\n\n"
        + "A" * 60
        + "\n\n### Subsection\n\n"
        + "B" * 60
    )
    chunker = RagnarMarkdownChunker(
        chunk_size=50,
        target_overlap=0,
        segment_by_heading_levels=[2],
        max_snap_distance=0,
    )
    chunks = chunker.chunk(md)
    sub_start = md.index("### Subsection") + len("### Subsection\n\n")
    sub_chunk = next(c for c in chunks if c.start_index >= sub_start)
    assert sub_chunk.context is not None
    assert sub_chunk.context.text == "# Title\n## Section\n### Subsection"


def test_chunker_max_snap_distance() -> None:
    text = "aaaaa bbbbb ccccc"
    chunker_snap = RagnarMarkdownChunker(
        chunk_size=5, target_overlap=0, max_snap_distance=2
    )
    chunker_no_snap = RagnarMarkdownChunker(
        chunk_size=5, target_overlap=0, max_snap_distance=0
    )
    chunks_snap = chunker_snap.chunk(text)
    chunks_no_snap = chunker_no_snap.chunk(text)
    # With snapping, boundaries move to surround whole words
    assert chunks_snap[1].start_index == 6
    assert chunks_snap[1].end_index == 12
    assert chunks_snap[1].text == "bbbbb "

    # Without snapping, boundaries fall exactly every five characters
    assert chunks_no_snap[1].start_index == 5
    assert chunks_no_snap[1].end_index == 10
    assert chunks_no_snap[1].text == " bbbb"
