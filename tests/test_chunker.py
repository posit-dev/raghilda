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
    # chunk in Section 2, should not contain the "Section 1" context
    assert chunks[5].context.text.find("Section 1") < 0





def test_chunker_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz0123"
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
    md = "# Title\n\n## Section\n\n" + "A" * 60 + "\n\n### Subsection\n\n" + "B" * 60
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


def test_chunker_heading_context_sibling_sections() -> None:
    md = (
        "# Title\n\n"
        "## Section A\n\n"
        "AAA\n\n"
        "## Section B\n\n"
        "BBB"
    )
    chunker = RagnarMarkdownChunker(
        chunk_size=50,
        target_overlap=0,
        segment_by_heading_levels=[2],
        max_snap_distance=0,
    )
    chunks = chunker.chunk(md)
    a_heading = md.index("## Section A")
    b_heading = md.index("## Section B")
    a_chunk = next(c for c in chunks if c.start_index == a_heading)
    b_chunk = next(c for c in chunks if c.start_index == b_heading)
    assert a_chunk.context is not None
    assert a_chunk.context.text == "# Title\n## Section A"
    assert b_chunk.context is not None
    assert b_chunk.context.text == "# Title\n## Section B"
    assert "Section A" not in b_chunk.context.text


def test_chunker_heading_context_nested_siblings() -> None:
    md = (
        "# Title\n\n"
        "## Section A\n\n"
        "AAA\n\n"
        "### Section A1\n\n"
        "AAA1\n\n"
        "## Section B\n\n"
        "BBB\n\n"
        "### Section B1\n\n"
        "BBB1"
    )
    chunker = RagnarMarkdownChunker(
        chunk_size=50,
        target_overlap=0,
        segment_by_heading_levels=[2, 3],
        max_snap_distance=0,
    )
    chunks = chunker.chunk(md)
    b_heading = md.index("## Section B")
    b1_heading = md.index("### Section B1")
    b_chunk = next(c for c in chunks if c.start_index == b_heading)
    b1_chunk = next(c for c in chunks if c.start_index == b1_heading)
    assert b_chunk.context is not None
    assert b_chunk.context.text == "# Title\n## Section B"
    assert "Section A" not in b_chunk.context.text
    assert b1_chunk.context is not None
    assert b1_chunk.context.text == "# Title\n## Section B\n### Section B1"
    assert "Section A" not in b1_chunk.context.text
    assert "Section A1" not in b1_chunk.context.text
