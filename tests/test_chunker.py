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

Even more text here.
"""

    chunker = RagnarMarkdownChunker(chunk_size=40)
    chunks = chunker.chunk(md)
    assert len(chunks) >= 3
    first = chunks[0]
    assert first.text.startswith("# Title")
    ctx = chunks[2].context
    assert ctx is not None
    assert "Section 1" in ctx.text
