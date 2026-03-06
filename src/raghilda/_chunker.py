from typing import Any, List, Optional, Sequence
import bisect
import re
import commonmark

from .chunk import Chunk, MarkdownChunk
from .document import Document


class BaseChunker:
    """Base class for chunkers."""

    def chunk(self, text: str) -> Sequence[Chunk]:
        raise NotImplementedError

    def chunk_document(self, doc: Document) -> Document:
        """Chunk a document and return it with chunks attached."""
        doc.chunks = list(self.chunk(doc.content))
        for chunk in doc.chunks:
            chunk.origin = doc.origin
        return doc

    def __call__(self, text: str) -> Sequence[Chunk]:
        return self.chunk(text)


class MarkdownChunker(BaseChunker):
    """Chunk Markdown documents into overlapping segments at semantic boundaries.

    This chunker divides Markdown text into smaller, overlapping chunks while
    intelligently positioning cut points at semantic boundaries like headings,
    paragraphs, sentences, and words. Rather than cutting rigidly at character
    counts, it nudges cut points to the nearest sensible boundary, producing
    more semantically coherent chunks suitable for RAG applications.

    Parameters
    ----------
    chunk_size
        Target size for each chunk in characters. The chunker attempts to
        create chunks near this size, though actual sizes may vary based on
        semantic boundaries. Default is 1600 characters.
    target_overlap
        Fraction of overlap between successive chunks, from 0 to 1.
        Default is 0.5 (50% overlap). Even with 0, some overlap may occur
        because the last chunk is anchored to the document end.
    max_snap_distance
        Maximum distance (in characters) to move a cut point to reach a
        semantic boundary. If no boundary is found within this distance,
        the cut point stays at its original position. Default is 20.
    segment_by_heading_levels
        List of heading levels (1-6) that act as hard boundaries. When
        specified, no chunk will cross these headings, and segments between
        them are chunked independently. For example, `[1, 2]` ensures chunks
        never span across h1 or h2 headings.

    Examples
    --------
    ```{python}
    from raghilda.chunker import MarkdownChunker

    chunker = MarkdownChunker(
        chunk_size=100,
        target_overlap=0.2,
        segment_by_heading_levels=[1, 2],
    )

    text = '''# Introduction
    This is the introduction section with some content.

    ## Background
    Here is background information that provides context.

    ## Methods
    The methods section describes our approach.
    '''

    chunks = chunker.chunk(text)
    for chunk in chunks:
        print(f"[{chunk.start_index}:{chunk.end_index}] {chunk.text[:40]}...")
    ```

    Notes
    -----
    The chunking algorithm works as follows:

    1. Parse the Markdown to identify semantic boundaries (headings,
       paragraphs, sentences, lines, words)
    2. If `segment_by_heading_levels` is set, split the document at those
       headings first
    3. For each segment, calculate target chunk boundaries based on
       `chunk_size` and `target_overlap`
    4. Snap each boundary to the nearest semantic boundary (preferring
       headings > paragraphs > sentences > lines > words)
    5. Extract chunks with their positional information and heading context
    """

    def __init__(
        self,
        chunk_size: int = 1600,
        target_overlap: float = 0.5,
        *,
        max_snap_distance: int = 20,
        segment_by_heading_levels: Optional[list[int]] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.target_overlap = target_overlap
        self.max_snap_distance = max_snap_distance
        self.segment_by_heading_levels = segment_by_heading_levels

    # Helpers ---------------------------------------------------------------
    @staticmethod
    def _make_segment_chunk_targets(
        start: int, end: int, chunk_size: int, overlap: float
    ) -> List[tuple[int, int]]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if end - start <= chunk_size:
            return [(start, end)]
        stride = max(int(round(chunk_size * (1 - overlap))), 1)
        starts = list(range(start, end - chunk_size, stride))
        last_start = end - chunk_size
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return [(s, s + chunk_size) for s in starts]

    def _make_chunk_targets(
        self, md_len: int, segment_breaks: List[int]
    ) -> List[tuple[int, int]]:
        seg_starts = [0] + sorted(segment_breaks)
        seg_ends = seg_starts[1:] + [md_len]
        out: List[tuple[int, int]] = []
        for s, e in zip(seg_starts, seg_ends, strict=False):
            out.extend(
                self._make_segment_chunk_targets(
                    s, e, self.chunk_size, self.target_overlap
                )
            )
        return out

    @staticmethod
    def _snap_nearest(
        x: List[int], candidates: List[int], max_dist: Optional[int]
    ) -> List[Optional[int]]:
        if not candidates:
            return [None] * len(x)
        candidates = sorted(set(candidates))
        out: List[Optional[int]] = []
        for xi in x:
            idx = bisect.bisect_left(candidates, xi)
            if idx == 0:
                pick = candidates[0]
            elif idx == len(candidates):
                pick = candidates[-1]
            else:
                left = candidates[idx - 1]
                right = candidates[idx]
                pick = left if abs(xi - left) <= abs(xi - right) else right
            if max_dist is not None and abs(pick - xi) > max_dist:
                out.append(None)
            else:
                out.append(pick)
        return out

    @staticmethod
    def _markdown_node_positions(
        md: str, node_types: Optional[Sequence[str]] = None
    ) -> List[dict[str, Any]]:
        if md == "":
            return []
        parser = commonmark.Parser(options={"sourcepos": True})
        ast = parser.parse(md)
        line_starts = [0] + [m.end() for m in re.finditer("\n", md)]

        def walk(node: Any, out: List[dict[str, Any]]) -> None:
            while node:
                if node.sourcepos and (node_types is None or node.t in node_types):
                    (sl, sc), (el, ec) = node.sourcepos
                    start = line_starts[sl - 1] + sc - 1
                    end = line_starts[el - 1] + ec
                    info: dict[str, Any] = {
                        "type": node.t,
                        "start": start,
                        "end": end,
                    }
                    if node.t == "heading":
                        info["level"] = node.level
                    out.append(info)
                if node.first_child:
                    walk(node.first_child, out)
                node = node.nxt

        results: List[dict[str, Any]] = []
        walk(ast, results)
        results.sort(key=lambda d: (d["start"], d["end"]))
        return results

    @staticmethod
    def _heading_positions(text: str) -> List[dict[str, Any]]:
        headings = MarkdownChunker._markdown_node_positions(text, ["heading"])
        for h in headings:
            h["text"] = text[h["start"] : h["end"]].strip()
        return headings

    @staticmethod
    def _paragraph_starts(text: str) -> List[int]:
        paragraphs = MarkdownChunker._markdown_node_positions(text, ["paragraph"])
        starts = [0, *[p["start"] for p in paragraphs]]
        return sorted(set(starts))

    @staticmethod
    def _sentence_starts(text: str) -> List[int]:
        starts = []
        for m in re.finditer(r"[.!?]\s+", text):
            starts.append(m.end())
        return starts

    @staticmethod
    def _line_starts(text: str) -> List[int]:
        return [0] + [m.end() for m in re.finditer("\n", text)]

    @staticmethod
    def _word_starts(text: str) -> List[int]:
        return [m.end() for m in re.finditer(r"\s+", text)]

    @staticmethod
    def _heading_context(headings: List[dict[str, Any]], pos: int) -> List[str]:
        """Return the hierarchy of headings active at ``pos``.

        This walks the list of headings in order, maintaining a stack of the
        most recent heading at each level. Only headings that start before
        ``pos`` are considered part of the context.

        Headers that start exactly at pos are excluded since they're
        already present in the chunk text, but they still affect the
        hierarchy by removing same-level headings from the context.
        """
        stack: List[dict[str, Any]] = []
        for h in headings:
            if h["start"] > pos:
                break
            elif h["start"] == pos:
                # Header starts at chunk position - don't include it but let it affect hierarchy
                level = h["level"]
                while stack and stack[-1]["level"] >= level:
                    stack.pop()
                # Don't append this heading to stack since it's in the chunk
            else:
                # Header starts before chunk position - include it normally
                level = h["level"]
                while stack and stack[-1]["level"] >= level:
                    stack.pop()
                stack.append(h)
        return [h["text"] for h in stack]

    # Main -----------------------------------------------------------------
    def chunk(self, text: str) -> List[MarkdownChunk]:
        md_len = len(text)
        headings = self._heading_positions(text)

        if self.segment_by_heading_levels is None:
            segment_breaks = []
        else:
            segment_breaks = [
                h["start"]
                for h in headings
                if h["level"] in self.segment_by_heading_levels
            ]

        chunk_targets = self._make_chunk_targets(md_len, segment_breaks)
        snap_points = sorted(
            {0, md_len, *[s for s, _ in chunk_targets], *[e for _, e in chunk_targets]}
        )
        snap_table: dict[int, int | None] = {
            p: (p if p in (0, md_len) else None) for p in snap_points
        }

        boundary_types = [
            ("heading", [h["start"] for h in headings]),
            ("paragraph", self._paragraph_starts(text)),
            ("sentence", self._sentence_starts(text)),
            ("line", self._line_starts(text)),
            ("word", self._word_starts(text)),
        ]
        for _name, candidates in boundary_types:
            unsnapped = [p for p, v in snap_table.items() if v is None]
            if not unsnapped:
                break
            snapped = self._snap_nearest(unsnapped, candidates, self.max_snap_distance)
            for p, snapped_p in zip(unsnapped, snapped, strict=False):
                if snapped_p is not None:
                    snap_table[p] = snapped_p
        for p, v in list(snap_table.items()):
            if v is None:
                snap_table[p] = p

        snap_table_int: dict[int, int] = {
            k: (v if v is not None else k) for k, v in snap_table.items()
        }

        # build chunks ------------------------------------------------------
        chunks: List[MarkdownChunk] = []
        sorted_snaps = sorted(snap_table_int.values())
        for start, end in chunk_targets:
            s = snap_table_int[start]
            e = snap_table_int[end]
            if s is None or e is None:
                continue
            if s >= e:
                # find next greater boundary
                for b in sorted_snaps:
                    if b > s:
                        e = b
                        break

            chunk_text = text[s:e]
            char_count = len(chunk_text)

            ctx_lines = self._heading_context(headings, s)
            ctx = "\n".join(ctx_lines) if len(ctx_lines) > 0 else None

            chunks.append(
                MarkdownChunk(
                    text=chunk_text,
                    start_index=s,
                    end_index=e,
                    context=ctx,
                    char_count=char_count,
                )
            )

        # remove duplicates
        unique: dict[int, MarkdownChunk] = {}
        for c in sorted(chunks, key=lambda c: (c.start_index, -c.end_index)):
            existing = unique.get(c.start_index)
            if existing is None or c.end_index > existing.end_index:
                unique[c.start_index] = c
        by_end: dict[int, MarkdownChunk] = {}
        for c in sorted(unique.values(), key=lambda c: (c.end_index, c.start_index)):
            existing = by_end.get(c.end_index)
            if existing is None or c.start_index < existing.start_index:
                by_end[c.end_index] = c
        result = sorted(by_end.values(), key=lambda c: c.start_index)
        return result
