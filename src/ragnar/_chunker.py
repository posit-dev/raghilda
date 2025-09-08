from chonkie.chunker.base import BaseChunker, Chunk
from dataclasses import dataclass
from typing import Optional, Callable, Any, Union, Sequence, List
import bisect
import re


@dataclass
class MarkdownChunk(Chunk):
    pass


class RagnarMarkdownChunker(BaseChunker):
    def __init__(
        self,
        tokenizer_or_token_counter: Union[str, Callable[[str], int], Any] = "character",
        chunk_size: int = 1600,
        target_overlap: float = 0.5,
        *,
        max_snap_distance: int = 20,
        segment_by_heading_levels: Optional[list[int]] = None,
        context: bool = True,
        text: bool = True,
    ) -> None:
        super().__init__(tokenizer_or_token_counter)
        self.chunk_size = chunk_size
        self.target_overlap = target_overlap
        self.max_snap_distance = max_snap_distance
        self.segment_by_heading_levels = segment_by_heading_levels or []
        self.context = context
        self.text = text

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
    def _heading_positions(text: str) -> List[dict[str, Any]]:
        headings: List[dict[str, Any]] = []
        for m in re.finditer(r"^(#{1,6})[ \t]+.*$", text, flags=re.MULTILINE):
            level = len(m.group(1))
            headings.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "level": level,
                    "text": m.group(0).strip(),
                }
            )
        return headings

    @staticmethod
    def _paragraph_starts(text: str) -> List[int]:
        starts = [0]
        for m in re.finditer(r"\n\s*\n", text):
            starts.append(m.end())
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

    # Main -----------------------------------------------------------------
    def chunk(self, text: str) -> Sequence[MarkdownChunk]:
        md_len = len(text)
        headings = self._heading_positions(text)
        segment_breaks = [
            h["start"] for h in headings if h["level"] in self.segment_by_heading_levels
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
            chunk_text = text[s:e] if self.text else ""
            ctx = None
            if self.context:
                ctx_lines = [h["text"] for h in headings if h["start"] < s]
                ctx = "\n".join(ctx_lines)
            token_count = self.tokenizer.count_tokens(chunk_text)
            chunks.append(
                MarkdownChunk(
                    text=chunk_text,
                    start_index=s,
                    end_index=e,
                    context=ctx,
                    token_count=token_count,
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
