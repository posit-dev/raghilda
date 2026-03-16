"""Shared helpers for store backends (DuckDB, PostgreSQL, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .chunk import Chunk, MarkdownChunk, RetrievedChunk
from ._attributes import AttributeType


class VSSMethod(StrEnum):
    COSINE_DISTANCE = "cosine_distance"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    NEGATIVE_INNER_PRODUCT = "negative_inner_product"


class IndexType(StrEnum):
    BM25 = "bm25"
    HNSW = "hnsw"
    FTS = "fts"


@dataclass(repr=False)
class StoreMarkdownChunk(MarkdownChunk):
    """MarkdownChunk with store-specific fields for database storage"""

    def __init__(
        self,
        text: str,
        start_index: int,
        end_index: int,
        context=None,
        char_count=None,
        origin=None,
        attributes=None,
    ):
        if char_count is None:
            char_count = len(text)

        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            char_count=char_count,
            context=context,
            origin=origin,
            attributes=attributes,
        )


@dataclass(repr=False)
class RetrievedStoreMarkdownChunk(StoreMarkdownChunk, RetrievedChunk):
    """StoreMarkdownChunk with retrieval metrics"""

    def __init__(
        self,
        text: str,
        start_index: int,
        end_index: int,
        context=None,
        char_count=None,
        origin=None,
        metrics=None,
        chunk_ids=None,
        attributes=None,
    ):
        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            context=context,
            char_count=char_count,
            origin=origin,
            attributes=attributes,
        )

        if metrics is None:
            metrics = []
        self.metrics = metrics
        if chunk_ids is None:
            chunk_ids = []
        self.chunk_ids = chunk_ids


RESERVED_SYSTEM_COLUMNS = {
    "chunk_id",
    "context",
    "embedding",
    "origin",
    "text",
    "start_index",
    "end_index",
    "char_count",
    "metric_name",
    "metric_value",
}

FILTERABLE_BASE_COLUMNS = {
    "chunk_id",
    "origin",
    "start_index",
    "end_index",
    "char_count",
    "context",
}


def quote_identifier(identifier: str) -> str:
    """Quotes an identifier for use in SQL queries."""
    identifier = identifier.replace('"', '""')
    return f'"{identifier}"'


def attributes_select_clause(
    alias: str, attributes_schema: Mapping[str, AttributeType]
) -> str:
    if not attributes_schema:
        return ""
    parts = [f"{alias}.{quote_identifier(column)}," for column in attributes_schema]
    return "\n            " + "\n            ".join(parts) + "\n            "


def coerce_index_type(value: IndexType | str) -> IndexType:
    if isinstance(value, IndexType):
        return value
    try:
        return IndexType(value)
    except ValueError as exc:
        allowed = ", ".join(x.value for x in IndexType)
        raise ValueError(
            f"Unknown index type '{value}'. Allowed values: {allowed}"
        ) from exc


def slice_chunk_text(content: str, *, start_index: int, end_index: int) -> str:
    if start_index < 0 or end_index < start_index or end_index > len(content):
        raise ValueError(
            "Chunk indices must satisfy 0 <= start_index <= end_index <= len(content). "
            f"Got start_index={start_index}, end_index={end_index}, "
            f"len(content)={len(content)}."
        )
    return content[start_index:end_index]


def validate_chunk_against_document(
    *, document_origin: str, content: str, chunk: Chunk
) -> None:
    validate_chunk_origin_matches_document_origin(
        document_origin=document_origin,
        chunk=chunk,
    )
    validate_chunk_text_matches_document_content(
        content=content,
        chunk=chunk,
    )


def validate_chunk_text_matches_document_content(*, content: str, chunk: Chunk) -> None:
    expected_text = slice_chunk_text(
        content,
        start_index=chunk.start_index,
        end_index=chunk.end_index,
    )
    if chunk.text != expected_text:
        raise ValueError(
            "Chunk text must match document.content[start_index:end_index]. "
            f"Got chunk.text={chunk.text!r}, expected {expected_text!r} "
            f"for start_index={chunk.start_index}, end_index={chunk.end_index}."
        )


def validate_chunk_origin_matches_document_origin(
    *, document_origin: str, chunk: Chunk
) -> None:
    if chunk.origin is None:
        return
    if chunk.origin != document_origin:
        raise ValueError(
            "Chunk origin must be None or match document.origin. "
            f"Got chunk.origin={chunk.origin!r}, document.origin={document_origin!r}."
        )
