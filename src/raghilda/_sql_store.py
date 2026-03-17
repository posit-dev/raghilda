"""SQLAlchemy Core base class for SQL-backed vector stores."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from typing import Any, Mapping, Optional, Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from ._attributes import (
    AttributeFilter,
    AttributeSpec,
    AttributeStructType,
    AttributeType,
    AttributeValue,
    coerce_attribute_value_for_output,
    compile_filter_to_sqlalchemy,
    filterable_attribute_paths,
    merge_attribute_values,
)
from ._deoverlap import deoverlap_chunks
from ._sql_constructs import FTSRank, TextSlice, ToSearchVector, VectorDistance
from ._store import BaseStore, WriteResult
from ._store_helpers import (
    FILTERABLE_BASE_COLUMNS,
    IndexType,
    RetrievedStoreMarkdownChunk,
    VSSMethod,
    coerce_index_type as _coerce_index_type,
    slice_chunk_text as _slice_chunk_text,
    validate_chunk_against_document as _validate_chunk_against_document,
)
from ._store_metadata import EmbeddedAttributesStoreMetadata
from .chunk import Chunk, MarkdownChunk, Metric
from .document import ChunkedMarkdownDocument, Document
from .embedding import EmbedInputType

logger = logging.getLogger(__name__)

# SA type mapping for attribute types
class TSVECTOR(sa.types.UserDefinedType[Any]):
    """Represents PostgreSQL's TSVECTOR type for SQLAlchemy."""

    cache_ok = True

    def get_col_spec(self, **kw: Any) -> str:
        return "TSVECTOR"


_SA_TYPE_MAP: dict[type, type[sa.types.TypeEngine[Any]]] = {
    str: sa.Text,
    int: sa.Integer,
    float: sa.Float,
    bool: sa.Boolean,
}


def _sa_column_type(attribute_type: AttributeType) -> sa.types.TypeEngine[Any]:
    """Return the SQLAlchemy column type for an attribute type."""
    if isinstance(attribute_type, AttributeStructType):
        from sqlalchemy.dialects.postgresql import JSONB

        return JSONB()
    if hasattr(attribute_type, "dimension"):
        # AttributeFloatVectorType
        return Vector(attribute_type.dimension)  # type: ignore[arg-type]
    sa_type_class = _SA_TYPE_MAP.get(attribute_type)  # type: ignore[arg-type]
    if sa_type_class is None:
        raise ValueError(f"Unsupported attribute type: {attribute_type}")
    return sa_type_class()


def build_tables(
    sa_metadata: sa.MetaData,
    attributes_spec: Mapping[str, AttributeSpec],
    embed_dimension: int | None,
) -> tuple[sa.Table, sa.Table]:
    """Build the documents and embeddings Table objects.

    Returns (documents_table, embeddings_table).
    """
    documents = sa.Table(
        "documents",
        sa_metadata,
        sa.Column("origin", sa.Text, primary_key=True),
        sa.Column("text", sa.Text),
    )

    cols: list[Any] = [
        sa.Column("origin", sa.Text, sa.ForeignKey("documents.origin"), nullable=False),
        sa.Column("chunk_id", sa.Integer, autoincrement=True),
        sa.Column("start_index", sa.Integer),
        sa.Column("end_index", sa.Integer),
        sa.Column("char_count", sa.Integer),
        sa.Column("context", sa.Text),
    ]

    # search_vector — TSVECTOR for full-text search
    cols.append(
        sa.Column("search_vector", TSVECTOR()),
    )

    # Attribute columns
    for col_name, spec in attributes_spec.items():
        cols.append(sa.Column(col_name, _sa_column_type(spec.attribute_type)))

    # Embedding column
    if embed_dimension is not None:
        cols.append(sa.Column("embedding", Vector(embed_dimension)))

    # Primary key constraint
    cols.append(
        sa.PrimaryKeyConstraint("origin", "start_index", "end_index"),
    )

    embeddings = sa.Table("embeddings", sa_metadata, *cols)

    return documents, embeddings


class SQLStore(BaseStore):
    """Base class for SQLAlchemy-backed vector stores.

    Subclasses provide create() / connect() class methods and any
    dialect-specific DDL. All shared query and DML logic lives here.
    """

    def __init__(
        self,
        engine: sa.Engine,
        metadata: EmbeddedAttributesStoreMetadata,
        sa_metadata: sa.MetaData,
        documents: sa.Table,
        embeddings: sa.Table,
    ):
        self.engine = engine
        self.metadata = metadata
        self.sa_metadata = sa_metadata
        self.documents = documents
        self.embeddings = embeddings
        self._db_lock = threading.Lock()

    # -- BaseStore abstract methods (create/connect provided by subclass) -----

    @staticmethod
    def create(*args: Any, **kwargs: Any) -> "SQLStore":
        raise NotImplementedError("Subclasses must implement create()")

    @staticmethod
    def connect(*args: Any, **kwargs: Any) -> "SQLStore":
        raise NotImplementedError("Subclasses must implement connect()")

    def close(self) -> None:
        """Dispose of the SA engine and release connections."""
        self.engine.dispose()

    # -- upsert ---------------------------------------------------------------

    def upsert(
        self,
        document: Document,
        *,
        skip_if_unchanged: bool = True,
    ) -> WriteResult[ChunkedMarkdownDocument]:
        """Upsert a document into the store."""
        if not isinstance(document, ChunkedMarkdownDocument):
            raise NotImplementedError(
                f"Upsert not implemented for type {type(document)}"
            )
        if not isinstance(document.origin, str) or not document.origin:
            raise ValueError("document.origin must be a non-empty string for upsert().")
        if len(document.chunks) == 0:
            raise ValueError("Document must contain at least one chunk.")
        for chunk in document.chunks:
            _validate_chunk_against_document(
                document_origin=document.origin,
                content=document.content,
                chunk=chunk,
            )

        with self._db_lock:
            with self.engine.connect() as conn:
                existing_rows = self._get_existing_documents_by_origin(
                    document.origin, conn
                )
                existing = existing_rows[0] if existing_rows else None
                if (
                    skip_if_unchanged
                    and existing is not None
                    and existing["text"] == document.content
                    and self._chunk_layout_matches_existing(
                        chunked_doc=document,
                        origin=existing["origin"],
                        conn=conn,
                    )
                ):
                    current_document = self._load_document_snapshot(
                        origin=existing["origin"],
                        text=existing["text"],
                        conn=conn,
                    )
                    return WriteResult(
                        action="skipped",
                        document=current_document,
                    )

        doc_row, chunk_rows = self._prepare_chunked_document_rows(document)

        with self._db_lock:
            with self.engine.begin() as conn:
                existing_rows = self._get_existing_documents_by_origin(
                    document.origin, conn
                )
                existing = existing_rows[0] if existing_rows else None
                if (
                    skip_if_unchanged
                    and existing is not None
                    and existing["text"] == document.content
                    and self._chunk_layout_matches_existing(
                        chunked_doc=document,
                        origin=existing["origin"],
                        conn=conn,
                    )
                ):
                    current_document = self._load_document_snapshot(
                        origin=existing["origin"],
                        text=existing["text"],
                        conn=conn,
                    )
                    return WriteResult(
                        action="skipped",
                        document=current_document,
                    )

                action = "inserted"
                replaced_document: ChunkedMarkdownDocument | None = None
                if existing is not None:
                    action = "replaced"
                    replaced_document = self._load_document_snapshot(
                        origin=existing["origin"],
                        text=existing["text"],
                        conn=conn,
                    )

                e = self.embeddings
                d = self.documents
                if action == "replaced":
                    conn.execute(
                        e.delete().where(e.c.origin == doc_row["origin"])
                    )
                    conn.execute(
                        d.update()
                        .where(d.c.origin == doc_row["origin"])
                        .values(text=doc_row["text"])
                    )
                else:
                    conn.execute(d.insert().values(**doc_row))
                self._insert_embedding_rows(conn, chunk_rows)

                current_document = self._load_document_snapshot(
                    origin=str(doc_row["origin"]),
                    text=document.content,
                    conn=conn,
                )
                return WriteResult(
                    action=action,
                    document=current_document,
                    replaced_document=replaced_document,
                )

    # -- retrieval ------------------------------------------------------------

    def retrieve(
        self,
        text: str,
        top_k: int = 3,
        *,
        deoverlap: bool = True,
        attributes_filter: Optional[AttributeFilter] = None,
    ) -> Sequence[RetrievedStoreMarkdownChunk]:
        """Retrieve chunks combining VSS and FTS."""
        retrieved_chunks: list[RetrievedStoreMarkdownChunk] = []
        if self.metadata.embed is not None:
            retrieved_chunks = self.retrieve_vss(
                text,
                top_k,
                attributes_filter=attributes_filter,
            )

        retrieved_chunks.extend(
            self.retrieve_fts(
                text,
                top_k,
                attributes_filter=attributes_filter,
            )
        )

        combined_chunks: dict[
            tuple[str | None, int | None], RetrievedStoreMarkdownChunk
        ] = {}
        for chunk in retrieved_chunks:
            first_chunk_id = chunk.chunk_ids[0] if chunk.chunk_ids else None
            key = (chunk.origin, first_chunk_id)
            if key not in combined_chunks:
                combined_chunks[key] = chunk
            else:
                combined_chunks[key].metrics.extend(chunk.metrics or [])

        chunks = list(combined_chunks.values())

        if deoverlap:
            chunks = deoverlap_chunks(chunks, key=lambda c: c.origin)

        return chunks

    def retrieve_vss(
        self,
        query: str | Sequence[float],
        top_k: int,
        *,
        method: VSSMethod = VSSMethod.COSINE_DISTANCE,
        attributes_filter: Optional[AttributeFilter] = None,
    ) -> list[RetrievedStoreMarkdownChunk]:
        """Retrieve chunks using vector similarity search."""
        if isinstance(query, str):
            if self.metadata.embed is None:
                raise ValueError("No embedding function available in the store")
            query = self.metadata.embed.embed([query], EmbedInputType.QUERY)[0]

        e = self.embeddings
        d = self.documents
        query_vector = "[" + ",".join(str(x) for x in query) + "]"

        allowed_filter_columns = self._filterable_columns()
        sa_filter = compile_filter_to_sqlalchemy(
            attributes_filter,
            allowed_columns=allowed_filter_columns,
            table=e,
        )

        attribute_columns = list(self.metadata.attributes_schema)
        text_slice = TextSlice(
            d.c.text, e.c.start_index + 1, e.c.end_index - e.c.start_index
        )
        metric_value = VectorDistance(e.c.embedding, query_vector, str(method))

        if sa_filter is None:
            # Optimization: pre-filter in a subquery when no attribute filter
            inner = (
                sa.select(
                    e,
                    metric_value.label("metric_value"),
                )
                .order_by(sa.literal_column("metric_value").asc())
                .limit(top_k)
                .subquery("e")
            )
            select_cols: list[Any] = [
                inner.c.chunk_id,
                d.c.origin.label("origin"),
                inner.c.start_index,
                inner.c.end_index,
                inner.c.char_count,
                inner.c.context,
            ]
            for col in attribute_columns:
                select_cols.append(inner.c[col])
            select_cols.extend([
                TextSlice(
                    d.c.text,
                    inner.c.start_index + 1,
                    inner.c.end_index - inner.c.start_index,
                ).label("text"),
                sa.literal(str(method)).label("metric_name"),
                inner.c.metric_value,
            ])
            stmt = (
                sa.select(*select_cols)
                .select_from(inner.join(d, inner.c.origin == d.c.origin))
                .order_by(inner.c.metric_value.asc())
                .limit(top_k)
            )
        else:
            select_cols = [
                e.c.chunk_id,
                d.c.origin.label("origin"),
                e.c.start_index,
                e.c.end_index,
                e.c.char_count,
                e.c.context,
            ]
            for col in attribute_columns:
                select_cols.append(e.c[col])
            select_cols.extend([
                text_slice.label("text"),
                sa.literal(str(method)).label("metric_name"),
                metric_value.label("metric_value"),
            ])
            stmt = (
                sa.select(*select_cols)
                .select_from(e.join(d, e.c.origin == d.c.origin))
                .where(sa_filter)
                .order_by(sa.literal_column("metric_value").asc())
                .limit(top_k)
            )

        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            col_names = list(result.keys())
            rows = result.fetchall()

        return _rows_to_retrieved_chunks(
            rows, col_names, self.metadata.attributes_schema
        )

    def retrieve_fts(
        self,
        query: str,
        top_k: int,
        *,
        attributes_filter: Optional[AttributeFilter] = None,
    ) -> list[RetrievedStoreMarkdownChunk]:
        """Retrieve chunks using full-text search."""
        e = self.embeddings
        d = self.documents
        allowed_filter_columns = self._filterable_columns()
        sa_filter = compile_filter_to_sqlalchemy(
            attributes_filter,
            allowed_columns=allowed_filter_columns,
            table=e,
        )
        attribute_columns = list(self.metadata.attributes_schema)

        text_slice = TextSlice(
            d.c.text, e.c.start_index + 1, e.c.end_index - e.c.start_index
        )
        fts_rank = FTSRank(e.c.search_vector, query)

        ranked_cols: list[Any] = [
            e.c.chunk_id,
            d.c.origin.label("origin"),
            e.c.start_index,
            e.c.end_index,
            e.c.char_count,
            e.c.context,
        ]
        for col in attribute_columns:
            ranked_cols.append(e.c[col])
        ranked_cols.extend([
            text_slice.label("text"),
            sa.literal("fts").label("metric_name"),
            fts_rank.label("metric_value"),
        ])

        ranked_stmt = (
            sa.select(*ranked_cols)
            .select_from(e.join(d, e.c.origin == d.c.origin))
        )
        if sa_filter is not None:
            ranked_stmt = ranked_stmt.where(sa_filter)
        ranked = ranked_stmt.cte("ranked")

        stmt = (
            sa.select(ranked)
            .where(ranked.c.metric_value > 0)
            .order_by(ranked.c.metric_value.desc())
            .limit(top_k)
        )

        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            col_names = list(result.keys())
            rows = result.fetchall()

        return _rows_to_retrieved_chunks(
            rows, col_names, self.metadata.attributes_schema
        )

    # -- index / size ---------------------------------------------------------

    def build_index(
        self,
        type: Optional[IndexType | str | list[IndexType | str]] = None,
    ) -> None:
        """Build indexes on the embeddings table."""
        if type is None:
            index_types = [IndexType.FTS, IndexType.HNSW]
        elif isinstance(type, (IndexType, str)):
            index_types = [_coerce_index_type(type)]
        else:
            index_types = [_coerce_index_type(item) for item in type]

        with self.engine.begin() as conn:
            if IndexType.FTS in index_types:
                conn.execute(
                    sa.text(
                        "CREATE INDEX IF NOT EXISTS idx_embeddings_search_vector "
                        "ON embeddings USING GIN (search_vector)"
                    )
                )

            if IndexType.HNSW in index_types:
                conn.execute(sa.text("DROP INDEX IF EXISTS store_hnsw_cosine_index"))
                conn.execute(sa.text("DROP INDEX IF EXISTS store_hnsw_l2_index"))
                conn.execute(sa.text("DROP INDEX IF EXISTS store_hnsw_ip_index"))
                conn.execute(
                    sa.text(
                        "CREATE INDEX store_hnsw_cosine_index "
                        "ON embeddings USING hnsw (embedding vector_cosine_ops)"
                    )
                )
                conn.execute(
                    sa.text(
                        "CREATE INDEX store_hnsw_l2_index "
                        "ON embeddings USING hnsw (embedding vector_l2_ops)"
                    )
                )
                conn.execute(
                    sa.text(
                        "CREATE INDEX store_hnsw_ip_index "
                        "ON embeddings USING hnsw (embedding vector_ip_ops)"
                    )
                )

    def size(self) -> int:
        """Count distinct documents."""
        d = self.documents
        stmt = sa.select(sa.func.count(sa.distinct(d.c.origin)))
        with self.engine.connect() as conn:
            result = conn.execute(stmt).scalar()
            if result is None:
                raise RuntimeError("Failed to get size of the store")
            return int(result)

    # -- private helpers ------------------------------------------------------

    def _filterable_columns(self) -> set[str]:
        filterable_attribute_columns = filterable_attribute_paths(
            self.metadata.attributes_schema
        )
        return FILTERABLE_BASE_COLUMNS | filterable_attribute_columns

    def _get_existing_documents_by_origin(
        self, origin: str, conn: sa.Connection
    ) -> list[dict[str, str]]:
        d = self.documents
        stmt = (
            sa.select(d.c.origin, d.c.text)
            .where(d.c.origin == origin)
            .order_by(d.c.origin)
        )
        rows = conn.execute(stmt).fetchall()
        return [{"origin": row[0], "text": row[1]} for row in rows]

    def _chunk_layout_matches_existing(
        self,
        *,
        chunked_doc: ChunkedMarkdownDocument,
        origin: str,
        conn: sa.Connection,
    ) -> bool:
        incoming = self._chunk_layout_records(chunked_doc)
        existing = self._chunk_layout_records_from_store(origin, conn)
        return incoming == existing

    def _chunk_layout_records(
        self, chunked_doc: ChunkedMarkdownDocument
    ) -> list[tuple[Any, ...]]:
        records: list[tuple[Any, ...]] = []
        attributes_columns = list(self.metadata.attributes_schema)
        for chunk in chunked_doc.chunks:
            resolved = merge_attribute_values(
                attributes_spec=self.metadata.attributes_spec,
                sources=[chunked_doc.attributes, chunk.attributes],
            )
            row: list[Any] = [
                chunk.start_index,
                chunk.end_index,
                chunk.char_count,
                chunk.context,
            ]
            row.extend(
                self._coerce_chunk_layout_attribute_value(col, resolved[col])
                for col in attributes_columns
            )
            records.append(tuple(row))
        records.sort(key=lambda item: (item[0], item[1]))
        return records

    def _chunk_layout_records_from_store(
        self, origin: str, conn: sa.Connection
    ) -> list[tuple[Any, ...]]:
        e = self.embeddings
        attributes_columns = list(self.metadata.attributes_schema)
        cols = [e.c.start_index, e.c.end_index, e.c.char_count, e.c.context]
        cols.extend(e.c[col] for col in attributes_columns)
        stmt = (
            sa.select(*cols)
            .where(e.c.origin == origin)
            .order_by(e.c.start_index, e.c.end_index)
        )
        rows = conn.execute(stmt).fetchall()
        records: list[tuple[Any, ...]] = []
        for row in rows:
            start_index = int(row[0])
            end_index = int(row[1])
            char_count = int(row[2])
            context = row[3]
            attribute_values = [
                self._coerce_chunk_layout_attribute_value(col, row[4 + idx])
                for idx, col in enumerate(attributes_columns)
            ]
            records.append(
                (start_index, end_index, char_count, context, *attribute_values)
            )
        return records

    def _coerce_chunk_layout_attribute_value(self, column: str, value: Any) -> Any:
        return coerce_attribute_value_for_output(
            column,
            value,
            self.metadata.attributes_schema[column],
        )

    def _load_document_snapshot(
        self, *, origin: str, text: str, conn: sa.Connection
    ) -> ChunkedMarkdownDocument:
        e = self.embeddings
        attribute_columns = list(self.metadata.attributes_schema)
        cols: list[sa.ColumnElement[Any]] = [
            e.c.start_index,
            e.c.end_index,
            e.c.char_count,
            e.c.context,
        ]
        cols.extend(e.c[col] for col in attribute_columns)
        stmt = (
            sa.select(*cols)
            .where(e.c.origin == origin)
            .order_by(e.c.start_index, e.c.end_index)
        )
        result = conn.execute(stmt)
        col_names = list(result.keys())
        rows = result.fetchall()

        chunks: list[Chunk] = []
        document_attributes: dict[str, Any] = {}
        for row in rows:
            row_dict = dict(zip(col_names, row))
            attributes = {
                key: row_dict[key] for key in attribute_columns if key in row_dict
            }
            for key, value in attributes.items():
                if key not in document_attributes or document_attributes[key] is None:
                    document_attributes[key] = value
            start_index = int(row_dict["start_index"])
            end_index = int(row_dict["end_index"])
            chunk_text = _slice_chunk_text(
                text,
                start_index=start_index,
                end_index=end_index,
            )
            chunks.append(
                MarkdownChunk(
                    start_index=start_index,
                    end_index=end_index,
                    text=chunk_text,
                    char_count=int(row_dict["char_count"]),
                    context=row_dict.get("context"),
                    origin=origin,
                    attributes=attributes or None,
                )
            )

        return ChunkedMarkdownDocument(
            origin=origin,
            content=text,
            attributes=document_attributes or None,
            chunks=chunks,
        )

    def _prepare_chunked_document_rows(
        self,
        chunked_doc: ChunkedMarkdownDocument,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        doc = {
            "origin": chunked_doc.origin,
            "text": chunked_doc.content,
        }
        chunks = [asdict(chunk) for chunk in chunked_doc.chunks]

        resolved_chunk_attributes: list[dict[str, AttributeValue]] = []
        for chunk in chunked_doc.chunks:
            chunk_attributes = getattr(chunk, "attributes", None)
            resolved_chunk_attributes.append(
                merge_attribute_values(
                    attributes_spec=self.metadata.attributes_spec,
                    sources=[chunked_doc.attributes, chunk_attributes],
                )
            )

        embedded_chunks = None
        chunk_texts = [chunk.text for chunk in chunked_doc.chunks]
        if self.metadata.embed is not None:
            embedded_chunks = self.metadata.embed.embed(
                chunk_texts, EmbedInputType.DOCUMENT
            )
            if len(embedded_chunks) != len(chunks):
                raise ValueError(
                    "Embedding provider must return exactly one embedding per chunk "
                    f"(got {len(embedded_chunks)} embeddings for {len(chunks)} chunks)"
                )

        chunk_rows: list[dict[str, Any]] = []
        for index, chunk_data in enumerate(chunks):
            row = dict(chunk_data)

            row.pop("attributes", None)
            row.pop("text", None)
            row.pop("id", None)
            row.pop("origin", None)
            row.pop("chunk_ids", None)

            if embedded_chunks is not None:
                row["embedding"] = embedded_chunks[index]
            else:
                row.pop("embedding", None)

            for column in self.metadata.attributes_schema:
                value = resolved_chunk_attributes[index][column]
                attr_type = self.metadata.attributes_schema[column]
                if isinstance(attr_type, AttributeStructType) and value is not None:
                    value = json.dumps(value)
                row[column] = value

            row["origin"] = doc["origin"]
            chunk_text = chunked_doc.chunks[index].text
            context_text = chunked_doc.chunks[index].context or ""
            row["_search_text"] = f"{context_text} {chunk_text}".strip()
            chunk_rows.append(row)

        return doc, chunk_rows

    def _insert_embedding_rows(
        self, conn: sa.Connection, rows: Sequence[Mapping[str, Any]]
    ) -> None:
        """Insert embedding rows, computing search_vector from _search_text."""
        e = self.embeddings
        for row in rows:
            search_text = row.get("_search_text", "")
            values: dict[str, Any] = {
                c: row[c] for c in row if c != "_search_text"
            }
            values["search_vector"] = ToSearchVector(search_text)
            conn.execute(e.insert().values(**values))


def _rows_to_retrieved_chunks(
    rows: Sequence[Any],
    columns: list[str] | Sequence[str],
    attributes_schema: Mapping[str, AttributeType],
) -> list[RetrievedStoreMarkdownChunk]:
    output: list[RetrievedStoreMarkdownChunk] = []
    for chunk in rows:
        chunk_dict = dict(zip(columns, chunk))
        name, value = chunk_dict.pop("metric_name"), chunk_dict.pop("metric_value")
        chunk_id = chunk_dict.pop("chunk_id", None)
        attribute_values: dict[str, AttributeValue] = {}
        for key, attribute_type in attributes_schema.items():
            if key in chunk_dict:
                attribute_values[key] = coerce_attribute_value_for_output(
                    key,
                    chunk_dict.pop(key),
                    attribute_type,
                )
        chunk_dict["chunk_ids"] = [] if chunk_id is None else [int(chunk_id)]
        chunk_dict["metrics"] = [Metric(name, value)]
        chunk_dict["attributes"] = attribute_values
        output.append(RetrievedStoreMarkdownChunk(**chunk_dict))
    return output
