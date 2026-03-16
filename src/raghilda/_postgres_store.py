"""PostgreSQL vector store backend using pgvector."""

from ._store import BaseStore, WriteResult
import json
import threading
from .embedding import EmbeddingProvider, EmbedInputType, embedding_from_config
from .chunk import Chunk, MarkdownChunk, Metric
from .document import ChunkedMarkdownDocument, Document
from typing import Any, Mapping, Optional, Sequence
from dataclasses import dataclass, asdict
import logging
from ._deoverlap import deoverlap_chunks
from ._attributes import (
    AttributeFilter,
    AttributeSpec,
    AttributesSchemaSpec,
    AttributeType,
    AttributeValue,
    attributes_spec_from_json_dict,
    attributes_spec_to_json_dict,
    coerce_attribute_value_for_output,
    compile_filter_to_sql_postgres,
    filterable_attribute_paths,
    normalize_attributes_spec,
    merge_attribute_values,
    postgres_sql_type_for_attribute_type,
)
from ._store_metadata import (
    EmbeddedAttributesStoreMetadata,
    attributes_schema_from_spec,
)
from ._store_helpers import (
    VSSMethod,
    IndexType,
    RetrievedStoreMarkdownChunk,
    RESERVED_SYSTEM_COLUMNS,
    FILTERABLE_BASE_COLUMNS,
    quote_identifier as _quote_identifier,
    attributes_select_clause as _attributes_select_clause,
    coerce_index_type as _coerce_index_type,
    slice_chunk_text as _slice_chunk_text,
    validate_chunk_against_document as _validate_chunk_against_document,
)

import psycopg

logger = logging.getLogger(__name__)


@dataclass
class PostgresStoreMetadata(EmbeddedAttributesStoreMetadata):
    name: str
    title: str
    embed: Optional[EmbeddingProvider]
    attributes: dict[str, AttributeSpec]

    @property
    def attributes_spec(self) -> dict[str, AttributeSpec]:
        return self.attributes

    @property
    def attributes_schema(self) -> dict[str, AttributeType]:
        return attributes_schema_from_spec(self.attributes)


class PostgresStore(BaseStore):
    """A vector store backed by PostgreSQL with pgvector.

    PostgresStore provides vector storage with support for both
    semantic search (using pgvector embeddings) and full-text search
    (using PostgreSQL tsvector/tsquery). Each store uses a separate
    PostgreSQL database.
    """

    @staticmethod
    def create(
        connection_string: str,
        embed: Optional[EmbeddingProvider],
        overwrite: bool = False,
        name: Optional[str] = None,
        title: Optional[str] = None,
        attributes: Optional[AttributesSchemaSpec] = None,
    ):
        """Create a new PostgreSQL store.

        Parameters
        ----------
        connection_string
            PostgreSQL connection string pointing to the target database.
            Example: ``"postgresql://user:pass@localhost:5432/my_store"``.
            The database will be created if it does not exist.
        embed
            Embedding provider for generating vector embeddings.
            If None, only full-text search will be available.
        overwrite
            Whether to drop and recreate the database if it already exists.
        name
            Internal name for the store.
        title
            Human-readable title for the store.
        attributes
            Optional schema for user-defined attribute columns stored per chunk.

        Returns
        -------
        PostgresStore
            A newly created store instance.
        """
        conninfo = psycopg.conninfo.conninfo_to_dict(connection_string)
        dbname = conninfo.get("dbname", "raghilda")

        # Connect to the default 'postgres' database to create/drop the target
        admin_conninfo = dict(conninfo)
        admin_conninfo["dbname"] = "postgres"
        admin_conn_str = psycopg.conninfo.make_conninfo(**admin_conninfo)

        with psycopg.connect(admin_conn_str, autocommit=True) as admin_con:
            if overwrite:
                admin_con.execute(
                    f"DROP DATABASE IF EXISTS {_quote_identifier(dbname)}"
                )
            # Check if database exists
            row = admin_con.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", [dbname]
            ).fetchone()
            if row is not None and not overwrite:
                raise ValueError(f"Database already exists: {dbname}")
            if row is None:
                admin_con.execute(f"CREATE DATABASE {_quote_identifier(dbname)}")

        # Now connect to the target database
        con = psycopg.connect(connection_string, autocommit=False)

        # Enable pgvector
        con.execute("CREATE EXTENSION IF NOT EXISTS vector")
        con.commit()

        if name is None:
            name = "raghilda_db"
        if title is None:
            title = "Raghilda PostgreSQL Store"

        attributes_spec = normalize_attributes_spec(
            attributes=attributes,
            reserved_columns=RESERVED_SYSTEM_COLUMNS,
        )
        attributes_schema = {
            key: spec.attribute_type for key, spec in attributes_spec.items()
        }

        if embed is None:
            embedding_column_sql = None
        else:
            embedding_size = len(embed.embed(["foo"])[0])
            embedding_column_sql = f"embedding vector({embedding_size})"

        embed_config_json = None
        if embed is not None:
            embed_config_json = json.dumps(embed.get_config())

        attributes_schema_json = json.dumps(
            attributes_spec_to_json_dict(attributes_spec)
        )
        attribute_column_defs = _postgres_attribute_column_defs(
            attributes_schema=attributes_schema,
        )
        tail_columns = list(attribute_column_defs)
        if embedding_column_sql is not None:
            tail_columns.append(embedding_column_sql)
        tail_columns_sql = ""
        if tail_columns:
            tail_columns_sql = ",\n            " + ",\n            ".join(tail_columns)

        con.execute(
            f"""
        CREATE TABLE IF NOT EXISTS metadata (
            name TEXT,
            title TEXT,
            embed_config TEXT,
            attributes_schema_json TEXT
        );

        CREATE TABLE IF NOT EXISTS documents (
            origin TEXT PRIMARY KEY,
            text TEXT
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            origin TEXT NOT NULL REFERENCES documents (origin),
            chunk_id SERIAL,
            start_index INTEGER,
            end_index INTEGER,
            char_count INTEGER,
            context TEXT,
            search_vector TSVECTOR,
            PRIMARY KEY (origin, start_index, end_index){tail_columns_sql}
        );

        CREATE OR REPLACE VIEW chunks AS (
            SELECT
            d.origin AS origin,
            e.chunk_id,
            e.start_index,
            e.end_index,
            e.char_count,
            e.context,
            substring(d.text FROM e.start_index + 1 FOR e.end_index - e.start_index) AS text
            FROM
            documents d
            JOIN
            embeddings e
            ON d.origin = e.origin
        );
        """
        )

        con.execute(
            """
            INSERT INTO metadata (
                name, title, embed_config, attributes_schema_json
            ) VALUES (%s, %s, %s, %s)
            """,
            [name, title, embed_config_json, attributes_schema_json],
        )
        con.commit()

        metadata = PostgresStoreMetadata(
            name=name,
            title=title,
            embed=embed,
            attributes=attributes_spec,
        )

        return PostgresStore(con, metadata)

    @staticmethod
    def connect(connection_string: str):
        """Connect to an existing PostgreSQL store.

        Parameters
        ----------
        connection_string
            PostgreSQL connection string.

        Returns
        -------
        PostgresStore
            A connected store instance.
        """
        con = psycopg.connect(connection_string, autocommit=False)
        _check_is_raghilda_con(con)

        row = con.execute(
            "SELECT name, title, embed_config, attributes_schema_json FROM metadata"
        ).fetchone()

        if row is None:
            raise ValueError("No metadata found in the database")

        name, title, embed_config_json, attributes_schema_json = row

        embed = None
        if embed_config_json is not None:
            embed_config = json.loads(embed_config_json)
            try:
                embed = embedding_from_config(embed_config)
            except ValueError as e:
                logger.warning(f"Could not restore embedding provider: {e}")

        if attributes_schema_json is None:
            raise ValueError("Missing attributes_schema_json in metadata table")
        attributes_spec = attributes_spec_from_json_dict(
            json.loads(attributes_schema_json),
        )

        metadata = PostgresStoreMetadata(
            name=name,
            title=title,
            embed=embed,
            attributes=attributes_spec,
        )

        return PostgresStore(con, metadata)

    def __init__(
        self,
        con: psycopg.Connection,
        metadata: EmbeddedAttributesStoreMetadata,
    ):
        self.con = con
        self.metadata = metadata
        self._db_lock = threading.Lock()

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
            existing_rows = self._get_existing_documents_by_origin(document.origin)
            existing = existing_rows[0] if existing_rows else None
            if (
                skip_if_unchanged
                and existing is not None
                and existing["text"] == document.content
                and self._chunk_layout_matches_existing(
                    chunked_doc=document,
                    origin=existing["origin"],
                )
            ):
                current_document = self._load_document_snapshot(
                    origin=existing["origin"],
                    text=existing["text"],
                )
                return WriteResult(
                    action="skipped",
                    document=current_document,
                )

        doc_row, chunk_rows = self._prepare_chunked_document_rows(document)

        with self._db_lock:
            existing_rows = self._get_existing_documents_by_origin(document.origin)
            existing = existing_rows[0] if existing_rows else None
            if (
                skip_if_unchanged
                and existing is not None
                and existing["text"] == document.content
                and self._chunk_layout_matches_existing(
                    chunked_doc=document,
                    origin=existing["origin"],
                )
            ):
                current_document = self._load_document_snapshot(
                    origin=existing["origin"],
                    text=existing["text"],
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
                )

            try:
                if action == "replaced":
                    self.con.execute(
                        "DELETE FROM embeddings WHERE origin = %s",
                        [doc_row["origin"]],
                    )
                    self.con.execute(
                        "UPDATE documents SET text = %s WHERE origin = %s",
                        [doc_row["text"], doc_row["origin"]],
                    )
                else:
                    _postgres_insert(self.con, "documents", [doc_row])
                _postgres_insert_embeddings(self.con, chunk_rows)
                self.con.commit()
            except Exception:
                try:
                    self.con.rollback()
                except Exception:
                    pass
                raise

            current_document = self._load_document_snapshot(
                origin=str(doc_row["origin"]),
                text=document.content,
            )
            return WriteResult(
                action=action,
                document=current_document,
                replaced_document=replaced_document,
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
                # pgvector expects a string like '[1.0,2.0,3.0]'
                row["embedding"] = (
                    "[" + ",".join(str(x) for x in embedded_chunks[index]) + "]"
                )
            else:
                row.pop("embedding", None)

            for column in self.metadata.attributes_schema:
                value = resolved_chunk_attributes[index][column]
                # JSONB columns need psycopg Json wrapper
                from ._attributes import AttributeStructType

                attr_type = self.metadata.attributes_schema[column]
                if isinstance(attr_type, AttributeStructType) and value is not None:
                    value = psycopg.types.json.Jsonb(value)
                row[column] = value

            row["origin"] = doc["origin"]
            # Store chunk text for search_vector computation
            chunk_text = chunked_doc.chunks[index].text
            context_text = chunked_doc.chunks[index].context or ""
            row["_search_text"] = f"{context_text} {chunk_text}".strip()
            chunk_rows.append(row)

        return doc, chunk_rows

    def _get_existing_documents_by_origin(self, origin: str) -> list[dict[str, str]]:
        rows = self.con.execute(
            "SELECT origin, text FROM documents WHERE origin = %s ORDER BY origin",
            [origin],
        ).fetchall()
        return [
            {
                "origin": row[0],
                "text": row[1],
            }
            for row in rows
        ]

    def _chunk_layout_matches_existing(
        self, *, chunked_doc: ChunkedMarkdownDocument, origin: str
    ) -> bool:
        incoming = self._chunk_layout_records(chunked_doc)
        existing = self._chunk_layout_records_from_store(origin)
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

    def _chunk_layout_records_from_store(self, origin: str) -> list[tuple[Any, ...]]:
        attributes_columns = list(self.metadata.attributes_schema)
        attribute_select = ", ".join(
            _quote_identifier(col) for col in attributes_columns
        )
        if attribute_select:
            attribute_select = ", " + attribute_select
        cur = self.con.execute(
            f"""
            SELECT
                e.start_index,
                e.end_index,
                e.char_count,
                e.context
                {attribute_select}
            FROM embeddings e
            WHERE e.origin = %s
            ORDER BY e.start_index, e.end_index
            """,
            [origin],
        )
        rows = cur.fetchall()
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
        self, *, origin: str, text: str
    ) -> ChunkedMarkdownDocument:
        attribute_columns = list(self.metadata.attributes_schema)
        attribute_select = ", ".join(
            _quote_identifier(col) for col in attribute_columns
        )
        if attribute_select:
            attribute_select = ", " + attribute_select
        cur = self.con.execute(
            f"""
            SELECT
                start_index,
                end_index,
                char_count,
                context
                {attribute_select}
            FROM embeddings
            WHERE origin = %s
            ORDER BY start_index, end_index
            """,
            [origin],
        )
        rows = cur.fetchall()
        if cur.description is None:
            raise RuntimeError("Failed to load replaced document snapshot.")
        columns = [desc[0] for desc in cur.description]

        chunks: list[Chunk] = []
        document_attributes: dict[str, Any] = {}
        for row in rows:
            row_dict = dict(zip(columns, row))
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

    def retrieve(
        self,
        text: str,
        top_k: int = 3,
        *,
        deoverlap: bool = True,
        attributes_filter: Optional[AttributeFilter] = None,
    ) -> Sequence[RetrievedStoreMarkdownChunk]:
        """Retrieve the most similar chunks to the given text.

        Combines results from vector similarity search (if embeddings are
        available) and full-text search, then optionally merges overlapping
        chunks.
        """
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

        # combine chunks by origin and backend chunk id, then merge metrics
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
        """Retrieve chunks using pgvector similarity search."""
        if isinstance(query, str):
            if self.metadata.embed is None:
                raise ValueError("No embedding function available in the store")
            query = self.metadata.embed.embed([query], EmbedInputType.QUERY)[0]

        operator, order = _pgvector_method_info(method)
        allowed_filter_columns = self._filterable_columns()
        compiled_filter = compile_filter_to_sql_postgres(
            attributes_filter,
            allowed_columns=allowed_filter_columns,
        )
        where_clause = f"WHERE {compiled_filter}" if compiled_filter else ""
        attribute_select = _attributes_select_clause(
            alias="e", attributes_schema=self.metadata.attributes_schema
        )
        query_vector = "[" + ",".join(str(x) for x in query) + "]"

        text_slice_sql = (
            "substring(doc.text FROM e.start_index + 1 FOR e.end_index - e.start_index)"
        )

        if compiled_filter is None:
            source_sql = f"""
            (
                SELECT
                    *,
                    embedding {operator} %s::vector AS metric_value
                FROM embeddings
                ORDER BY metric_value {order}
                LIMIT {top_k}
            ) AS e
            """
            metric_value_sql = "e.metric_value"
        else:
            source_sql = "embeddings e"
            metric_value_sql = f"e.embedding {operator} %s::vector"

        sql = f"""
        SELECT
            e.chunk_id,
            doc.origin AS origin,
            e.start_index,
            e.end_index,
            e.char_count,
            e.context,
            {attribute_select}
            {text_slice_sql} AS text,
            '{method}' AS metric_name,
            {metric_value_sql} AS metric_value
        FROM {source_sql}
        JOIN documents doc ON doc.origin = e.origin
        {where_clause}
        ORDER BY metric_value {order}
        LIMIT {top_k}
        """

        with self._db_lock:
            cur = self.con.execute(sql, [query_vector])
            rows = cur.fetchall()

            if cur.description is None:
                raise RuntimeError("Failed get result description.")

            columns = [desc[0] for desc in cur.description]

        return _rows_to_retrieved_chunks(rows, columns, self.metadata.attributes_schema)

    def retrieve_fts(
        self,
        query: str,
        top_k: int,
        *,
        attributes_filter: Optional[AttributeFilter] = None,
    ) -> list[RetrievedStoreMarkdownChunk]:
        """Retrieve chunks using PostgreSQL full-text search (ts_rank)."""
        allowed_filter_columns = self._filterable_columns()
        compiled_filter = compile_filter_to_sql_postgres(
            attributes_filter,
            allowed_columns=allowed_filter_columns,
        )
        where_clause = f"WHERE {compiled_filter}" if compiled_filter else ""
        attribute_select = _attributes_select_clause(
            alias="e", attributes_schema=self.metadata.attributes_schema
        )
        text_slice_sql = (
            "substring(doc.text FROM e.start_index + 1 FOR e.end_index - e.start_index)"
        )

        sql = f"""
        WITH ranked AS (
            SELECT
                e.chunk_id,
                doc.origin AS origin,
                e.start_index,
                e.end_index,
                e.char_count,
                e.context,
                {attribute_select}
                {text_slice_sql} AS text,
                'fts' AS metric_name,
                ts_rank(e.search_vector, plainto_tsquery('english', %(query)s)) AS metric_value
            FROM embeddings e
            JOIN documents doc ON doc.origin = e.origin
            {where_clause}
        )
        SELECT *
        FROM ranked
        WHERE metric_value > 0
        ORDER BY metric_value DESC
        LIMIT %(top_k)s
        """

        with self._db_lock:
            cur = self.con.execute(
                sql,
                {"query": query, "top_k": top_k},
            )
            rows = cur.fetchall()

            if cur.description is None:
                raise RuntimeError("Failed get result description.")

            columns = [desc[0] for desc in cur.description]

        return _rows_to_retrieved_chunks(rows, columns, self.metadata.attributes_schema)

    def build_index(
        self,
        type: Optional[IndexType | str | list[IndexType | str]] = None,
    ):
        """Build indexes on the embeddings table.

        Parameters
        ----------
        type
            The type of index to build. Can be ``"fts"`` or ``"hnsw"``
            or a list of those. If None, builds both.
        """
        if type is None:
            index_types = [IndexType.FTS, IndexType.HNSW]
        elif isinstance(type, (IndexType, str)):
            index_types = [_coerce_index_type(type)]
        else:
            index_types = [_coerce_index_type(item) for item in type]

        if IndexType.FTS in index_types:
            self.con.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_search_vector "
                "ON embeddings USING GIN (search_vector)"
            )
            self.con.commit()

        if IndexType.HNSW in index_types:
            self.con.execute("DROP INDEX IF EXISTS store_hnsw_cosine_index")
            self.con.execute("DROP INDEX IF EXISTS store_hnsw_l2_index")
            self.con.execute("DROP INDEX IF EXISTS store_hnsw_ip_index")
            self.con.execute(
                "CREATE INDEX store_hnsw_cosine_index "
                "ON embeddings USING hnsw (embedding vector_cosine_ops)"
            )
            self.con.execute(
                "CREATE INDEX store_hnsw_l2_index "
                "ON embeddings USING hnsw (embedding vector_l2_ops)"
            )
            self.con.execute(
                "CREATE INDEX store_hnsw_ip_index "
                "ON embeddings USING hnsw (embedding vector_ip_ops)"
            )
            self.con.commit()

    def size(self) -> int:
        with self._db_lock:
            result = self.con.execute(
                "SELECT COUNT(DISTINCT origin) FROM documents"
            ).fetchone()
            if result is None:
                raise RuntimeError("Failed to get size of the store")
            return result[0]

    def _filterable_columns(self) -> set[str]:
        filterable_attribute_columns = filterable_attribute_paths(
            self.metadata.attributes_schema
        )
        return FILTERABLE_BASE_COLUMNS | filterable_attribute_columns


# --- Helper functions ---


def _postgres_attribute_column_defs(
    *,
    attributes_schema: Mapping[str, AttributeType],
) -> list[str]:
    if not attributes_schema:
        return []
    lines: list[str] = []
    for column, attribute_type in attributes_schema.items():
        sql_type = postgres_sql_type_for_attribute_type(attribute_type)
        lines.append(f"{_quote_identifier(column)} {sql_type}")
    return lines


def _check_is_raghilda_con(con: psycopg.Connection) -> None:
    row = con.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'metadata')"
    ).fetchone()
    if row is None or not row[0]:
        raise ValueError("Not a valid Raghilda database connection")


def _postgres_insert(
    con: psycopg.Connection,
    table: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return
    columns = list(rows[0])
    column_list = ", ".join(_quote_identifier(c) for c in columns)
    placeholders = ", ".join("%s" for _ in columns)
    values = [tuple(row[c] for c in columns) for row in rows]
    cur = con.cursor()
    cur.executemany(
        f"INSERT INTO {_quote_identifier(table)} ({column_list}) VALUES ({placeholders})",
        values,
    )


def _postgres_insert_embeddings(
    con: psycopg.Connection,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Insert embedding rows with computed search_vector from _search_text."""
    if not rows:
        return
    for row in rows:
        search_text = row.get("_search_text", "")
        # Build column list excluding _search_text, adding search_vector
        columns = [c for c in row if c != "_search_text"]
        columns.append("search_vector")
        column_list = ", ".join(_quote_identifier(c) for c in columns)
        placeholders = ", ".join(
            "to_tsvector('english', %s)" if c == "search_vector" else "%s"
            for c in columns
        )
        values = tuple(search_text if c == "search_vector" else row[c] for c in columns)
        con.execute(
            f"INSERT INTO embeddings ({column_list}) VALUES ({placeholders})",
            values,
        )


def _pgvector_method_info(method: VSSMethod) -> tuple[str, str]:
    """Returns the pgvector operator and sort order for a VSSMethod."""
    method_mapping = {
        VSSMethod.COSINE_DISTANCE: ("<=>", "ASC"),
        VSSMethod.EUCLIDEAN_DISTANCE: ("<->", "ASC"),
        VSSMethod.NEGATIVE_INNER_PRODUCT: ("<#>", "ASC"),
    }
    if method not in method_mapping:
        raise ValueError(f"Unknown method: {method}")
    return method_mapping[method]


def _rows_to_retrieved_chunks(
    rows: list[tuple[Any, ...]],
    columns: list[str],
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
