from ._store import BaseStore, WriteResult
from collections.abc import Sized
import json
import os
import threading
from .embedding import EmbeddingProvider, EmbedInputType, embedding_from_config
from .chunk import Chunk, MarkdownChunk, RetrievedChunk, Metric
from .chunker import MarkdownChunker
from .read import read_as_markdown
from .document import Document, MarkdownDocument
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
import duckdb
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
from enum import StrEnum
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
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
    compile_filter_to_sql,
    duckdb_sql_type_for_attribute_type,
    filterable_attribute_paths,
    normalize_attributes_spec,
    merge_attribute_values,
)
from ._utils import lazy_map
from ._store_metadata import (
    EmbeddedAttributesStoreMetadata,
    attributes_schema_from_spec,
)


logger = logging.getLogger(__name__)

_RESERVED_SYSTEM_COLUMNS = {
    "chunk_id",
    "context",
    "embedding",
    "origin",
    "text",
    "start_index",
    "end_index",
    "token_count",
    "metric_name",
    "metric_value",
}

_FILTERABLE_BASE_COLUMNS = {
    "chunk_id",
    "origin",
    "start_index",
    "end_index",
    "context",
}


@dataclass(repr=False)
class DuckDBMarkdownChunk(MarkdownChunk):
    """MarkdownChunk with DuckDB-specific fields for database storage"""

    def __init__(
        self,
        text: str,
        start_index: int,
        end_index: int,
        context=None,
        token_count=None,
        origin=None,
        attributes=None,
    ):
        # Compute token_count if not provided
        if token_count is None:
            token_count = len(text)

        # Initialize parent class
        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            token_count=token_count,
            context=context,
            origin=origin,
            attributes=attributes,
        )


@dataclass(repr=False)
class RetrievedDuckDBMarkdownChunk(DuckDBMarkdownChunk, RetrievedChunk):
    """DuckDBMarkdownChunk with retrieval metrics"""

    def __init__(
        self,
        text: str,
        start_index: int,
        end_index: int,
        context=None,
        token_count=None,
        origin=None,
        metrics=None,
        chunk_ids=None,
        attributes=None,
    ):
        # Initialize DuckDBMarkdownChunk
        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            context=context,
            token_count=token_count,
            origin=origin,
            attributes=attributes,
        )

        # Initialize metrics
        if metrics is None:
            metrics = []
        self.metrics = metrics
        if chunk_ids is None:
            chunk_ids = []
        self.chunk_ids = chunk_ids


@dataclass
class DuckDBStoreMetadata(EmbeddedAttributesStoreMetadata):
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


class VSSMethod(StrEnum):
    COSINE_DISTANCE = "cosine_distance"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    NEGATIVE_INNER_PRODUCT = "negative_inner_product"


class IndexType(StrEnum):
    BM25 = "bm25"
    HNSW = "hnsw"


class DuckDBStore(BaseStore):
    """A vector store backed by DuckDB.

    DuckDBStore provides local vector storage with support for both
    semantic search (using embeddings) and full-text search (using BM25).
    Data is persisted to a DuckDB database file.

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.store import DuckDBStore
    from raghilda.embedding import EmbeddingOpenAI

    # Create a new store with embeddings
    store = DuckDBStore.create(
        location="my_store.db",
        embed=EmbeddingOpenAI(),
    )

    # Insert documents
    store.ingest(["https://example.com/doc1.md", "https://example.com/doc2.md"])

    # Retrieve similar chunks
    chunks = store.retrieve("How do I use this?", top_k=5)
    ```
    """

    @staticmethod
    def connect(
        location: str | Path = ":memory:",
        read_only: bool = False,
    ):
        """Connect to an existing DuckDB store.

        Parameters
        ----------
        location
            Path to the DuckDB database file.
        read_only
            Whether to open the database in read-only mode.
        Returns
        -------
        DuckDBStore
            A connected store instance.
        """
        con = duckdb.connect(database=location, read_only=read_only)
        _check_is_raghilda_con(con)

        row = con.execute(
            "SELECT name, title, embed_config, attributes_schema_json FROM metadata"
        ).fetchone()

        if row is None:
            raise ValueError("No metadata found in the database")

        name, title, embed_config_json, attributes_schema_json = row

        # Restore embedding provider from config
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

        metadata = DuckDBStoreMetadata(
            name=name,
            title=title,
            embed=embed,
            attributes=attributes_spec,
        )

        return DuckDBStore(con, metadata)

    @staticmethod
    def create(
        location: str | Path,
        embed: Optional[EmbeddingProvider],
        overwrite: bool = False,
        name: Optional[str] = None,
        title: Optional[str] = None,
        attributes: Optional[AttributesSchemaSpec] = None,
    ):
        """Create a new DuckDB store.

        Parameters
        ----------
        location
            Path where the DuckDB database file will be created.
        embed
            Embedding provider for generating vector embeddings.
            If None, only full-text search will be available.
        overwrite
            Whether to overwrite an existing database at the location.
        name
            Internal name for the store.
        title
            Human-readable title for the store.
        attributes
            Optional schema for user-defined attribute columns stored per chunk.
            Example: `{"tenant": str, "priority": int}`.
            Attribute names use identifier-style syntax.
            Built-in backend columns that cannot be declared as attributes are:
            `chunk_id`, `origin`, `start_index`, `end_index`, and `context`.

        Returns
        -------
        DuckDBStore
            A newly created store instance.
        """
        _overwrite_or_error(location, overwrite)
        con = duckdb.connect(database=location)

        if name is None:
            name = "raghilda_db"

        if title is None:
            title = "Raghilda DuckDB Store"

        attributes_spec = normalize_attributes_spec(
            attributes=attributes,
            reserved_columns=_RESERVED_SYSTEM_COLUMNS,
        )
        attributes_schema = {
            key: spec.attribute_type for key, spec in attributes_spec.items()
        }

        if embed is None:
            embedding_column_sql = None
        else:
            embedding_size = len(embed.embed(["foo"])[0])
            embedding_column_sql = f"embedding FLOAT[{embedding_size}]"

        # Get embed config as JSON if provider is given
        embed_config_json = None
        if embed is not None:
            embed_config_json = json.dumps(embed.get_config())

        attributes_schema_json = json.dumps(
            attributes_spec_to_json_dict(attributes_spec)
        )
        attribute_column_defs_sql = _duckdb_attribute_column_defs(
            attributes_schema=attributes_schema,
        )
        tail_columns = list(attribute_column_defs_sql)
        if embedding_column_sql is not None:
            tail_columns.append(embedding_column_sql)
        tail_columns_sql = ""
        if tail_columns:
            tail_columns_sql = ",\n            " + ",\n            ".join(tail_columns)

        con.execute(
            f"""
        CREATE SEQUENCE chunk_id_seq START 1; -- need a unique id for fts

        CREATE OR REPLACE TABLE metadata (
            name VARCHAR,
            title VARCHAR,
            embed_config VARCHAR,
            attributes_schema_json VARCHAR
        );

        CREATE OR REPLACE TABLE documents (
            origin VARCHAR PRIMARY KEY,
            text VARCHAR
        );

        CREATE OR REPLACE TABLE embeddings (
            origin VARCHAR NOT NULL,
            FOREIGN KEY (origin) REFERENCES documents (origin),
            chunk_id INTEGER DEFAULT nextval('chunk_id_seq'),
            start_index INTEGER,
            end_index INTEGER,
            PRIMARY KEY (origin, start_index, end_index),
            context VARCHAR{tail_columns_sql}
        );

        CREATE OR REPLACE VIEW chunks AS (
            SELECT
            d.origin as origin,
            e.*,
            d.text[e.start_index + 1:e.end_index] as text
            FROM
            documents d
            JOIN
            embeddings e
            USING
            (origin)
        );
        """
        )

        # Insert metadata
        con.execute(
            """
            INSERT INTO metadata (
                name,
                title,
                embed_config,
                attributes_schema_json
            ) VALUES (?, ?, ?, ?)
            """,
            [
                name,
                title,
                embed_config_json,
                attributes_schema_json,
            ],
        )

        return DuckDBStore(
            con,
            DuckDBStoreMetadata(
                name=name,
                title=title,
                embed=embed,
                attributes=attributes_spec,
            ),
        )

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        metadata: EmbeddedAttributesStoreMetadata,
    ):
        self.con = con
        self.metadata = metadata
        _validate_required_schema(
            con=self.con,
            attributes_schema=self.metadata.attributes_schema,
            require_embedding=self.metadata.embed is not None,
        )
        self._db_lock = threading.Lock()

    def upsert(
        self,
        document: Document,
        *,
        skip_if_unchanged: bool = True,
    ) -> WriteResult:
        if not isinstance(document, MarkdownDocument):
            raise NotImplementedError(
                f"Upsert not implemented for type {type(document)}"
            )
        if not isinstance(document.origin, str) or not document.origin:
            raise ValueError("document.origin must be a non-empty string for upsert().")
        if document.chunks is None:
            raise ValueError("Document must be chunked before insertion.")
        if len(document.chunks) == 0:
            raise ValueError("Document must contain at least one chunk.")
        for chunk in document.chunks:
            _validate_chunk_against_document(
                document_origin=document.origin,
                content=document.content,
                chunk=chunk,
            )

        # DuckDB connections are not thread-safe for reads or writes.
        # Hold the lock here to avoid unnecessary embedding work when the
        # existing stored content/chunk layout is identical.
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
            replaced_document: MarkdownDocument | None = None
            if existing is not None:
                action = "replaced"
                replaced_document = self._load_document_snapshot(
                    origin=existing["origin"],
                    text=existing["text"],
                )

            try:
                self.con.begin()
                if action == "replaced":
                    self.con.execute(
                        "DELETE FROM embeddings WHERE origin = ?",
                        [doc_row["origin"]],
                    )
                    self.con.execute(
                        "UPDATE documents SET text = ? WHERE origin = ?",
                        [doc_row["text"], doc_row["origin"]],
                    )
                else:
                    _duckdb_append(self.con, "documents", [doc_row])
                _duckdb_append(self.con, "embeddings", chunk_rows)
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

    def ingest(
        self,
        items: Iterable[Any],
        prepare: Optional[Callable[[Any], Document]] = None,
        num_workers: Optional[int] = None,
        progress: bool = True,
    ) -> None:
        """
        Ingest multiple documents in parallel.

        This method processes items through a prepare function to create Documents,
        then inserts them into the store. Items are processed in parallel using
        a thread pool for improved performance.

        Parameters
        ----------
        items
            An iterable of items to ingest. Can be any iterable including lists,
            generators, or other iterables. Each item will be passed to the prepare
            function to create a Document. By default, items are expected to be
            URIs (file paths or URLs) that will be read with read_as_markdown
            and chunked automatically.
        prepare
            A callable that takes an item and returns a Document with chunks computed.
            Use this to customize how items are converted to documents. The function
            should handle chunking if needed. If None, items are treated as URIs
            and processed with read_as_markdown followed by MarkdownChunker.
        num_workers
            The number of worker threads to use for parallel ingestion.
            If None, defaults to 4 to avoid API rate limiting.
        progress
            Whether to display a progress bar during ingestion. Default is True.
            The progress bar shows the total count only if items has a known length
            (e.g., a list). For generators, it shows progress without a total.

        Examples
        --------
        Ingest files from a list of paths:

        ```{python}
        #| eval: false
        store.ingest(["doc1.md", "doc2.pdf", "doc3.html"])
        ```

        Ingest from a generator:

        ```{python}
        #| eval: false
        def get_urls():
            for url in scrape_sitemap("https://example.com/sitemap.xml"):
                yield url

        store.ingest(get_urls())
        ```

        Ingest with a custom prepare function:

        ```{python}
        #| eval: false
        from raghilda.chunker import MarkdownChunker

        chunker = MarkdownChunker()

        def prepare_record(record: dict) -> MarkdownDocument:
            doc = MarkdownDocument(
                origin=record["id"],
                content=record["text"]
            )
            return chunker.chunk_document(doc)

        records = [{"id": "1", "text": "Hello"}, {"id": "2", "text": "World"}]
        store.ingest(records, prepare=prepare_record)
        ```
        """
        if num_workers is None:
            num_workers = 4

        if prepare is None:
            chunker = MarkdownChunker()

            def default_prepare(uri: str) -> Document:
                return chunker.chunk_document(read_as_markdown(uri))

            prepare = default_prepare

        total = len(items) if isinstance(items, Sized) else None

        def do_ingest_work(item: Any) -> None:
            try:
                doc = prepare(item)
                self.upsert(doc)
            except Exception as e:
                raise RuntimeError(f"Failed to ingest '{item}': {e}") from e

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for future in tqdm(
                lazy_map(pool, do_ingest_work, items), total=total, disable=not progress
            ):
                future.result()

    def _prepare_chunked_document_rows(
        self,
        chunked_doc: MarkdownDocument,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        assert chunked_doc.chunks is not None

        doc = {
            "origin": chunked_doc.origin,
            "text": chunked_doc.content,
        }
        chunks = [asdict(chunk) for chunk in chunked_doc.chunks]

        resolved_chunk_attributes: list[dict[str, AttributeValue]] = []
        for chunk in chunked_doc.chunks or []:
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

            row.pop("token_count", None)
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
                row[column] = resolved_chunk_attributes[index][column]

            row["origin"] = doc["origin"]
            chunk_rows.append(row)

        return doc, chunk_rows

    def _get_existing_documents_by_origin(self, origin: str) -> list[dict[str, str]]:
        rows = self.con.execute(
            "SELECT origin, text FROM documents WHERE origin = ? ORDER BY origin",
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
        self, *, chunked_doc: MarkdownDocument, origin: str
    ) -> bool:
        incoming = self._chunk_layout_records(chunked_doc)
        existing = self._chunk_layout_records_from_store(origin)
        return incoming == existing

    def _chunk_layout_records(
        self, chunked_doc: MarkdownDocument
    ) -> list[tuple[Any, ...]]:
        records: list[tuple[Any, ...]] = []
        attributes_columns = list(self.metadata.attributes_schema)
        for chunk in chunked_doc.chunks or []:
            resolved = merge_attribute_values(
                attributes_spec=self.metadata.attributes_spec,
                sources=[chunked_doc.attributes, chunk.attributes],
            )
            row: list[Any] = [
                chunk.start_index,
                chunk.end_index,
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
        result = self.con.execute(
            f"""
            SELECT
                e.start_index,
                e.end_index,
                e.context
                {attribute_select}
            FROM embeddings e
            WHERE e.origin = ?
            ORDER BY e.start_index, e.end_index
            """,
            [origin],
        )
        rows = result.fetchall()
        records: list[tuple[Any, ...]] = []
        for row in rows:
            start_index = int(row[0])
            end_index = int(row[1])
            context = row[2]
            attribute_values = [
                self._coerce_chunk_layout_attribute_value(col, row[3 + idx])
                for idx, col in enumerate(attributes_columns)
            ]
            records.append((start_index, end_index, context, *attribute_values))
        return records

    def _coerce_chunk_layout_attribute_value(self, column: str, value: Any) -> Any:
        return coerce_attribute_value_for_output(
            column,
            value,
            self.metadata.attributes_schema[column],
        )

    def _load_document_snapshot(self, *, origin: str, text: str) -> MarkdownDocument:
        attribute_columns = list(self.metadata.attributes_schema)
        attribute_select = ", ".join(
            _quote_identifier(col) for col in attribute_columns
        )
        if attribute_select:
            attribute_select = ", " + attribute_select
        result = self.con.execute(
            f"""
            SELECT
                start_index,
                end_index,
                context
                {attribute_select}
            FROM embeddings
            WHERE origin = ?
            ORDER BY start_index, end_index
            """,
            [origin],
        )
        rows = result.fetchall()
        if result.description is None:
            raise RuntimeError("Failed to load replaced document snapshot.")
        columns = [desc[0] for desc in result.description]

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
                    token_count=len(chunk_text),
                    context=row_dict.get("context"),
                    origin=origin,
                    attributes=attributes or None,
                )
            )

        return MarkdownDocument(
            origin=origin,
            content=text,
            chunks=chunks,
            attributes=document_attributes or None,
        )

    def retrieve(
        self,
        text: str,
        top_k: int = 3,
        *,
        deoverlap: bool = True,
        attributes_filter: Optional[AttributeFilter] = None,
    ) -> Sequence[RetrievedDuckDBMarkdownChunk]:
        """Retrieve the most similar chunks to the given text.

        Combines results from vector similarity search (if embeddings are available)
        and BM25 full-text search, then optionally merges overlapping chunks.

        Parameters
        ----------
        text
            The query text to search for.
        top_k
            The maximum number of chunks to return from each retrieval method.
        deoverlap
            If True (default), merge overlapping chunks from the same document.
            Overlapping chunks are identified by their `start_index` and `end_index`
            positions. When merged, the resulting chunk spans the union of the
            original ranges, combines metrics, and aggregates attribute values
            into per-chunk lists in start-order. The `context` value is kept
            from the first chunk in each merged overlap group.
        attributes_filter
            Optional filter to scope retrieval using attribute columns.
            Can be a SQL-like string or a dict AST.
            Example string: `"tenant = 'docs' AND priority >= 2"`.
            Supports declared attributes plus built-in columns:
            `chunk_id`, `origin`, `start_index`, `end_index`, and `context`.

        Returns
        -------
        Sequence[RetrievedDuckDBMarkdownChunk]
            The retrieved chunks with their relevance metrics.
        """
        retrieved_chunks = []
        if self.metadata.embed is not None:
            retrieved_chunks = self.retrieve_vss(
                text,
                top_k,
                attributes_filter=attributes_filter,
            )

        retrieved_chunks.extend(
            self.retrieve_bm25(
                text,
                top_k,
                attributes_filter=attributes_filter,
            )
        )

        # combine chunks by origin and backend chunk id, then merge metrics
        combined_chunks: dict[
            tuple[str | None, int | None], RetrievedDuckDBMarkdownChunk
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
    ) -> list[RetrievedDuckDBMarkdownChunk]:
        """Retrieve chunks using vector similarity search.

        Uses DuckDB's `vss` extension for vector similarity search.
        See https://duckdb.org/docs/extensions/vss.html for more details.

        Parameters
        ----------
        query
            The query text or embedding vector. If a string is provided,
            it will be embedded using the store's embedding provider.
        top_k
            The maximum number of chunks to return.
        method
            The similarity method to use. Options are:
            - `COSINE_DISTANCE`: Cosine distance (default)
            - `EUCLIDEAN_DISTANCE`: L2/Euclidean distance
            - `NEGATIVE_INNER_PRODUCT`: Negative dot product
        attributes_filter
            Optional attribute filter as SQL-like string or dict AST.
            Supports declared attributes plus built-in columns:
            `chunk_id`, `origin`, `start_index`, `end_index`, and `context`.

        Returns
        -------
        list[RetrievedDuckDBMarkdownChunk]
            The most similar chunks with similarity metrics.

        Raises
        ------
        ValueError
            If query is a string but no embedding provider is configured.
        """
        if isinstance(query, str):
            if self.metadata.embed is None:
                raise ValueError("No embedding function available in the store")
            query = self.metadata.embed.embed([query], EmbedInputType.QUERY)[0]

        func, order = _vss_method_info(method)
        allowed_filter_columns = self._filterable_columns()
        compiled_filter = compile_filter_to_sql(
            attributes_filter,
            allowed_columns=allowed_filter_columns,
        )
        where_clause = f"WHERE {compiled_filter}" if compiled_filter else ""
        attribute_select = _attributes_select_clause(
            alias="e", attributes_schema=self.metadata.attributes_schema
        )
        query_vector = f"[{','.join(str(x) for x in query)}]::FLOAT[{len(query)}]"

        if compiled_filter is None:
            source_sql = f"""
            (
                SELECT
                    *,
                    {func}(embedding, {query_vector}) AS metric_value
                FROM embeddings
                ORDER BY metric_value {order}
                LIMIT {top_k}
            ) AS e
            """
            metric_value_sql = "e.metric_value"
        else:
            source_sql = "embeddings e"
            metric_value_sql = f"{func}(e.embedding, {query_vector})"

        sql = f"""
        SELECT
            e.chunk_id,
            doc.origin AS origin,
            e.start_index,
            e.end_index,
            e.context,
            {attribute_select}
            doc.text[e.start_index + 1:e.end_index] AS text,
            '{method}' AS metric_name,
            {metric_value_sql} AS metric_value
        FROM {source_sql}
        JOIN documents doc USING (origin)
        {where_clause}
        ORDER BY metric_value {order}
        LIMIT {top_k}
        """

        with self._db_lock:
            result = self.con.execute(sql)
            rows = result.fetchall()

            if result.description is None:
                raise RuntimeError("Failed get result description.")

            columns = [desc[0] for desc in result.description]

        output: list[RetrievedDuckDBMarkdownChunk] = []
        for chunk in rows:
            chunk_dict = dict(zip(columns, chunk))
            name, value = chunk_dict.pop("metric_name"), chunk_dict.pop("metric_value")
            chunk_id = chunk_dict.pop("chunk_id", None)
            attribute_values: dict[str, AttributeValue] = {}
            for key, attribute_type in self.metadata.attributes_schema.items():
                if key in chunk_dict:
                    attribute_values[key] = coerce_attribute_value_for_output(
                        key,
                        chunk_dict.pop(key),
                        attribute_type,
                    )
            chunk_dict["chunk_ids"] = [] if chunk_id is None else [int(chunk_id)]
            chunk_dict["metrics"] = [Metric(name, value)]
            chunk_dict["attributes"] = attribute_values
            output.append(RetrievedDuckDBMarkdownChunk(**chunk_dict))

        return output

    def retrieve_bm25(
        self,
        query: str,
        top_k: int,
        *,
        k: float = 1.2,
        b: float = 0.75,
        conjunctive: bool = False,
        attributes_filter: Optional[AttributeFilter] = None,
    ) -> list[RetrievedDuckDBMarkdownChunk]:
        """Retrieve chunks using BM25 full-text search.

        Uses DuckDB's `fts` (Full-Text Search) extension for BM25 ranking.
        See https://duckdb.org/docs/extensions/full_text_search.html for more details.

        Parameters
        ----------
        query
            The search query text.
        top_k
            The maximum number of chunks to return.
        k
            BM25 term frequency saturation parameter. Higher values increase
            the impact of term frequency. Default is 1.2.
        b
            BM25 length normalization parameter (0-1). Higher values penalize
            longer documents more. Default is 0.75.
        conjunctive
            If True, all query terms must be present (AND). If False (default),
            any query term can match (OR).
        attributes_filter
            Optional attribute filter as SQL-like string or dict AST.
            Supports declared attributes plus built-in columns:
            `chunk_id`, `origin`, `start_index`, `end_index`, and `context`.

        Returns
        -------
        list[RetrievedDuckDBMarkdownChunk]
            The matching chunks ranked by BM25 score.
        """
        allowed_filter_columns = self._filterable_columns()
        compiled_filter = compile_filter_to_sql(
            attributes_filter,
            allowed_columns=allowed_filter_columns,
        )
        where_clause = f"WHERE {compiled_filter}" if compiled_filter else ""
        metric_not_null_clause = (
            "WHERE metric_value IS NOT NULL" if compiled_filter else ""
        )
        attribute_select = _attributes_select_clause(
            alias="e", attributes_schema=self.metadata.attributes_schema
        )

        sql = f"""
        WITH ranked AS (
            SELECT
                e.chunk_id, 
                doc.origin AS origin,
                e.start_index, 
                e.end_index, 
                e.context, 
                {attribute_select}
                doc.text[e.start_index + 1:e.end_index] AS text,
                'bm25' AS metric_name,
                fts_main_chunks.match_bm25(chunk_id, $query, k := $k, b := $b, conjunctive := $conjunctive) AS metric_value
            FROM embeddings e
            JOIN documents doc USING (origin)
            {where_clause}
        )
        SELECT *
        FROM ranked
        {metric_not_null_clause}
        ORDER BY metric_value DESC
        LIMIT $top_k
        """

        with self._db_lock:
            result = self.con.execute(
                sql,
                {
                    "query": query,
                    "top_k": top_k,
                    "k": k,
                    "b": b,
                    "conjunctive": conjunctive,
                },
            )
            rows = result.fetchall()

            if result.description is None:
                raise RuntimeError("Failed get result description.")

            columns = [desc[0] for desc in result.description]

        output: list[RetrievedDuckDBMarkdownChunk] = []
        for chunk in rows:
            chunk_dict = dict(zip(columns, chunk))
            name, value = chunk_dict.pop("metric_name"), chunk_dict.pop("metric_value")
            chunk_id = chunk_dict.pop("chunk_id", None)
            attribute_values: dict[str, AttributeValue] = {}
            for key, attribute_type in self.metadata.attributes_schema.items():
                if key in chunk_dict:
                    attribute_values[key] = coerce_attribute_value_for_output(
                        key,
                        chunk_dict.pop(key),
                        attribute_type,
                    )
            chunk_dict["chunk_ids"] = [] if chunk_id is None else [int(chunk_id)]
            chunk_dict["metrics"] = [Metric(name, value)]
            chunk_dict["attributes"] = attribute_values
            output.append(RetrievedDuckDBMarkdownChunk(**chunk_dict))

        return output

    def build_index(
        self,
        type: Optional[IndexType | str | list[IndexType | str]] = None,
    ):
        """
        Build the specified index types on the embeddings table.

        Parameters
        ----------
        type
            The type of index to build. Can be a single IndexType/string
            (`"bm25"` or `"hnsw"`) or a list of those values.
            If None, builds both BM25 and HNSW indexes.
        """
        if type is None:
            index_types = [IndexType.BM25, IndexType.HNSW]
        elif isinstance(type, (IndexType, str)):
            index_types = [_coerce_index_type(type)]
        else:
            index_types = [_coerce_index_type(item) for item in type]

        if IndexType.BM25 in index_types:
            self.con.execute("INSTALL FTS; LOAD FTS;")
            try:
                self.con.begin()
                self._create_fts_index()
                self.con.commit()
            except Exception as e:
                self.con.rollback()
                raise e

        if IndexType.HNSW in index_types:
            self.con.execute("INSTALL vss; LOAD vss;")
            try:
                self.con.begin()
                self._create_hnsw_index()
                self.con.commit()
            except Exception as e:
                self.con.rollback()
                raise e

    def _create_fts_index(self):
        self.con.execute(
            """
        ALTER VIEW chunks RENAME TO chunks_view;

        CREATE TABLE chunks AS
          SELECT chunk_id, context, text FROM chunks_view;
        """
        )

        self.con.execute(
            """
        PRAGMA create_fts_index(
          'chunks',            -- input_table
          'chunk_id',          -- input_id
          'context', 'text',   -- *input_values
          overwrite = 1
        );
        """
        )

        self.con.execute(
            """
        DROP TABLE chunks;
        ALTER VIEW chunks_view RENAME TO chunks;
        """
        )

    def _create_hnsw_index(self):
        self.con.execute(
            """
        SET hnsw_enable_experimental_persistence = true;

        DROP INDEX IF EXISTS store_hnsw_cosine_index;
        DROP INDEX IF EXISTS store_hnsw_l2sq_index;
        DROP INDEX IF EXISTS store_hnsw_ip_index;

        CREATE INDEX store_hnsw_cosine_index ON embeddings USING HNSW (embedding) WITH (metric = 'cosine');
        CREATE INDEX store_hnsw_l2sq_index   ON embeddings USING HNSW (embedding) WITH (metric = 'l2sq'); -- array_distance?
        CREATE INDEX store_hnsw_ip_index     ON embeddings USING HNSW (embedding) WITH (metric = 'ip');  -- array_dot_product
        """
        )

    def size(self) -> int:
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
        return _FILTERABLE_BASE_COLUMNS | filterable_attribute_columns


def _attributes_select_clause(
    alias: str, attributes_schema: Mapping[str, AttributeType]
) -> str:
    if not attributes_schema:
        return ""
    parts = [f"{alias}.{_quote_identifier(column)}," for column in attributes_schema]
    return "\n            " + "\n            ".join(parts) + "\n            "


def _duckdb_attribute_column_defs(
    *,
    attributes_schema: Mapping[str, AttributeType],
) -> list[str]:
    if not attributes_schema:
        return []

    lines: list[str] = []
    for column, attribute_type in attributes_schema.items():
        sql_type = duckdb_sql_type_for_attribute_type(attribute_type)
        lines.append(f"{_quote_identifier(column)} {sql_type}")
    return lines


def _overwrite_or_error(location: str | Path, overwrite: bool) -> None:
    if location == ":memory:":
        return

    location = Path(location)
    if os.path.exists(location) or os.path.exists(location / ".wal"):
        if overwrite:
            logger.info(f"Overwriting existing database at: {location}")
            if os.path.exists(location):
                os.remove(location)
            if os.path.exists(location / ".wal"):
                os.remove(location / ".wal")
        else:
            raise FileExistsError(f"File already exists: {location}")


def _check_is_raghilda_con(con: duckdb.DuckDBPyConnection):
    tables = con.execute("SHOW TABLES").fetchall()
    tables = [t[0] for t in tables]

    if "metadata" not in tables:
        raise ValueError("Not a valid Raghilda database connection")


def _validate_required_schema(
    con: duckdb.DuckDBPyConnection,
    *,
    attributes_schema: Mapping[str, AttributeType],
    require_embedding: bool,
) -> None:
    _assert_table_has_columns(
        con,
        table="metadata",
        required_columns={"name", "title", "embed_config", "attributes_schema_json"},
    )
    _assert_table_has_columns(
        con,
        table="documents",
        required_columns={"origin", "text"},
    )
    required_embeddings_columns = {
        "origin",
        "chunk_id",
        "start_index",
        "end_index",
        "context",
        *attributes_schema.keys(),
    }
    if require_embedding:
        required_embeddings_columns.add("embedding")
    _assert_table_has_columns(
        con,
        table="embeddings",
        required_columns=required_embeddings_columns,
    )


def _assert_table_has_columns(
    con: duckdb.DuckDBPyConnection,
    *,
    table: str,
    required_columns: set[str],
) -> None:
    columns = _table_columns(con, table=table)
    missing_columns = sorted(required_columns - columns)
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"Invalid DuckDB store schema: table '{table}' missing required columns: {missing}"
        )


def _table_columns(con: duckdb.DuckDBPyConnection, *, table: str) -> set[str]:
    try:
        rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    except duckdb.Error as exc:
        raise ValueError(
            f"Invalid DuckDB store schema: missing required table '{table}'"
        ) from exc
    if not rows:
        raise ValueError(
            f"Invalid DuckDB store schema: missing required table '{table}'"
        )
    return {str(row[1]) for row in rows}


def _duckdb_append(
    con: duckdb.DuckDBPyConnection, table: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    if not rows:
        return

    columns = list(rows[0])
    values = [tuple(row[column] for column in columns) for row in rows]
    column_list = ", ".join(_quote_identifier(column) for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    con.executemany(
        f"INSERT INTO {_quote_identifier(table)} ({column_list}) VALUES ({placeholders})",
        values,
    )


def _quote_identifier(identifier: str) -> str:
    """
    Quotes an identifier for use in SQL queries.
    """
    identifier = identifier.replace('"', '""')
    return f'"{identifier}"'


def _vss_method_info(method: VSSMethod) -> tuple[str, str]:
    """
    Returns the duckdb function name and ordering direction given a VSSMethod.
    """
    method_mapping = {
        VSSMethod.COSINE_DISTANCE: ("array_cosine_distance", "ASC"),
        VSSMethod.EUCLIDEAN_DISTANCE: ("array_distance", "ASC"),
        VSSMethod.NEGATIVE_INNER_PRODUCT: ("array_negative_inner_product", "ASC"),
    }

    if method not in method_mapping:
        raise ValueError(f"Unknown method: {method}")

    return method_mapping[method]


def _coerce_index_type(value: IndexType | str) -> IndexType:
    if isinstance(value, IndexType):
        return value
    try:
        return IndexType(value)
    except ValueError as exc:
        allowed = ", ".join(x.value for x in IndexType)
        raise ValueError(
            f"Unknown index type '{value}'. Allowed values: {allowed}"
        ) from exc


def _slice_chunk_text(content: str, *, start_index: int, end_index: int) -> str:
    if start_index < 0 or end_index < start_index or end_index > len(content):
        raise ValueError(
            "Chunk indices must satisfy 0 <= start_index <= end_index <= len(content). "
            f"Got start_index={start_index}, end_index={end_index}, "
            f"len(content)={len(content)}."
        )
    return content[start_index:end_index]


def _validate_chunk_against_document(
    *, document_origin: str, content: str, chunk: Chunk
) -> None:
    _validate_chunk_origin_matches_document_origin(
        document_origin=document_origin,
        chunk=chunk,
    )
    _validate_chunk_text_matches_document_content(
        content=content,
        chunk=chunk,
    )


def _validate_chunk_text_matches_document_content(
    *, content: str, chunk: Chunk
) -> None:
    expected_text = _slice_chunk_text(
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


def _validate_chunk_origin_matches_document_origin(
    *, document_origin: str, chunk: Chunk
) -> None:
    if chunk.origin is None:
        return
    if chunk.origin != document_origin:
        raise ValueError(
            "Chunk origin must be None or match document.origin. "
            f"Got chunk.origin={chunk.origin!r}, document.origin={document_origin!r}."
        )
