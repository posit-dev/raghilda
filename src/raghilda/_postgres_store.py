from ._store import BaseStore, WriteResult
import json
from .embedding import EmbeddingProvider, EmbedInputType, embedding_from_config
from .document import Document, ChunkedMarkdownDocument
from .chunk import Chunk, MarkdownChunk, RetrievedChunk, Metric
from ._deoverlap import deoverlap_chunks
from typing import Any, Mapping, Optional, Sequence, Union
from enum import StrEnum
import psycopg2
import logging

ConnectionLike = Union[str, psycopg2.extensions.connection]
from ._attributes import (
    AttributeFloatVectorType,
    AttributeStructType,
    AttributeType,
    AttributeValue,
    AttributesSchemaSpec,
    attributes_spec_from_json_dict,
    attributes_spec_to_json_dict,
    merge_attribute_values,
    normalize_attributes_spec,
)
from ._attribute_schema import AttributeFilter, filterable_attribute_paths
from ._attribute_filters import compile_filter_to_sql, pg_column_expression

logger = logging.getLogger(__name__)

_FILTERABLE_BASE_COLUMNS = {
    "chunk_id",
    "origin",
    "start_index",
    "end_index",
    "char_count",
    "context",
}

_RESERVED_SYSTEM_COLUMNS = {
    "chunk_id",
    "context",
    "embedding",
    "fts_search_vector",
    "origin",
    "text",
    "start_index",
    "end_index",
    "char_count",
}


def _postgres_sql_type_for_attribute_type(attribute_type: AttributeType) -> str:
    if attribute_type is str:
        return "VARCHAR"
    if attribute_type is int:
        return "INTEGER"
    if attribute_type is float:
        return "DOUBLE PRECISION"
    if attribute_type is bool:
        return "BOOLEAN"
    if isinstance(attribute_type, AttributeFloatVectorType):
        return f"vector({attribute_type.dimension})"
    if isinstance(attribute_type, AttributeStructType):
        return "JSONB"
    raise ValueError(f"Unsupported attribute type: {attribute_type}")


def _postgres_attribute_column_defs(
    *, attributes_schema: Mapping[str, AttributeType]
) -> list[str]:
    if not attributes_schema:
        return []
    lines: list[str] = []
    for column, attribute_type in attributes_schema.items():
        sql_type = _postgres_sql_type_for_attribute_type(attribute_type)
        lines.append(f'"{column}" {sql_type}')
    return lines


class VSSMethod(StrEnum):
    """Distance method for pgvector similarity search.

    Since this is a :class:`~enum.StrEnum`, you can pass string values
    directly (e.g. ``"cosine_distance"``) wherever a ``VSSMethod`` is
    expected.
    """

    COSINE_DISTANCE = "cosine_distance"
    L2_DISTANCE = "l2_distance"
    INNER_PRODUCT = "inner_product"

    @property
    def pg_operator(self) -> str:
        return {
            VSSMethod.COSINE_DISTANCE: "<=>",
            VSSMethod.L2_DISTANCE: "<->",
            VSSMethod.INNER_PRODUCT: "<#>",
        }[self]

    @property
    def metric_name(self) -> str:
        return self.value

    @property
    def pg_ops_class(self) -> str:
        return {
            VSSMethod.COSINE_DISTANCE: "vector_cosine_ops",
            VSSMethod.L2_DISTANCE: "vector_l2_ops",
            VSSMethod.INNER_PRODUCT: "vector_ip_ops",
        }[self]


def _resolve_connection(con: ConnectionLike) -> psycopg2.extensions.connection:
    if isinstance(con, str):
        return psycopg2.connect(con)
    return con


class PostgreSQLStore(BaseStore):
    """A store backed by a PostgreSQL database with pgvector.

    Uses PostgreSQL for storage with two retrieval methods:

    - **Full-text search** via :meth:`retrieve_fts`: uses PostgreSQL's
      built-in ``tsvector``/``tsquery`` with ``ts_rank`` for ranking.
      A pre-computed ``tsvector`` column with a GIN index is created
      automatically.
    - **Vector similarity search** via :meth:`retrieve_vss`: uses
      pgvector for nearest-neighbor search over embeddings. An HNSW
      index for cosine distance is created automatically when an
      embedding provider is given. Use :meth:`build_index` to add
      indexes for other distance methods (L2, inner product).
    """

    def __init__(
        self, con: psycopg2.extensions.connection, metadata: dict, schema: str
    ):
        self.con = con
        self._metadata = metadata
        self._schema = psycopg2.extensions.quote_ident(schema, con)

    def close(self) -> None:
        """Close the store's database connection."""
        if self.con and not self.con.closed:
            self.con.close()

    @staticmethod
    def create(
        con: ConnectionLike,
        embed: Optional[EmbeddingProvider],
        name: Optional[str] = None,
        title: Optional[str] = None,
        attributes: Optional[AttributesSchemaSpec] = None,
        vss_index: Optional[VSSMethod] = VSSMethod.COSINE_DISTANCE,
        schema: str = "raghilda",
        overwrite: bool = False,
    ) -> "PostgreSQLStore":
        """Create a new PostgreSQL store.

        Parameters
        ----------
        con
            A PostgreSQL connection string (e.g.
            ``"postgresql://user:pass@localhost/mydb"``).
        embed
            Embedding provider for generating vector embeddings.
            If None, only full-text search will be available.
        name
            Internal name for the store.
        title
            Human-readable title for the store.
        attributes
            Optional schema for user-defined attribute columns stored per chunk.
        vss_index
            The distance method to build an HNSW index for. Defaults to
            cosine distance. Set to ``None`` to skip creating a VSS index.
            Ignored when ``embed`` is ``None``.
        schema
            PostgreSQL schema to create the store tables in. Defaults to
            ``"raghilda"``. The schema is created if it does not exist.
        overwrite
            If False (default), raise an error when the schema already
            contains store tables. Set to True to drop the existing
            store and recreate it.

        Returns
        -------
        PostgreSQLStore
            A newly created store instance.

        Raises
        ------
        ValueError
            If ``overwrite`` is False and the schema already contains
            a store.
        """
        con = _resolve_connection(con)

        if name is None:
            name = "raghilda_db"

        if title is None:
            title = "Raghilda PostgreSQL Store"

        attributes_spec = normalize_attributes_spec(
            attributes=attributes,
            reserved_columns=_RESERVED_SYSTEM_COLUMNS,
        )
        attributes_schema = {
            key: spec.attribute_type for key, spec in attributes_spec.items()
        }

        if embed is None:
            embedding_column_sql = ""
        else:
            embedding_size = len(embed.embed(["foo"])[0])
            embedding_column_sql = f", embedding vector({embedding_size})"

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
        if embedding_column_sql:
            tail_columns.append(embedding_column_sql.lstrip(", "))
        tail_columns_sql = ""
        if tail_columns:
            tail_columns_sql = (
                ",\n                    " + ",\n                    ".join(tail_columns)
            )

        with con.cursor() as cur:
            if embed is not None:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                except psycopg2.errors.UndefinedFile:
                    con.rollback()
                    raise RuntimeError(
                        "pgvector extension is not available in this PostgreSQL installation. "
                        "Install pgvector: https://github.com/pgvector/pgvector"
                    )

            schema_id = psycopg2.extensions.quote_ident(schema, con)

            # Check if the store already exists
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = 'metadata'",
                [schema],
            )
            if cur.fetchone() is not None:
                if not overwrite:
                    raise ValueError(
                        f"A store already exists in schema {schema!r}. "
                        "Use overwrite=True to replace it."
                    )
                cur.execute("DROP SCHEMA %s CASCADE;" % schema_id)

            cur.execute("CREATE SCHEMA IF NOT EXISTS %s;" % schema_id)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema_id}.metadata (
                    name VARCHAR,
                    title VARCHAR,
                    embed_config VARCHAR,
                    attributes_schema_json VARCHAR
                );
            """)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema_id}.documents (
                    origin VARCHAR PRIMARY KEY,
                    text VARCHAR
                );
            """)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema_id}.embeddings (
                    origin VARCHAR NOT NULL REFERENCES {schema_id}.documents (origin),
                    chunk_id SERIAL,
                    start_index INTEGER,
                    end_index INTEGER,
                    char_count INTEGER,
                    context VARCHAR,
                    fts_search_vector tsvector,
                    PRIMARY KEY (origin, start_index, end_index)
                    {tail_columns_sql}
                );
            """)

            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_embeddings_fts
                ON {schema_id}.embeddings USING GIN (fts_search_vector);
            """)

            if embed is not None and vss_index is not None:
                vss_index = VSSMethod(vss_index)
                ops_class = vss_index.pg_ops_class
                index_name = f"idx_embeddings_vss_{vss_index.value}"
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {schema_id}.embeddings USING hnsw (embedding {ops_class});
                """)

            cur.execute(
                f"""
                INSERT INTO {schema_id}.metadata (name, title, embed_config, attributes_schema_json)
                VALUES (%s, %s, %s, %s)
                """,
                [name, title, embed_config_json, attributes_schema_json],
            )

        con.commit()

        metadata = {
            "name": name,
            "title": title,
            "embed": embed,
            "attributes": attributes_spec,
        }

        return PostgreSQLStore(con, metadata, schema)

    @staticmethod
    def connect(
        con: ConnectionLike,
        schema: str = "raghilda",
    ) -> "PostgreSQLStore":
        """Connect to an existing PostgreSQL store.

        Parameters
        ----------
        con
            A PostgreSQL connection string (e.g.
            ``"postgresql://user:pass@localhost/mydb"``).
        schema
            PostgreSQL schema where the store tables live. Defaults to
            ``"raghilda"``.

        Returns
        -------
        PostgreSQLStore
            A connected store instance.
        """
        con = _resolve_connection(con)
        schema_id = psycopg2.extensions.quote_ident(schema, con)
        with con.cursor() as cur:
            try:
                cur.execute(
                    f"SELECT name, title, embed_config, attributes_schema_json FROM {schema_id}.metadata"
                )
                row = cur.fetchone()
            except psycopg2.errors.UndefinedTable:
                con.rollback()
                raise ValueError("No metadata found in the database")

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
            attributes_spec = {}
        else:
            attributes_spec = attributes_spec_from_json_dict(
                json.loads(attributes_schema_json),
            )

        metadata = {
            "name": name,
            "title": title,
            "embed": embed,
            "attributes": attributes_spec,
        }

        return PostgreSQLStore(con, metadata, schema)

    def upsert(
        self,
        document: Document,
        *,
        skip_if_unchanged: bool = True,
    ) -> WriteResult[ChunkedMarkdownDocument]:
        """Upsert a document into the store.

        The document must be a
        :py:class:`~raghilda.document.ChunkedMarkdownDocument`.
        Use :py:class:`~raghilda.chunker.MarkdownChunker` to chunk a
        :py:class:`~raghilda.document.MarkdownDocument` before upserting.

        Parameters
        ----------
        document
            The chunked document to upsert.
        skip_if_unchanged
            If True (default), skip the write when the existing document
            for the same origin already has identical content. This avoids
            re-computing embeddings.
        """
        if not isinstance(document, ChunkedMarkdownDocument):
            raise NotImplementedError(
                f"Upsert not implemented for type {type(document)}"
            )
        if not isinstance(document.origin, str) or not document.origin:
            raise ValueError("document.origin must be a non-empty string for upsert().")
        if len(document.chunks) == 0:
            raise ValueError("Document must contain at least one chunk.")
        for chunk in document.chunks:
            expected = document.content[chunk.start_index : chunk.end_index]
            if chunk.text != expected:
                raise ValueError(
                    "Chunk text must match document.content[start_index:end_index]. "
                    f"Got chunk.text={chunk.text!r}, expected {expected!r} "
                    f"for start_index={chunk.start_index}, end_index={chunk.end_index}."
                )

        # Check if we can skip early
        with self.con.cursor() as cur:
            cur.execute(
                f"SELECT text FROM {self._schema}.documents WHERE origin = %s",
                [document.origin],
            )
            existing = cur.fetchone()

        if (
            skip_if_unchanged
            and existing is not None
            and existing[0] == document.content
        ):
            current_document = self._load_document_snapshot(
                origin=document.origin,
                text=existing[0],
            )
            return WriteResult(action="skipped", document=current_document)

        # Compute embeddings outside the transaction
        embed = self._metadata.get("embed")
        embeddings = None
        if embed is not None:
            chunk_texts = [chunk.text for chunk in document.chunks]
            embeddings = embed.embed(chunk_texts, EmbedInputType.DOCUMENT)

        # Resolve attributes for each chunk
        attributes_spec = self._metadata.get("attributes", {})
        resolved_chunk_attributes: list[dict[str, AttributeValue]] = []
        for chunk in document.chunks:
            chunk_attributes = getattr(chunk, "attributes", None)
            resolved_chunk_attributes.append(
                merge_attribute_values(
                    attributes_spec=attributes_spec,
                    sources=[document.attributes, chunk_attributes],
                )
            )

        # Write to the database
        with self.con.cursor() as cur:
            action = "inserted"
            replaced_document: ChunkedMarkdownDocument | None = None
            if existing is not None:
                action = "replaced"
                replaced_document = self._load_document_snapshot(
                    origin=document.origin,
                    text=existing[0],
                )
                cur.execute(
                    f"DELETE FROM {self._schema}.embeddings WHERE origin = %s",
                    [document.origin],
                )
                cur.execute(
                    f"UPDATE {self._schema}.documents SET text = %s WHERE origin = %s",
                    [document.content, document.origin],
                )
            else:
                cur.execute(
                    f"INSERT INTO {self._schema}.documents (origin, text) VALUES (%s, %s)",
                    [document.origin, document.content],
                )

            for i, chunk in enumerate(document.chunks):
                columns = [
                    "origin",
                    "start_index",
                    "end_index",
                    "char_count",
                    "context",
                    "fts_search_vector",
                ]
                values: list = [
                    document.origin,
                    chunk.start_index,
                    chunk.end_index,
                    len(chunk.text),
                    chunk.context,
                    chunk.text,
                ]
                if embeddings is not None:
                    columns.append("embedding")
                    values.append(str(embeddings[i]))

                for attr_name in attributes_spec:
                    columns.append(f'"{attr_name}"')
                    attr_val = resolved_chunk_attributes[i][attr_name]
                    if isinstance(attr_val, (dict, list)):
                        attr_val = json.dumps(attr_val)
                    values.append(attr_val)

                placeholders = []
                for col in columns:
                    if col == "fts_search_vector":
                        placeholders.append("to_tsvector('simple', %s)")
                    else:
                        placeholders.append("%s")
                placeholders_sql = ", ".join(placeholders)
                columns_sql = ", ".join(columns)
                cur.execute(
                    f"INSERT INTO {self._schema}.embeddings ({columns_sql}) VALUES ({placeholders_sql})",
                    values,
                )

        self.con.commit()

        return WriteResult(
            action=action,
            document=document,
            replaced_document=replaced_document,
        )

    def _load_document_snapshot(
        self, *, origin: str, text: str
    ) -> ChunkedMarkdownDocument:
        attributes_spec = self._metadata.get("attributes", {})
        attribute_columns = list(attributes_spec)
        attribute_select = ""
        if attribute_columns:
            cols = ", ".join(f'"{col}"' for col in attribute_columns)
            attribute_select = ", " + cols

        with self.con.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    start_index,
                    end_index,
                    char_count,
                    context
                    {attribute_select}
                FROM {self._schema}.embeddings
                WHERE origin = %s
                ORDER BY start_index, end_index
                """,
                [origin],
            )
            rows = cur.fetchall()
            if cur.description is None:
                raise RuntimeError("Failed to load document snapshot.")
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
            chunk_text = text[start_index:end_index]
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
            chunks=chunks,
            attributes=document_attributes or None,
        )

    def _filterable_columns(self) -> set[str]:
        attributes_spec = self._metadata.get("attributes", {})
        attributes_schema = {
            key: spec.attribute_type for key, spec in attributes_spec.items()
        }
        return _FILTERABLE_BASE_COLUMNS | filterable_attribute_paths(attributes_schema)

    def retrieve(
        self,
        text: str,
        top_k: int = 3,
        *,
        deoverlap: bool = True,
        attributes_filter: Optional[AttributeFilter] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most similar chunks to the given text.

        Combines results from vector similarity search (if embeddings are
        available) and full-text search, then deduplicates by chunk id,
        merging metrics from both methods.

        Parameters
        ----------
        text
            The query text to search for.
        top_k
            The maximum number of chunks to return from each retrieval
            method (VSS and FTS). Because results from both methods are
            combined before deoverlapping, the final count may differ
            from ``top_k``.
        deoverlap
            If True (default), merge overlapping chunks from the same
            document. Overlapping chunks are identified by their
            ``start_index`` and ``end_index`` positions.
        attributes_filter
            Optional filter to scope retrieval using attribute columns.
            Can be a SQL-like string or a dict AST.
            Example string: ``"tenant = 'docs' AND priority >= 2"``.

        Returns
        -------
        list[RetrievedChunk]
            The retrieved chunks with their relevance metrics.
        """
        retrieved_chunks: list[RetrievedChunk] = []
        if self._metadata.get("embed") is not None:
            retrieved_chunks = self.retrieve_vss(
                text, top_k, attributes_filter=attributes_filter
            )

        retrieved_chunks.extend(
            self.retrieve_fts(text, top_k, attributes_filter=attributes_filter)
        )

        # Deduplicate by (origin, chunk_id), merging metrics
        combined: dict[tuple[str | None, int | None], RetrievedChunk] = {}
        for chunk in retrieved_chunks:
            first_chunk_id = chunk.chunk_ids[0] if chunk.chunk_ids else None
            key = (chunk.origin, first_chunk_id)
            if key not in combined:
                combined[key] = chunk
            else:
                combined[key].metrics.extend(chunk.metrics or [])

        chunks = list(combined.values())

        if deoverlap:
            chunks = deoverlap_chunks(chunks, key=lambda c: c.origin)

        return chunks

    def retrieve_fts(
        self,
        text: str,
        top_k: int = 3,
        *,
        attributes_filter: Optional[AttributeFilter] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using PostgreSQL full-text search.

        Uses ``to_tsvector`` / ``plainto_tsquery`` with ``ts_rank`` for
        ranking results.

        Parameters
        ----------
        text
            The query text to search for.
        top_k
            The maximum number of chunks to return.
        attributes_filter
            Optional filter to scope retrieval using attribute columns.
            Can be a SQL-like string or a dict AST.

        Returns
        -------
        list[RetrievedChunk]
            The matching chunks ranked by ts_rank score.
        """
        compiled_filter = compile_filter_to_sql(
            attributes_filter,
            allowed_columns=self._filterable_columns(),
            column_expr=pg_column_expression,
        )
        where_clause = "WHERE e.fts_search_vector @@ plainto_tsquery('simple', %s)"
        if compiled_filter:
            where_clause += f" AND ({compiled_filter})"

        sql = f"""
            SELECT
                d.origin,
                e.start_index,
                e.end_index,
                e.char_count,
                e.context,
                e.chunk_id,
                SUBSTRING(d.text FROM e.start_index + 1 FOR e.end_index - e.start_index) AS chunk_text,
                ts_rank(e.fts_search_vector, plainto_tsquery('simple', %s)) AS rank
            FROM {self._schema}.embeddings e
            JOIN {self._schema}.documents d USING (origin)
            {where_clause}
            ORDER BY rank DESC
            LIMIT %s
        """
        with self.con.cursor() as cur:
            cur.execute(sql, [text, text, top_k])
            rows = cur.fetchall()

        chunks: list[RetrievedChunk] = []
        for row in rows:
            (
                origin,
                start_index,
                end_index,
                char_count,
                context,
                chunk_id,
                chunk_text,
                rank,
            ) = row
            chunks.append(
                RetrievedChunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    char_count=char_count,
                    context=context,
                    origin=origin,
                    metrics=[Metric(name="ts_rank", value=float(rank))],
                    chunk_ids=[chunk_id],
                )
            )

        return chunks

    def retrieve_vss(
        self,
        query: str | Sequence[float],
        top_k: int = 3,
        *,
        method: Optional[VSSMethod] = None,
        attributes_filter: Optional[AttributeFilter] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using pgvector similarity search.

        Uses pgvector distance operators for nearest-neighbor search.
        For best performance, ensure an HNSW index exists for the chosen
        distance method (created automatically for cosine distance, or
        via :meth:`build_index` for others).

        Parameters
        ----------
        query
            The query text or embedding vector. If a string is provided,
            it will be embedded using the store's embedding provider.
        top_k
            The maximum number of chunks to return.
        method
            The distance method to use. Defaults to cosine distance.
            Can be a :class:`VSSMethod` enum or a string like
            ``"cosine_distance"``, ``"l2_distance"``, or
            ``"inner_product"``.
        attributes_filter
            Optional filter to scope retrieval using attribute columns.
            Can be a SQL-like string or a dict AST.

        Returns
        -------
        list[RetrievedChunk]
            The most similar chunks with distance metrics.

        Raises
        ------
        ValueError
            If ``query`` is a string but no embedding provider is
            configured.
        """
        if method is None:
            method = VSSMethod.COSINE_DISTANCE
        else:
            method = VSSMethod(method)

        embed = self._metadata.get("embed")
        if isinstance(query, str):
            if embed is None:
                raise ValueError("No embedding function available in the store")
            query = embed.embed([query], EmbedInputType.QUERY)[0]

        operator = method.pg_operator
        metric_name = method.metric_name

        query_literal = "[" + ",".join(str(x) for x in query) + "]"

        compiled_filter = compile_filter_to_sql(
            attributes_filter,
            allowed_columns=self._filterable_columns(),
            column_expr=pg_column_expression,
        )
        where_clause = ""
        if compiled_filter:
            where_clause = f"WHERE {compiled_filter}"

        sql = f"""
            SELECT
                d.origin,
                e.start_index,
                e.end_index,
                e.char_count,
                e.context,
                e.chunk_id,
                SUBSTRING(d.text FROM e.start_index + 1 FOR e.end_index - e.start_index) AS chunk_text,
                (e.embedding {operator} %s::vector) AS distance
            FROM {self._schema}.embeddings e
            JOIN {self._schema}.documents d USING (origin)
            {where_clause}
            ORDER BY distance ASC
            LIMIT %s
        """
        with self.con.cursor() as cur:
            cur.execute(sql, [query_literal, top_k])
            rows = cur.fetchall()

        chunks: list[RetrievedChunk] = []
        for row in rows:
            (
                origin,
                start_index,
                end_index,
                char_count,
                context,
                chunk_id,
                chunk_text,
                distance,
            ) = row
            chunks.append(
                RetrievedChunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    char_count=char_count,
                    context=context,
                    origin=origin,
                    metrics=[Metric(name=metric_name, value=float(distance))],
                    chunk_ids=[chunk_id],
                )
            )

        return chunks

    def build_index(self, method: VSSMethod) -> None:
        """Build an HNSW index on the embedding column for the given distance method.

        A cosine distance index is created by default when calling
        :meth:`create` with an embedding provider. Use this method to
        add indexes for other distance methods.

        Parameters
        ----------
        method
            The distance method to index for.
        """
        method = VSSMethod(method)
        ops_class = method.pg_ops_class
        index_name = f"idx_embeddings_vss_{method.value}"
        with self.con.cursor() as cur:
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {self._schema}.embeddings USING hnsw (embedding {ops_class});
            """)
        self.con.commit()

    def size(self) -> int:
        """Count the number of documents in the store.

        Returns
        -------
        int
            The number of documents (not chunks) in the store.
        """
        with self.con.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._schema}.documents")
            row = cur.fetchone()
        return row[0] if row else 0
