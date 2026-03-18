"""PostgreSQL vector store backend using pgvector and SQLAlchemy."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import sqlalchemy as sa

from .._attributes import (
    AttributeFilter,
    AttributeFloatVectorType,
    AttributeSpec,
    AttributeStructType,
    AttributeType,
    AttributeValue,
    AttributesSchemaSpec,
    attributes_spec_from_json_dict,
    attributes_spec_to_json_dict,
    coerce_attribute_value_for_output,
    compile_filter_to_sqlalchemy,
    normalize_attributes_spec,
)
from pgvector.sqlalchemy import Vector

from ._base import SQLStore, _sa_column_type, build_tables
from ._constructs import FTSRank, TextSlice, VectorDistance
from .._store_helpers import (
    IndexType,
    RESERVED_SYSTEM_COLUMNS,
    RetrievedStoreMarkdownChunk,
    VSSMethod,
    coerce_index_type as _coerce_index_type,
)
from .._store_metadata import (
    EmbeddedAttributesStoreMetadata,
    attributes_schema_from_spec,
)
from ..chunk import Metric
from ..embedding import EmbedInputType, EmbeddingProvider, embedding_from_config

logger = logging.getLogger(__name__)


def _sa_connection_string(connection_string: str) -> str:
    """Convert a postgresql:// URI to postgresql+psycopg:// for SQLAlchemy.

    SQLAlchemy defaults to psycopg2 for ``postgresql://``. Since we use
    psycopg (v3), we need the ``+psycopg`` driver suffix.
    """
    if connection_string.startswith("postgresql://"):
        return "postgresql+psycopg://" + connection_string[len("postgresql://") :]
    if connection_string.startswith("postgres://"):
        return "postgresql+psycopg://" + connection_string[len("postgres://") :]
    return connection_string


def _pg_column_type(attribute_type: AttributeType) -> sa.types.TypeEngine[Any]:
    """Return the SQLAlchemy column type for an attribute type (PostgreSQL)."""
    if isinstance(attribute_type, AttributeStructType):
        from sqlalchemy.dialects.postgresql import JSONB

        return JSONB()
    if isinstance(attribute_type, AttributeFloatVectorType):
        return Vector(attribute_type.dimension)  # type: ignore[arg-type]
    return _sa_column_type(attribute_type)


@dataclass
class PostgreSQLStoreMetadata(EmbeddedAttributesStoreMetadata):
    name: str
    title: str
    embed: Optional[EmbeddingProvider]
    attributes: dict[str, AttributeSpec]

    @property
    def attributes_spec(self) -> dict[str, AttributeSpec]:
        return self.attributes

    @property
    def attributes_schema(self) -> dict[str, AttributeType]:
        try:
            return self._attributes_schema_cache
        except AttributeError:
            self._attributes_schema_cache = attributes_schema_from_spec(self.attributes)
            return self._attributes_schema_cache


def _build_metadata_table(sa_metadata: sa.MetaData) -> sa.Table:
    """Build the metadata table definition."""
    return sa.Table(
        "metadata",
        sa_metadata,
        sa.Column("name", sa.Text),
        sa.Column("title", sa.Text),
        sa.Column("embed_config", sa.Text),
        sa.Column("embed_dimension", sa.Integer),
        sa.Column("attributes_schema_json", sa.Text),
        extend_existing=True,
    )


class PostgreSQLStore(SQLStore):
    """A vector store backed by PostgreSQL with pgvector.

    PostgreSQLStore provides vector storage with support for both
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
    ) -> "PostgreSQLStore":
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
        PostgreSQLStore
            A newly created store instance.
        """
        import psycopg
        import psycopg.conninfo
        from psycopg import sql

        conninfo = psycopg.conninfo.conninfo_to_dict(connection_string)
        dbname = str(conninfo.get("dbname", "raghilda"))

        # Connect to the default 'postgres' database to create/drop the target
        admin_conninfo: dict[str, str] = {k: str(v) for k, v in conninfo.items()}
        admin_conninfo["dbname"] = "postgres"
        admin_conn_str = psycopg.conninfo.make_conninfo(**admin_conninfo)

        with psycopg.connect(admin_conn_str, autocommit=True) as admin_con:
            if overwrite:
                admin_con.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(dbname))
                )
            row = admin_con.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", [dbname]
            ).fetchone()
            if row is not None and not overwrite:
                raise ValueError(f"Database already exists: {dbname}")
            if row is None:
                admin_con.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname))
                )

        # Create SA engine
        engine = sa.create_engine(_sa_connection_string(connection_string))

        # Enable pgvector extension
        with engine.begin() as conn:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

        if name is None:
            name = "raghilda_db"
        if title is None:
            title = "Raghilda PostgreSQL Store"

        attributes_spec = normalize_attributes_spec(
            attributes=attributes,
            reserved_columns=RESERVED_SYSTEM_COLUMNS,
        )

        embed_dimension: int | None = None
        if embed is not None:
            embed_dimension = len(embed.embed(["foo"])[0])

        embed_config_json = None
        if embed is not None:
            embed_config_json = json.dumps(embed.get_config())

        attributes_schema_json = json.dumps(
            attributes_spec_to_json_dict(attributes_spec)
        )

        # Build SA table objects
        sa_metadata = sa.MetaData()
        documents, embeddings = build_tables(
            sa_metadata,
            attributes_spec,
            embed_dimension,
            column_type_resolver=_pg_column_type,
            embedding_column_type=(
                Vector(embed_dimension) if embed_dimension is not None else None
            ),
        )

        # Create all tables via SA DDL
        sa_metadata.create_all(engine)

        # Create the chunks VIEW
        with engine.begin() as conn:
            conn.execute(
                sa.text("""
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
                )
                """)
            )

        # Insert metadata row
        meta_table = _build_metadata_table(sa_metadata)
        meta_table.create(engine, checkfirst=True)

        with engine.begin() as conn:
            conn.execute(
                meta_table.insert().values(
                    name=name,
                    title=title,
                    embed_config=embed_config_json,
                    embed_dimension=embed_dimension,
                    attributes_schema_json=attributes_schema_json,
                )
            )

        metadata = PostgreSQLStoreMetadata(
            name=name,
            title=title,
            embed=embed,
            attributes=attributes_spec,
        )

        return PostgreSQLStore(engine, metadata, sa_metadata, documents, embeddings)

    @staticmethod
    def connect(connection_string: str) -> "PostgreSQLStore":
        """Connect to an existing PostgreSQL store.

        Parameters
        ----------
        connection_string
            PostgreSQL connection string.

        Returns
        -------
        PostgreSQLStore
            A connected store instance.
        """
        engine = sa.create_engine(_sa_connection_string(connection_string))

        # Validate this is a raghilda database
        inspector = sa.inspect(engine)
        if not inspector.has_table("metadata"):
            raise ValueError("Not a valid Raghilda database connection")

        # Read metadata
        sa_metadata = sa.MetaData()
        meta_table = _build_metadata_table(sa_metadata)

        with engine.connect() as conn:
            row = conn.execute(
                sa.select(
                    meta_table.c.name,
                    meta_table.c.title,
                    meta_table.c.embed_config,
                    meta_table.c.embed_dimension,
                    meta_table.c.attributes_schema_json,
                )
            ).fetchone()

        if row is None:
            raise ValueError("No metadata found in the database")

        (
            store_name,
            store_title,
            embed_config_json,
            embed_dimension,
            attributes_schema_json,
        ) = row

        embed: EmbeddingProvider | None = None
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

        documents, embeddings = build_tables(
            sa_metadata,
            attributes_spec,
            embed_dimension,
            column_type_resolver=_pg_column_type,
            embedding_column_type=(
                Vector(embed_dimension) if embed_dimension is not None else None
            ),
        )

        metadata = PostgreSQLStoreMetadata(
            name=store_name,
            title=store_title,
            embed=embed,
            attributes=attributes_spec,
        )

        return PostgreSQLStore(engine, metadata, sa_metadata, documents, embeddings)

    # -- retrieval (PostgreSQL-specific) ----------------------------------------

    def _retrieve(
        self,
        text: str,
        top_k: int,
        *,
        attributes_filter: Optional[AttributeFilter] = None,
    ) -> Sequence[RetrievedStoreMarkdownChunk]:
        """Return raw retrieved chunks from VSS and FTS."""
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

        return retrieved_chunks

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
        metric_value = VectorDistance(e.c.embedding, query_vector, str(method))

        def _build_select_cols(src: Any) -> list[Any]:
            cols: list[Any] = [
                src.c.chunk_id,
                d.c.origin.label("origin"),
                src.c.start_index,
                src.c.end_index,
                src.c.char_count,
                src.c.context,
            ]
            for col in attribute_columns:
                cols.append(src.c[col])
            cols.extend(
                [
                    TextSlice(
                        d.c.text,
                        src.c.start_index + 1,
                        src.c.end_index - src.c.start_index,
                    ).label("text"),
                    sa.literal(str(method)).label("metric_name"),
                    src.c.metric_value
                    if hasattr(src.c, "metric_value")
                    else metric_value.label("metric_value"),
                ]
            )
            return cols

        if sa_filter is None:
            # Optimization: pre-filter in a subquery when no attribute filter.
            # Select only the columns we need — excludes the large embedding
            # vector and search_vector tsvector columns.
            inner_cols: list[Any] = [
                e.c.chunk_id,
                e.c.origin,
                e.c.start_index,
                e.c.end_index,
                e.c.char_count,
                e.c.context,
            ]
            for col in attribute_columns:
                inner_cols.append(e.c[col])
            inner_cols.append(metric_value.label("metric_value"))
            inner = (
                sa.select(*inner_cols)
                .order_by(sa.literal_column("metric_value").asc())
                .limit(top_k)
                .subquery("e")
            )
            stmt = (
                sa.select(*_build_select_cols(inner))
                .select_from(inner.join(d, inner.c.origin == d.c.origin))
                .order_by(inner.c.metric_value.asc())
                .limit(top_k)
            )
        else:
            stmt = (
                sa.select(*_build_select_cols(e))
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
        ranked_cols.extend(
            [
                text_slice.label("text"),
                sa.literal("fts").label("metric_name"),
                fts_rank.label("metric_value"),
            ]
        )

        ranked_stmt = sa.select(*ranked_cols).select_from(
            e.join(d, e.c.origin == d.c.origin)
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
                _HNSW_OPS = {
                    VSSMethod.COSINE_DISTANCE: (
                        "store_hnsw_cosine_index",
                        "vector_cosine_ops",
                    ),
                    VSSMethod.EUCLIDEAN_DISTANCE: (
                        "store_hnsw_l2_index",
                        "vector_l2_ops",
                    ),
                    VSSMethod.NEGATIVE_INNER_PRODUCT: (
                        "store_hnsw_ip_index",
                        "vector_ip_ops",
                    ),
                }
                for idx_name, ops in _HNSW_OPS.values():
                    conn.execute(sa.text(f"DROP INDEX IF EXISTS {idx_name}"))
                for idx_name, ops in _HNSW_OPS.values():
                    conn.execute(
                        sa.text(
                            f"CREATE INDEX {idx_name} "
                            f"ON embeddings USING hnsw (embedding {ops})"
                        )
                    )


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
