"""PostgreSQL vector store backend using pgvector."""

from ._store import BaseStore, WriteResult
import json
import threading
from .embedding import EmbeddingProvider, EmbedInputType, embedding_from_config
from .chunk import Chunk, MarkdownChunk, RetrievedChunk, Metric
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
from psycopg.rows import tuple_row

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
                admin_con.execute(
                    f"CREATE DATABASE {_quote_identifier(dbname)}"
                )

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
            tail_columns_sql = (
                ",\n            " + ",\n            ".join(tail_columns)
            )

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
        raise NotImplementedError("upsert not yet implemented for PostgresStore")

    def retrieve(
        self,
        text: str,
        top_k: int = 3,
        *,
        deoverlap: bool = True,
        attributes_filter: Optional[AttributeFilter] = None,
    ) -> Sequence[RetrievedStoreMarkdownChunk]:
        raise NotImplementedError("retrieve not yet implemented for PostgresStore")

    def size(self) -> int:
        raise NotImplementedError("size not yet implemented for PostgresStore")


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
