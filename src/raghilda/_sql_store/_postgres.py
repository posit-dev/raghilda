"""PostgreSQL vector store backend using pgvector and SQLAlchemy."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import sqlalchemy as sa

from .._attributes import (
    AttributeSpec,
    AttributeType,
    AttributesSchemaSpec,
    attributes_spec_from_json_dict,
    attributes_spec_to_json_dict,
    normalize_attributes_spec,
)
from ._base import SQLStore, build_tables
from .._store_helpers import RESERVED_SYSTEM_COLUMNS
from .._store_metadata import (
    EmbeddedAttributesStoreMetadata,
    attributes_schema_from_spec,
)
from ..embedding import EmbeddingProvider, embedding_from_config

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


class PostgresStore(SQLStore):
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
    ) -> "PostgresStore":
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
            sa_metadata, attributes_spec, embed_dimension
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
        # We use the metadata table created by SA
        meta_table = sa.Table(
            "metadata",
            sa_metadata,
            sa.Column("name", sa.Text),
            sa.Column("title", sa.Text),
            sa.Column("embed_config", sa.Text),
            sa.Column("attributes_schema_json", sa.Text),
            extend_existing=True,
        )
        meta_table.create(engine, checkfirst=True)

        with engine.begin() as conn:
            conn.execute(
                meta_table.insert().values(
                    name=name,
                    title=title,
                    embed_config=embed_config_json,
                    attributes_schema_json=attributes_schema_json,
                )
            )

        metadata = PostgresStoreMetadata(
            name=name,
            title=title,
            embed=embed,
            attributes=attributes_spec,
        )

        return PostgresStore(engine, metadata, sa_metadata, documents, embeddings)

    @staticmethod
    def connect(connection_string: str) -> "PostgresStore":
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
        engine = sa.create_engine(_sa_connection_string(connection_string))

        # Validate this is a raghilda database
        inspector = sa.inspect(engine)
        if not inspector.has_table("metadata"):
            raise ValueError("Not a valid Raghilda database connection")

        # Read metadata
        sa_metadata = sa.MetaData()
        meta_table = sa.Table(
            "metadata",
            sa_metadata,
            sa.Column("name", sa.Text),
            sa.Column("title", sa.Text),
            sa.Column("embed_config", sa.Text),
            sa.Column("attributes_schema_json", sa.Text),
        )

        with engine.connect() as conn:
            row = conn.execute(
                sa.select(
                    meta_table.c.name,
                    meta_table.c.title,
                    meta_table.c.embed_config,
                    meta_table.c.attributes_schema_json,
                )
            ).fetchone()

        if row is None:
            raise ValueError("No metadata found in the database")

        store_name, store_title, embed_config_json, attributes_schema_json = row

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

        embed_dimension: int | None = None
        if embed is not None:
            embed_dimension = len(embed.embed(["foo"])[0])

        documents, embeddings = build_tables(
            sa_metadata, attributes_spec, embed_dimension
        )

        metadata = PostgresStoreMetadata(
            name=store_name,
            title=store_title,
            embed=embed,
            attributes=attributes_spec,
        )

        return PostgresStore(engine, metadata, sa_metadata, documents, embeddings)
