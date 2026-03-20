from ._store import BaseStore
import json
from .embedding import EmbeddingProvider, embedding_from_config
from typing import Mapping, Optional
import psycopg2
import logging
from ._attributes import (
    AttributeFloatVectorType,
    AttributeStructType,
    AttributeType,
    AttributesSchemaSpec,
    attributes_spec_from_json_dict,
    attributes_spec_to_json_dict,
    normalize_attributes_spec,
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


class PostgreSQLStore(BaseStore):
    """
    A store backed by a PostgreSQL database.
    """

    def __init__(self, con: psycopg2.extensions.connection, metadata: dict):
        self.con = con
        self._metadata = metadata

    @staticmethod
    def create(
        con: psycopg2.extensions.connection,
        embed: Optional[EmbeddingProvider],
        name: Optional[str] = None,
        title: Optional[str] = None,
        attributes: Optional[AttributesSchemaSpec] = None,
    ) -> "PostgreSQLStore":
        """Create a new PostgreSQL store.

        Parameters
        ----------
        con
            An open psycopg connection to a PostgreSQL database.
        embed
            Embedding provider for generating vector embeddings.
            If None, only full-text search will be available.
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
            tail_columns_sql = ",\n                    " + ",\n                    ".join(tail_columns)

        with con.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    name VARCHAR,
                    title VARCHAR,
                    embed_config VARCHAR,
                    attributes_schema_json VARCHAR
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    origin VARCHAR PRIMARY KEY,
                    text VARCHAR
                );
            """)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    origin VARCHAR NOT NULL REFERENCES documents (origin),
                    chunk_id SERIAL,
                    start_index INTEGER,
                    end_index INTEGER,
                    char_count INTEGER,
                    context VARCHAR,
                    PRIMARY KEY (origin, start_index, end_index)
                    {tail_columns_sql}
                );
            """)

            cur.execute(
                """
                INSERT INTO metadata (name, title, embed_config, attributes_schema_json)
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

        return PostgreSQLStore(con, metadata)

    @staticmethod
    def connect(con: psycopg2.extensions.connection) -> "PostgreSQLStore":
        """Connect to an existing PostgreSQL store.

        Parameters
        ----------
        con
            An open psycopg2 connection to a PostgreSQL database
            that already contains a raghilda store.

        Returns
        -------
        PostgreSQLStore
            A connected store instance.
        """
        with con.cursor() as cur:
            try:
                cur.execute(
                    "SELECT name, title, embed_config, attributes_schema_json FROM metadata"
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

        return PostgreSQLStore(con, metadata)

    def upsert(self, document, *, skip_if_unchanged=True):  # noqa: ARG002
        raise NotImplementedError("upsert is not yet implemented")

    def retrieve(self, text, top_k, *args, **kwargs):  # noqa: ARG002
        raise NotImplementedError("retrieve is not yet implemented")

    def size(self):
        raise NotImplementedError("size is not yet implemented")

