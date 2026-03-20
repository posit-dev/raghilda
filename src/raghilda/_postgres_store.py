from ._store import BaseStore
import json
from .embedding import EmbeddingProvider, embedding_from_config
from typing import Optional
import psycopg2
import logging

logger = logging.getLogger(__name__)


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

        Returns
        -------
        PostgreSQLStore
            A newly created store instance.
        """
        if name is None:
            name = "raghilda_db"

        if title is None:
            title = "Raghilda PostgreSQL Store"

        if embed is None:
            embedding_column_sql = ""
        else:
            embedding_size = len(embed.embed(["foo"])[0])
            embedding_column_sql = f", embedding vector({embedding_size})"

        embed_config_json = None
        if embed is not None:
            embed_config_json = json.dumps(embed.get_config())

        with con.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    name VARCHAR,
                    title VARCHAR,
                    embed_config VARCHAR
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
                    {embedding_column_sql}
                );
            """)

            cur.execute(
                """
                INSERT INTO metadata (name, title, embed_config)
                VALUES (%s, %s, %s)
                """,
                [name, title, embed_config_json],
            )

        con.commit()

        metadata = {
            "name": name,
            "title": title,
            "embed": embed,
        }

        return PostgreSQLStore(con, metadata)

    @staticmethod
    def connect(con: psycopg2.extensions.connection) -> "PostgreSQLStore":
        raise NotImplementedError("connect is not yet implemented")

    def upsert(self, document, *, skip_if_unchanged=True):
        raise NotImplementedError("upsert is not yet implemented")

    def retrieve(self, text, top_k, *args, **kwargs):
        raise NotImplementedError("retrieve is not yet implemented")

    def size(self):
        raise NotImplementedError("size is not yet implemented")

