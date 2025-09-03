from abc import ABC, abstractmethod
import os
from ._embedding import EmbeddingProvider
from .document import ChunkedDocument, Document, MarkdownDocument, LazyMarkdownChunk
from typing import Optional, overload, Union
import duckdb
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class Store(ABC):
    @staticmethod
    @abstractmethod
    def connect(*args, **kwargs) -> "Store":
        pass

    @staticmethod
    @abstractmethod
    def create(*args, **kwargs) -> "Store":
        pass

    @abstractmethod
    def insert(self, document: Union[Document, ChunkedDocument]) -> None:
        pass


@dataclass
class DuckDBStoreMetadata:
    name: str
    title: str
    embed: Optional[EmbeddingProvider]


class DuckDBStore(Store):
    @staticmethod
    def connect(
        location: str | Path = ":memory:",
        read_only: bool = False,
    ):
        con = duckdb.connect(database=location, read_only=read_only)
        _check_is_ragnar_con(con)

        metadata = con.execute("SELECT name, title from metadata").fetchall()
        metadata = DuckDBStoreMetadata(
            name=metadata[0][0],
            title=metadata[0][1],
            # TODO: figure out the api for reloading embedding functions
            embed=None,
        )

        return DuckDBStore(con, metadata)

    @staticmethod
    def create(
        location: str | Path,
        embed: Optional[EmbeddingProvider],
        overwrite: bool = False,
        name: Optional[str] = None,
        title: Optional[str] = None,
    ):
        _overwrite_or_error(location, overwrite)
        con = duckdb.connect(database=location)

        if name is None:
            name = "ragnar_db"

        if title is None:
            title = "Ragnar DuckDB Store"

        if embed is None:
            embedding_sql = ""
        else:
            embedding_size = len(embed.embed(["foo"])[0])
            embedding_sql = f"embedding FLOAT[{embedding_size}]"

        con.execute(f"""
        CREATE SEQUENCE chunk_id_seq START 1; -- need a unique id for fts
        CREATE SEQUENCE doc_id_seq START 1;

        CREATE OR REPLACE TABLE documents (
            doc_id INTEGER PRIMARY KEY DEFAULT nextval('doc_id_seq'),
            origin VARCHAR UNIQUE,
            text VARCHAR
        );

        CREATE OR REPLACE TABLE embeddings (
            doc_id INTEGER NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id),
            chunk_id INTEGER DEFAULT nextval('chunk_id_seq'),
            start INTEGER,
            "end" INTEGER,
            PRIMARY KEY (doc_id, start, "end"),
            context VARCHAR,
            {embedding_sql}
        );

        CREATE OR REPLACE VIEW chunks AS (
            SELECT
            d.origin as origin,
            e.*,
            d.text[ e.start : e."end" ] as text
            FROM
            documents d
            JOIN
            embeddings e
            USING
            (doc_id)
        );
        """)

        return DuckDBStore(
            con,
            DuckDBStoreMetadata(
                name=name,
                title=title,
                embed=embed,
            ),
        )

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        metadata: DuckDBStoreMetadata,
    ):
        self.con = con
        self.metadata = metadata

    @overload
    def insert(
        self, document: ChunkedDocument[MarkdownDocument, LazyMarkdownChunk]
    ) -> None: ...

    def insert(
        self,
        document: Union[
            MarkdownDocument, ChunkedDocument[MarkdownDocument, LazyMarkdownChunk]
        ],
    ) -> None:
        if isinstance(document, ChunkedDocument):
            self._insert_chunked_document(document)
        else:
            raise NotImplementedError(
                f"Insert not implemented for type {type(document)}"
            )

    def _insert_chunked_document(
        self, chunked_doc: ChunkedDocument[MarkdownDocument, LazyMarkdownChunk]
    ) -> None:
        doc = pd.DataFrame(asdict(chunked_doc.document), index=[0])
        chunks = pd.DataFrame([asdict(x) for x in chunked_doc.chunks])

        if self.metadata.embed is not None:
            chunks.embedding = self.metadata.embed.embed(chunks.content.tolist())

        try:
            self.con.begin()
            (doc_id,) = self.con.execute("SELECT nextval('doc_id_seq')").fetchone()
            doc["doc_id"] = doc_id
            doc.rename(columns={"content": "text"}, inplace=True)  # content -> text
            chunks["doc_id"] = [doc_id] * len(chunks)

            _duckdb_append(self.con, "documents", doc)
            _duckdb_append(self.con, "embeddings", chunks)
            self.con.commit()
        except Exception as e:
            self.con.rollback()
            raise e


def _overwrite_or_error(location: str | Path, overwrite: bool) -> None:
    if not overwrite:
        return

    if location == ":memory:":
        return

    location = Path(location)
    if os.path.exists(location) or os.path.exists(location / ".wal"):
        if overwrite:
            logger.info(f"Overwriting existing database at: {location}")
            os.remove(location)
            os.remove(location / ".wal")
        else:
            raise FileExistsError(f"File already exists: {location}")


def _check_is_ragnar_con(con: duckdb.DuckDBPyConnection):
    tables = con.execute("SHOW TABLES").fetchall()
    tables = [t[0] for t in tables]

    if "metadata" not in tables:
        raise ValueError("Not a valid Ragnar database connection")


def _duckdb_append(con: duckdb.DuckDBPyConnection, table: str, data):
    try:
        # reorder columns in the same order as in the database
        cols = con.execute(f"DESCRIBE {table}").fetchdf().column_name.tolist()
        data = data[cols]

        con.register(f"tmp_data_{table}", data)
        con.execute(f"INSERT INTO {table} SELECT * FROM tmp_data_{table}")
    finally:
        con.unregister("tmp_data")
