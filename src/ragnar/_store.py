from abc import ABC, abstractmethod
import os
from ._embedding import EmbeddingProvider
from .document import ChunkedDocument, Document, MarkdownDocument, MarkdownChunk, Chunk
from typing import Optional, Union, Sequence
import duckdb
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import pandas as pd
from enum import StrEnum

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

    @abstractmethod
    def retrieve(self, text: str, top_k: int, *args, **kwargs) -> Sequence[Chunk]:
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Counts the number of documents in the store.
        Note: It's documents, not chunks!
        """
        pass


@dataclass
class DuckDBStoreMetadata:
    name: str
    title: str
    embed: Optional[EmbeddingProvider]


class VSSMethod(StrEnum):
    COSINE_DISTANCE = "cosine_distance"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    NEGATIVE_INNER_PRODUCT = "negative_inner_product"


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

    def insert(
        self,
        document: Union[Document, ChunkedDocument[MarkdownDocument, MarkdownChunk]],
    ) -> None:
        if isinstance(document, ChunkedDocument):
            self._insert_chunked_document(document)
        else:
            raise NotImplementedError(
                f"Insert not implemented for type {type(document)}"
            )

    def _insert_chunked_document(
        self, chunked_doc: ChunkedDocument[MarkdownDocument, MarkdownChunk]
    ) -> None:
        doc = pd.DataFrame([asdict(chunked_doc.document)])
        chunks = pd.DataFrame([asdict(x) for x in chunked_doc.chunks])

        if self.metadata.embed is not None:
            chunks["embedding"] = self.metadata.embed.embed(chunks.content.tolist())

        try:
            self.con.begin()
            result = self.con.execute("SELECT nextval('doc_id_seq')").fetchone()
            if result is None:
                raise RuntimeError("Failed to get next document ID")
            (doc_id,) = result
            doc["doc_id"] = doc_id
            doc.rename(columns={"content": "text"}, inplace=True)  # content -> text
            chunks["doc_id"] = [doc_id] * len(chunks)

            _duckdb_append(self.con, "documents", doc)
            _duckdb_append(self.con, "embeddings", chunks)
            self.con.commit()
        except Exception as e:
            self.con.rollback()
            raise e

    def retrieve(
        self, text: str, top_k: int = 3, *, deoverlap: bool = True
    ) -> Sequence[MarkdownChunk]:
        raise NotImplementedError("Retrieve method not implemented yet")

    def retrieve_vss(
        self,
        query: str | Sequence[float],
        top_k: int,
        *,
        method: VSSMethod = VSSMethod.COSINE_DISTANCE,
    ) -> list[MarkdownChunk]:
        if isinstance(query, str):
            if self.metadata.embed is None:
                raise ValueError("No embedding function available in the store")
            query = self.metadata.embed.embed([query])[0]

        func, order = _vss_method_info(method)
        sql = f"""
        SELECT
            e.doc_id, 
            e.chunk_id, 
            e.start, 
            e.end, 
            e.context, 
            doc.text[ e.start: e.end ] AS content 
        FROM (
            SELECT
                *,
                '{method}' AS metric_name,
                {func}(embedding, [{",".join(str(x) for x in query)}]::FLOAT[{len(query)}]) AS metric_value
            FROM embeddings
            ORDER BY metric_value {order}
            LIMIT {top_k}
        ) AS e
        JOIN documents doc USING (doc_id)
        ORDER BY metric_value
        """

        cursor = self.con.cursor()

        cursor.execute(sql)
        results = cursor.fetchall()

        if cursor.description is None:
            raise RuntimeError("Failed get cursor description.")

        columns = [desc[0] for desc in cursor.description]

        return [MarkdownChunk(**dict(zip(columns, chunk))) for chunk in results]

    def size(self) -> int:
        result = self.con.execute(
            "SELECT COUNT(DISTINCT doc_id) FROM documents"
        ).fetchone()
        if result is None:
            raise RuntimeError("Failed to get size of the store")
        return result[0]


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
        try:
            con.unregister(f"tmp_data_{table}")
        except Exception:
            pass


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
