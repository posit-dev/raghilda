from abc import ABC, abstractmethod
import os
from ._embedding import EmbeddingProvider
from .document import (
    ChunkedDocument,
    Document,
    MarkdownDocument,
    MarkdownChunk,
    RetrievedChunk,
    RetrievedMarkdownChunk,
    Metric,
)
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
    def retrieve(
        self, text: str, top_k: int, *args, **kwargs
    ) -> Sequence[RetrievedChunk]:
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


class IndexType(StrEnum):
    BM25 = "bm25"
    HNSW = "hnsw"


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
    ) -> Sequence[RetrievedMarkdownChunk]:
        vss_chunks = self.retrieve_vss(text, top_k)
        bm25_chunks = self.retrieve_bm25(text, top_k)

        # combine chunks by `doc_id` and `chunk_id` and then merge metrics
        combined_chunks: dict[
            tuple[int | None, int | None], RetrievedMarkdownChunk
        ] = {}
        for chunk in vss_chunks + bm25_chunks:
            key = (chunk.doc_id, chunk.chunk_id)
            if key not in combined_chunks:
                combined_chunks[key] = chunk
            else:
                combined_chunks[key].metrics.extend(chunk.metrics or [])

        if deoverlap:
            raise NotImplementedError("Deoverlap not implemented yet")

        return list(combined_chunks.values())

    def retrieve_vss(
        self,
        query: str | Sequence[float],
        top_k: int,
        *,
        method: VSSMethod = VSSMethod.COSINE_DISTANCE,
    ) -> list[RetrievedMarkdownChunk]:
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
            doc.text[ e.start: e.end ] AS content,
            e.metric_name,
            e.metric_value 
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

        output: list[RetrievedMarkdownChunk] = []
        for chunk in results:
            chunk_dict = dict(zip(columns, chunk))
            name, value = chunk_dict.pop("metric_name"), chunk_dict.pop("metric_value")
            chunk_dict["metrics"] = [Metric(name, value)]
            output.append(RetrievedMarkdownChunk(**chunk_dict))

        return output

    def retrieve_bm25(
        self,
        query: str,
        top_k: int,
        *,
        k: float = 1.2,
        b: float = 0.75,
        conjunctive: bool = False,
    ) -> list[RetrievedMarkdownChunk]:
        sql = """
        SELECT
            e.doc_id, 
            e.chunk_id, 
            e.start, 
            e.end, 
            e.context, 
            doc.text[ e.start: e.end ] AS content,
            'bm25' AS metric_name,
            fts_main_chunks.match_bm25(chunk_id, $query, k := $k, b := $b, conjunctive := $conjunctive) AS metric_value
        FROM embeddings e
        JOIN documents doc USING (doc_id)
        ORDER BY metric_value DESC
        LIMIT $top_k
        """

        cursor = self.con.cursor()

        cursor.execute(
            sql,
            {
                "query": query,
                "top_k": top_k,
                "k": k,
                "b": b,
                "conjunctive": conjunctive,
            },
        )
        results = cursor.fetchall()

        if cursor.description is None:
            raise RuntimeError("Failed get cursor description.")

        columns = [desc[0] for desc in cursor.description]

        output: list[RetrievedMarkdownChunk] = []
        for chunk in results:
            chunk_dict = dict(zip(columns, chunk))
            name, value = chunk_dict.pop("metric_name"), chunk_dict.pop("metric_value")
            chunk_dict["metrics"] = [Metric(name, value)]
            output.append(RetrievedMarkdownChunk(**chunk_dict))

        return output

    def build_index(self, type: Optional[IndexType | list[IndexType]] = None):
        """
        Build the specified index types on the embeddings table.

        Parameters
        ----------
        type
            The type of index to build. Can be a single IndexType or a list of IndexTypes.
            If None, builds both BM25 and HNSW indexes.
        """
        if type is None:
            type = [IndexType.BM25, IndexType.HNSW]

        if isinstance(type, IndexType):
            type = [type]

        if IndexType.BM25 in type:
            self.con.execute("INSTALL FTS; LOAD FTS;")
            try:
                self.con.begin()
                self._create_fts_index()
                self.con.commit()
            except Exception as e:
                self.con.rollback()
                raise e

        if IndexType.HNSW in type:
            self.con.execute("INSTALL vss; LOAD vss;")
            try:
                self.con.begin()
                self._create_hnsw_index()
                self.con.commit()
            except Exception as e:
                self.con.rollback()
                raise e

    def _create_fts_index(self):
        self.con.execute("""
        ALTER VIEW chunks RENAME TO chunks_view;

        CREATE TABLE chunks AS
          SELECT chunk_id, context, text FROM chunks_view;
        """)

        self.con.execute("""
        PRAGMA create_fts_index(
          'chunks',            -- input_table
          'chunk_id',          -- input_id
          'context', 'text',   -- *input_values
          overwrite = 1
        );
        """)

        self.con.execute("""
        DROP TABLE chunks;
        ALTER VIEW chunks_view RENAME TO chunks;
        """)

    def _create_hnsw_index(self):
        self.con.execute("""
        SET hnsw_enable_experimental_persistence = true;

        DROP INDEX IF EXISTS store_hnsw_cosine_index;
        DROP INDEX IF EXISTS store_hnsw_l2sq_index;
        DROP INDEX IF EXISTS store_hnsw_ip_index;

        CREATE INDEX store_hnsw_cosine_index ON embeddings USING HNSW (embedding) WITH (metric = 'cosine');
        CREATE INDEX store_hnsw_l2sq_index   ON embeddings USING HNSW (embedding) WITH (metric = 'l2sq'); -- array_distance?
        CREATE INDEX store_hnsw_ip_index     ON embeddings USING HNSW (embedding) WITH (metric = 'ip');  -- array_dot_product
        """)

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
