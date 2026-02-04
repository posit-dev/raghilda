from abc import ABC, abstractmethod
import os
from .embedding import EmbeddingProvider
from .chunk import MarkdownChunk, RetrievedChunk, Metric
from .chunker import MarkdownChunker
from .read import read_as_markdown
from .document import Document, MarkdownDocument
from typing import Optional, Sequence, Callable
import duckdb
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import pandas as pd
from enum import StrEnum
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


logger = logging.getLogger(__name__)


@dataclass
class DuckDBMarkdownChunk(MarkdownChunk):
    """MarkdownChunk with DuckDB-specific fields for database storage"""

    doc_id: Optional[int] = None
    chunk_id: Optional[int] = None

    def __init__(
        self,
        text: str,
        start_index: int,
        end_index: int,
        context=None,
        token_count=None,
        doc_id=None,
        chunk_id=None,
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
        )

        # Set DuckDB-specific fields
        self.doc_id = doc_id
        self.chunk_id = chunk_id


@dataclass
class RetrievedDuckDBMarkdownChunk(DuckDBMarkdownChunk, RetrievedChunk):
    """DuckDBMarkdownChunk with retrieval metrics"""

    def __init__(
        self,
        text: str,
        start_index: int,
        end_index: int,
        context=None,
        token_count=None,
        doc_id=None,
        chunk_id=None,
        metrics=None,
    ):
        # Initialize DuckDBMarkdownChunk
        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            context=context,
            token_count=token_count,
            doc_id=doc_id,
            chunk_id=chunk_id,
        )

        # Initialize metrics
        if metrics is None:
            metrics = []
        self.metrics = metrics


class BaseStore(ABC):
    """Abstract base class for vector stores.

    A store is responsible for storing documents and their embeddings,
    and retrieving relevant chunks based on similarity search.

    Subclasses must implement all abstract methods to provide a concrete
    storage backend (e.g., DuckDB, OpenAI Vector Store).
    """

    @staticmethod
    @abstractmethod
    def connect(*args, **kwargs) -> "BaseStore":
        """Connect to an existing store.

        Returns
        -------
        BaseStore
            A connected store instance.
        """
        pass

    @staticmethod
    @abstractmethod
    def create(*args, **kwargs) -> "BaseStore":
        """Create a new store.

        Returns
        -------
        BaseStore
            A newly created store instance.
        """
        pass

    @abstractmethod
    def insert(self, document: Document) -> None:
        """Insert a document into the store.

        The document will be chunked and embedded before storage.

        Parameters
        ----------
        document
            The document to insert.
        """
        pass

    @abstractmethod
    def retrieve(
        self, text: str, top_k: int, *args, **kwargs
    ) -> Sequence[RetrievedChunk]:
        """Retrieve the most similar chunks to the given text.

        Parameters
        ----------
        text
            The query text to search for.
        top_k
            The maximum number of chunks to return.

        Returns
        -------
        Sequence[RetrievedChunk]
            The most similar chunks, ordered by relevance.
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Count the number of documents in the store.

        Returns
        -------
        int
            The number of documents (not chunks) in the store.
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

        if embed is None:
            embedding_sql = ""
        else:
            embedding_size = len(embed.embed(["foo"])[0])
            embedding_sql = f"embedding FLOAT[{embedding_size}]"

        con.execute(f"""
        CREATE SEQUENCE chunk_id_seq START 1; -- need a unique id for fts

        CREATE OR REPLACE TABLE documents (
            doc_id VARCHAR PRIMARY KEY DEFAULT uuid(),
            origin VARCHAR UNIQUE,
            text VARCHAR
        );

        CREATE OR REPLACE TABLE embeddings (
            doc_id VARCHAR NOT NULL,
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
        document: Document,
    ) -> None:
        if isinstance(document, MarkdownDocument):
            self._insert_chunked_document(document)
        else:
            raise NotImplementedError(
                f"Insert not implemented for type {type(document)}"
            )

    def ingest(
        self,
        uris: Sequence[str],
        prepare: Optional[Callable[[str], Document]] = None,
        num_workers: Optional[int] = None,
        progress=True,
    ) -> None:
        """
        Ingest multiple documents from a list of URIs.

        Parameters
        ----------
        uris
            A sequence of URIs (file paths, URLs, etc.) to ingest.
        prepare
            A callable that takes a URI and returns a MarkdownDocument with chunks computed..
        num_workers
            The number of worker threads to use for parallel ingestion. If None, to the number of CPU cores.
        progress
            Whether to display a progress bar during ingestion. Default is True.
        """
        # This functions uses a thread pool to insert documents in the databse
        if num_workers is None:
            num_workers = os.cpu_count() or 1

        if prepare is None:
            chunker = MarkdownChunker()

            def _prepare(uri: str) -> Document:
                return chunker.chunk_document(read_as_markdown(uri))

            prepare = _prepare

        def do_ingest_work(uri: str) -> None:
            chunked_doc = prepare(uri)
            self.insert(chunked_doc)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            list(
                tqdm(
                    pool.map(do_ingest_work, uris),
                    total=len(uris),
                    disable=not progress,
                )
            )

    def _insert_chunked_document(self, chunked_doc: MarkdownDocument) -> None:
        # Document should be chunked for insertion
        assert chunked_doc.chunks is not None

        doc = pd.DataFrame([asdict(chunked_doc)])
        # Shpuld we really drop metadata? Perhaps we want to add it to the schema definition
        doc.drop(columns=["chunks", "metadata"], inplace=True, errors="ignore")
        chunks = pd.DataFrame([asdict(x) for x in chunked_doc.chunks])

        if self.metadata.embed is not None:
            chunks["embedding"] = self.metadata.embed.embed(chunks.text.tolist())
        else:
            chunks.drop(columns=["embedding"], inplace=True, errors="ignore")

        # Map Chunk field names to database field names
        chunks.rename(
            columns={"start_index": "start", "end_index": "end"}, inplace=True
        )
        # Remove token_count since it's not stored in the database
        if "token_count" in chunks.columns:
            chunks.drop(columns=["token_count"], inplace=True)
        # Remove text since it's not stored in embeddings table (it's computed from documents table)
        if "text" in chunks.columns:
            chunks.drop(columns=["text"], inplace=True)

        # local cursor is used so we can use multiple threads
        # see https://duckdb.org/docs/stable/guides/python/multiple_threads.html
        cursor = self.con.cursor()
        try:
            cursor.begin()
            doc.rename(
                columns={"content": "text", "id": "doc_id"}, inplace=True
            )  # content -> text
            chunks["doc_id"] = [doc["doc_id"][0]] * len(chunks)
            chunks.drop(
                columns=["id"], inplace=True, errors="ignore"
            )  # id -> chunk_id (auto). the id here can be discarded

            _duckdb_append(cursor, "documents", doc)
            _duckdb_append(cursor, "embeddings", chunks)
            cursor.commit()
        except Exception as e:
            try:
                cursor.rollback()
            except Exception:
                pass
            finally:
                raise e

    def retrieve(
        self, text: str, top_k: int = 3, *, deoverlap: bool = True
    ) -> Sequence[RetrievedDuckDBMarkdownChunk]:
        retrieved_chunks = []
        if self.metadata.embed is not None:
            retrieved_chunks = self.retrieve_vss(text, top_k)

        retrieved_chunks.extend(self.retrieve_bm25(text, top_k))

        # combine chunks by `doc_id` and `chunk_id` and then merge metrics
        combined_chunks: dict[
            tuple[int | None, int | None], RetrievedDuckDBMarkdownChunk
        ] = {}
        for chunk in retrieved_chunks:
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
    ) -> list[RetrievedDuckDBMarkdownChunk]:
        if isinstance(query, str):
            if self.metadata.embed is None:
                raise ValueError("No embedding function available in the store")
            query = self.metadata.embed.embed([query])[0]

        func, order = _vss_method_info(method)
        sql = f"""
        SELECT
            e.doc_id, 
            e.chunk_id, 
            e.start AS start_index, 
            e.end AS end_index, 
            e.context, 
            doc.text[ e.start: e.end ] AS text,
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

        output: list[RetrievedDuckDBMarkdownChunk] = []
        for chunk in results:
            chunk_dict = dict(zip(columns, chunk))
            name, value = chunk_dict.pop("metric_name"), chunk_dict.pop("metric_value")
            chunk_dict["metrics"] = [Metric(name, value)]
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
    ) -> list[RetrievedDuckDBMarkdownChunk]:
        sql = """
        SELECT
            e.doc_id, 
            e.chunk_id, 
            e.start AS start_index, 
            e.end AS end_index, 
            e.context, 
            doc.text[ e.start: e.end ] AS text,
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

        output: list[RetrievedDuckDBMarkdownChunk] = []
        for chunk in results:
            chunk_dict = dict(zip(columns, chunk))
            name, value = chunk_dict.pop("metric_name"), chunk_dict.pop("metric_value")
            chunk_dict["metrics"] = [Metric(name, value)]
            output.append(RetrievedDuckDBMarkdownChunk(**chunk_dict))

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


def _check_is_raghilda_con(con: duckdb.DuckDBPyConnection):
    tables = con.execute("SHOW TABLES").fetchall()
    tables = [t[0] for t in tables]

    if "metadata" not in tables:
        raise ValueError("Not a valid Raghilda database connection")


def _duckdb_append(con: duckdb.DuckDBPyConnection, table: str, data):
    try:
        con.register(f"tmp_data_{table}", data)
        column_list = ", ".join(_quote_identifier(col) for col in data.columns)
        con.execute(
            f"INSERT INTO {table} ({column_list}) SELECT * FROM tmp_data_{table}"
        )
    finally:
        try:
            con.unregister(f"tmp_data_{table}")
        except Exception:
            pass


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
