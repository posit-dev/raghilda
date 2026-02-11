from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
from collections.abc import Sized
from typing import Any, Iterable, Optional, Sequence, Callable, TYPE_CHECKING, Union
from concurrent.futures import ThreadPoolExecutor

from ._store import BaseStore
from .chunk import MarkdownChunk, RetrievedChunk, Metric
from .chunker import MarkdownChunker
from .document import Document, MarkdownDocument
from .read import read_as_markdown
from ._deoverlap import deoverlap_chunks
from ._embedding import (
    ChromaConvertible,
    EmbeddingProvider,
    EmbedInputType,
    embedding_from_config,
)
from tqdm import tqdm

if TYPE_CHECKING:
    import numpy as np
    from chromadb.api.types import EmbeddingFunction

    ChromaEmbedding = Union[EmbeddingProvider, EmbeddingFunction]


_METADATA_TITLE_KEY = "raghilda_title"
_ADAPTER_NAME = "raghilda_embedding_adapter"


# ChromaEmbeddingAdapter is only defined when chromadb is installed
try:
    from chromadb import EmbeddingFunction as _EmbeddingFunctionBase
    from chromadb.utils.embedding_functions import register_embedding_function

    class ChromaEmbeddingAdapter(_EmbeddingFunctionBase):
        """Adapter to use any raghilda EmbeddingProvider with ChromaDB.

        This adapter wraps a raghilda `EmbeddingProvider` to make it compatible with
        ChromaDB's `EmbeddingFunction` protocol, including serialization support.
        Use this for custom embedding providers that don't have a native ChromaDB equivalent.

        The adapter is automatically used when passing an `EmbeddingProvider` to
        `ChromaDBStore.create()` or `connect()` if the provider doesn't implement
        `ChromaConvertible`.

        Note: This adapter stores the provider config for serialization, but cross-language
        compatibility (e.g., TypeScript) is not supported since the provider is Python-only.

        Parameters
        ----------
        provider
            A raghilda EmbeddingProvider instance.

        Examples
        --------
        ```{python}
        #| eval: false
        from raghilda.embedding import EmbeddingOpenAI
        from raghilda.store import ChromaDBStore, ChromaEmbeddingAdapter

        # Manual wrapping (usually not needed - automatic conversion is preferred)
        provider = EmbeddingOpenAI(model="text-embedding-3-small")
        adapter = ChromaEmbeddingAdapter(provider)

        store = ChromaDBStore.create(
            location="my_store",
            name="docs",
            embed=adapter,
        )
        ```
        """

        def __init__(self, provider: EmbeddingProvider) -> None:
            self._provider = provider

        def __call__(self, input: Sequence[str]) -> list[np.ndarray]:
            """Generate embeddings for documents.

            This method is called by ChromaDB when adding/upserting documents.
            """
            import numpy as np

            embeddings = self._provider.embed(list(input), EmbedInputType.DOCUMENT)
            return [np.array(emb, dtype=np.float32) for emb in embeddings]

        def embed_query(self, input: Sequence[str]) -> list[np.ndarray]:
            """Generate embeddings for queries.

            This method is called by ChromaDB when querying the collection.
            """
            import numpy as np

            embeddings = self._provider.embed(list(input), EmbedInputType.QUERY)
            return [np.array(emb, dtype=np.float32) for emb in embeddings]

        @staticmethod
        def name() -> str:
            """Return the name of this embedding function for ChromaDB registry."""
            return _ADAPTER_NAME

        def get_config(self) -> dict[str, Any]:
            """Return configuration for serialization.

            The config includes the wrapped provider's config so it can be restored.
            """
            return {
                "provider_config": self._provider.get_config(),
            }

        @staticmethod
        def build_from_config(config: dict[str, Any]) -> "ChromaEmbeddingAdapter":
            """Restore the adapter from a configuration dict.

            This reconstructs both the adapter and the wrapped provider.
            """
            provider_config = config.get("provider_config", {})
            provider = embedding_from_config(provider_config)
            return ChromaEmbeddingAdapter(provider)

    # Register on module load
    register_embedding_function(ChromaEmbeddingAdapter)

except ImportError:
    # ChromaDB not installed - ChromaEmbeddingAdapter will not be available
    pass


def _to_chroma_embedding_function(
    embed: Optional[ChromaEmbedding],
) -> Optional[EmbeddingFunction]:
    """Convert an embedding provider to a ChromaDB embedding function if needed.

    Parameters
    ----------
    embed
        Either a raghilda EmbeddingProvider (implementing ChromaConvertible)
        or a ChromaDB embedding function.

    Returns
    -------
    EmbeddingFunction | None
        A ChromaDB-compatible embedding function.
    """
    if embed is None:
        return None
    if isinstance(embed, ChromaConvertible):
        return embed.to_chroma()
    if isinstance(embed, EmbeddingProvider):
        # Fallback: wrap in adapter for providers without ChromaDB equivalent
        return ChromaEmbeddingAdapter(embed)
    return embed  # type: ignore[return-value]


def _import_chromadb():
    try:
        chromadb = importlib.import_module("chromadb")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ChromaDB is required to use ChromaDBStore. Install with `pip install chromadb`."
        ) from exc
    return chromadb


def _get_client(location: str | Path | None):
    chromadb = _import_chromadb()
    if location is None or str(location) == ":memory:":
        return chromadb.Client()
    return chromadb.PersistentClient(path=str(location))


@dataclass
class ChromaDBMarkdownChunk(MarkdownChunk):
    """MarkdownChunk with ChromaDB-specific fields for storage."""

    doc_id: Optional[str] = None
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
        if token_count is None:
            token_count = len(text)

        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            token_count=token_count,
            context=context,
        )

        self.doc_id = doc_id
        self.chunk_id = chunk_id


@dataclass
class RetrievedChromaDBMarkdownChunk(ChromaDBMarkdownChunk, RetrievedChunk):
    """ChromaDBMarkdownChunk with retrieval metrics."""

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
        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            context=context,
            token_count=token_count,
            doc_id=doc_id,
            chunk_id=chunk_id,
        )

        if metrics is None:
            metrics = []
        self.metrics = metrics


@dataclass
class ChromaDBStoreMetadata:
    name: str
    title: str


class ChromaDBStore(BaseStore):
    """A vector store backed by ChromaDB.

    ChromaDBStore provides local vector storage using Chroma's embedded client.
    Documents are chunked by raghilda and embeddings are generated by Chroma's
    embedding function (defaults to Chroma's built-in embedding).

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.store import ChromaDBStore

    store = ChromaDBStore.create(location="raghilda_chroma", name="docs")

    store.insert(markdown_doc)
    chunks = store.retrieve("hello world", top_k=3)
    ```
    """

    @staticmethod
    def create(
        location: str | Path | None = None,
        *,
        overwrite: bool = False,
        name: Optional[str] = None,
        title: Optional[str] = None,
        embed: Optional[ChromaEmbedding] = None,
        collection_metadata: Optional[dict[str, Any]] = None,
        client: Any = None,
    ):
        """Create a new ChromaDB store.

        Parameters
        ----------
        location
            Path where ChromaDB will persist its data. Use ":memory:" or None
            for an in-memory store.
        overwrite
            Whether to overwrite an existing collection with the same name.
        name
            Collection name for the store.
        title
            Human-readable title for the store.
        embed
            Optional embedding function. Can be either a raghilda EmbeddingProvider
            (e.g., EmbeddingOpenAI, EmbeddingCohere) or a ChromaDB embedding function.
            Raghilda providers are automatically converted to their ChromaDB equivalents.
            If None, Chroma's default embedding function is used.
        collection_metadata
            Additional metadata to attach to the Chroma collection.
        client
            Optional pre-configured Chroma client (e.g., HttpClient).

        Returns
        -------
        ChromaDBStore
            A newly created store instance.
        """
        embedding_function = _to_chroma_embedding_function(embed)

        if name is None:
            name = "raghilda_chroma"
        if title is None:
            title = "Raghilda ChromaDB Store"

        if client is None:
            client = _get_client(location)

        if overwrite:
            try:
                client.delete_collection(name=name)
            except Exception:
                pass

        store_metadata = {_METADATA_TITLE_KEY: title}

        merged_metadata = dict(collection_metadata or {})
        merged_metadata.update(store_metadata)

        collection = client.create_collection(
            name=name,
            metadata=merged_metadata,
            embedding_function=embedding_function,
        )

        return ChromaDBStore(
            client=client,
            collection=collection,
            metadata=ChromaDBStoreMetadata(name=name, title=title),
        )

    @staticmethod
    def connect(
        name: str,
        location: str | Path | None = None,
        *,
        embed: Optional[ChromaEmbedding] = None,
        client: Any = None,
    ):
        """Connect to an existing ChromaDB store.

        Parameters
        ----------
        name
            Collection name for the store.
        location
            Path where ChromaDB persists its data. Use ":memory:" or None
            for an in-memory store.
        embed
            Optional embedding function. Can be either a raghilda EmbeddingProvider
            (e.g., EmbeddingOpenAI, EmbeddingCohere) or a ChromaDB embedding function.
            Raghilda providers are automatically converted to their ChromaDB equivalents.
            If None, ChromaDB will attempt to restore the embedding function from
            the collection's stored configuration.
        client
            Optional pre-configured Chroma client (e.g., HttpClient).

        Returns
        -------
        ChromaDBStore
            A connected store instance.
        """
        embedding_function = _to_chroma_embedding_function(embed)

        if client is None:
            client = _get_client(location)

        collection = client.get_collection(
            name=name, embedding_function=embedding_function
        )
        metadata = collection.metadata or {}
        title = metadata.get(_METADATA_TITLE_KEY, "Raghilda ChromaDB Store")

        return ChromaDBStore(
            client=client,
            collection=collection,
            metadata=ChromaDBStoreMetadata(name=name, title=title),
        )

    def __init__(self, client: Any, collection: Any, metadata: ChromaDBStoreMetadata):
        self.client = client
        self.collection = collection
        self.metadata = metadata

    def insert(self, document: Document) -> None:
        if not isinstance(document, MarkdownDocument):
            raise ValueError("Only MarkdownDocument is supported for ChromaDBStore")
        if document.chunks is None:
            raise ValueError("Document must be chunked before insertion")

        texts = [chunk.text for chunk in document.chunks]

        ids = []
        metadatas = []
        for idx, chunk in enumerate(document.chunks):
            ids.append(f"{document.id}:{idx}")
            metadata = {
                "doc_id": document.id,
                "chunk_id": idx,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
                "token_count": chunk.token_count,
                "context": chunk.context,
                "origin": document.origin,
            }
            metadatas.append({k: v for k, v in metadata.items() if v is not None})

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    def ingest(
        self,
        items: Iterable[Any],
        prepare: Optional[Callable[[Any], Document]] = None,
        num_workers: Optional[int] = None,
        progress: bool = True,
    ) -> None:
        """
        Ingest multiple documents in parallel.

        This method processes items through a prepare function to create Documents,
        then inserts them into the store. Items are processed in parallel using
        a thread pool for improved performance.

        Parameters
        ----------
        items
            An iterable of items to ingest. Can be any iterable including lists,
            generators, or other iterables. Each item will be passed to the prepare
            function to create a Document. By default, items are expected to be
            URIs (file paths or URLs) that will be read with read_as_markdown
            and chunked automatically.
        prepare
            A callable that takes an item and returns a Document with chunks computed.
            Use this to customize how items are converted to documents. The function
            should handle chunking if needed. If None, items are treated as URIs
            and processed with read_as_markdown followed by MarkdownChunker.
        num_workers
            The number of worker threads to use for parallel ingestion.
            If None, defaults to the number of CPU cores.
        progress
            Whether to display a progress bar during ingestion. Default is True.
            The progress bar shows the total count only if items has a known length
            (e.g., a list). For generators, it shows progress without a total.

        Examples
        --------
        Ingest files from a list of paths:

        ```{python}
        #| eval: false
        store.ingest(["doc1.md", "doc2.pdf", "doc3.html"])
        ```

        Ingest from a generator:

        ```{python}
        #| eval: false
        def get_urls():
            for url in scrape_sitemap("https://example.com/sitemap.xml"):
                yield url

        store.ingest(get_urls())
        ```

        Ingest with a custom prepare function:

        ```{python}
        #| eval: false
        from raghilda.chunker import MarkdownChunker

        chunker = MarkdownChunker()

        def prepare_record(record: dict) -> MarkdownDocument:
            doc = MarkdownDocument(
                origin=record["id"],
                content=record["text"]
            )
            return chunker.chunk_document(doc)

        records = [{"id": "1", "text": "Hello"}, {"id": "2", "text": "World"}]
        store.ingest(records, prepare=prepare_record)
        ```
        """
        if num_workers is None:
            num_workers = os.cpu_count() or 1

        if prepare is None:
            chunker = MarkdownChunker()

            def default_prepare(uri: str) -> Document:
                return chunker.chunk_document(read_as_markdown(uri))

            prepare = default_prepare

        total = len(items) if isinstance(items, Sized) else None

        def do_ingest_work(item: Any) -> None:
            try:
                doc = prepare(item)
                self.insert(doc)
            except Exception as e:
                raise RuntimeError(f"Failed to ingest '{item}': {e}") from e

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            list(
                tqdm(
                    pool.map(do_ingest_work, items),
                    total=total,
                    disable=not progress,
                )
            )

    def retrieve(
        self, text: str, top_k: int, *, deoverlap: bool = True, **kwargs
    ) -> Sequence[RetrievedChromaDBMarkdownChunk]:
        """Retrieve the most similar chunks to the given text.

        Uses ChromaDB's vector similarity search to find relevant chunks,
        then optionally merges overlapping chunks from the same document.

        Parameters
        ----------
        text
            The query text to search for.
        top_k
            The maximum number of chunks to return.
        deoverlap
            If True (default), merge overlapping chunks from the same document.
            Overlapping chunks are identified by their `start_index` and `end_index`
            positions. When merged, the resulting chunk spans the union of the
            original ranges and combines their metrics.
        **kwargs
            Additional arguments passed to ChromaDB's `query()` method,
            such as `where` for metadata filtering.

        Returns
        -------
        Sequence[RetrievedChromaDBMarkdownChunk]
            The retrieved chunks with their relevance metrics.
        """
        results = self.collection.query(
            query_texts=[text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            **kwargs,
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        output: list[RetrievedChromaDBMarkdownChunk] = []
        for doc_text, metadata, distance in zip(
            documents, metadatas, distances, strict=False
        ):
            metadata = metadata or {}
            start_index = int(metadata.get("start_index", 0))
            end_index = int(metadata.get("end_index", start_index + len(doc_text)))
            token_count = int(metadata.get("token_count", len(doc_text)))
            metrics = []
            if distance is not None:
                metrics.append(Metric(name="distance", value=distance))
            chunk = RetrievedChromaDBMarkdownChunk(
                text=doc_text,
                start_index=start_index,
                end_index=end_index,
                context=metadata.get("context"),
                token_count=token_count,
                doc_id=metadata.get("doc_id"),
                chunk_id=metadata.get("chunk_id"),
                metrics=metrics,
            )
            output.append(chunk)

        if deoverlap:
            output = deoverlap_chunks(output, key=lambda c: c.doc_id)

        return output

    def size(self) -> int:
        results = self.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []
        doc_ids = {
            metadata.get("doc_id")
            for metadata in metadatas
            if metadata and metadata.get("doc_id")
        }
        return len(doc_ids)
