from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import threading
from collections.abc import Sized
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Union,
)
from concurrent.futures import ThreadPoolExecutor

from ._store import BaseStore, WriteResult
from ._utils import lazy_map
from .chunk import Chunk, MarkdownChunk, RetrievedChunk, Metric
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
from ._attributes import (
    AttributeFilter,
    AttributesSchemaSpec,
    AttributeSpec,
    AttributeType,
    attributes_spec_from_json_dict,
    attributes_spec_to_json_dict,
    compile_filter_to_chroma_where,
    merge_attribute_values,
    normalize_attributes_spec,
)
from tqdm import tqdm
from ._store_metadata import AttributesStoreMetadata, attributes_schema_from_spec

if TYPE_CHECKING:
    import numpy as np
    from chromadb.api.types import EmbeddingFunction

    ChromaEmbedding = Union[EmbeddingProvider, EmbeddingFunction]


_METADATA_TITLE_KEY = "raghilda_title"
_ATTRIBUTES_SCHEMA_METADATA_KEY = "raghilda_attributes_schema_json"
_ADAPTER_NAME = "raghilda_embedding_adapter"
_CONTENT_HASH_METADATA_KEY = "_raghilda_content_hash"
_CONTENT_TEXT_METADATA_KEY = "_raghilda_content_text"

_RESERVED_SYSTEM_COLUMNS = {
    "chunk_id",
    "start_index",
    "end_index",
    "token_count",
    "context",
    "origin",
    _CONTENT_HASH_METADATA_KEY,
    _CONTENT_TEXT_METADATA_KEY,
}

_FILTERABLE_BASE_COLUMNS = {
    "chunk_id",
    "start_index",
    "end_index",
    "token_count",
    "context",
    "origin",
}


def _ensure_no_reserved_attributes(
    attributes_spec: dict[str, AttributeSpec],
    reserved_keys: set[str],
) -> None:
    for key in attributes_spec:
        if key in reserved_keys:
            raise ValueError(f"Attribute column '{key}' is reserved")


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


@dataclass(repr=False)
class ChromaDBMarkdownChunk(MarkdownChunk):
    """MarkdownChunk with ChromaDB-specific fields for storage."""

    def __init__(
        self,
        text: str,
        start_index: int,
        end_index: int,
        context=None,
        token_count=None,
        origin=None,
        attributes=None,
    ):
        if token_count is None:
            token_count = len(text)

        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            token_count=token_count,
            context=context,
            origin=origin,
            attributes=attributes,
        )


@dataclass(repr=False)
class RetrievedChromaDBMarkdownChunk(ChromaDBMarkdownChunk, RetrievedChunk):
    """ChromaDBMarkdownChunk with retrieval metrics."""

    def __init__(
        self,
        text: str,
        start_index: int,
        end_index: int,
        context=None,
        token_count=None,
        origin=None,
        metrics=None,
        chunk_ids=None,
        attributes=None,
    ):
        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            context=context,
            token_count=token_count,
            origin=origin,
            attributes=attributes,
        )

        if metrics is None:
            metrics = []
        self.metrics = metrics
        if chunk_ids is None:
            chunk_ids = []
        self.chunk_ids = chunk_ids


@dataclass
class ChromaDBStoreMetadata(AttributesStoreMetadata):
    name: str
    title: str
    attributes: dict[str, AttributeSpec]

    @property
    def attributes_spec(self) -> dict[str, AttributeSpec]:
        return self.attributes

    @property
    def attributes_schema(self) -> dict[str, AttributeType]:
        return attributes_schema_from_spec(self.attributes)


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

    store.upsert(markdown_doc)
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
        attributes: Optional[AttributesSchemaSpec] = None,
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
        attributes
            Optional schema for user-defined attribute columns.
            Attribute names use identifier-style syntax.
            Chroma also provides built-in filterable columns:
            `chunk_id`, `start_index`, `end_index`, `token_count`,
            `context`, and `origin`.
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

        attributes_spec = normalize_attributes_spec(
            attributes=attributes,
            reserved_columns=_RESERVED_SYSTEM_COLUMNS,
            allow_vector_types=False,
            allow_struct_types=False,
            allow_optional_values=False,
        )
        if client is None:
            client = _get_client(location)

        if overwrite:
            try:
                client.delete_collection(name=name)
            except Exception:
                pass

        store_metadata = {
            _METADATA_TITLE_KEY: title,
            _ATTRIBUTES_SCHEMA_METADATA_KEY: json.dumps(
                attributes_spec_to_json_dict(attributes_spec)
            ),
        }

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
            metadata=ChromaDBStoreMetadata(
                name=name,
                title=title,
                attributes=attributes_spec,
            ),
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
        attributes_spec: dict[str, AttributeSpec] = {}
        if metadata.get(_ATTRIBUTES_SCHEMA_METADATA_KEY) is not None:
            attributes_spec = attributes_spec_from_json_dict(
                json.loads(metadata[_ATTRIBUTES_SCHEMA_METADATA_KEY]),
                allow_vector_types=False,
                allow_struct_types=False,
                allow_optional_values=False,
            )
        _ensure_no_reserved_attributes(attributes_spec, _RESERVED_SYSTEM_COLUMNS)
        return ChromaDBStore(
            client=client,
            collection=collection,
            metadata=ChromaDBStoreMetadata(
                name=name,
                title=title,
                attributes=attributes_spec,
            ),
        )

    def __init__(self, client: Any, collection: Any, metadata: AttributesStoreMetadata):
        self.client = client
        self.collection = collection
        self.metadata = metadata
        self._origin_locks: dict[str, threading.Lock] = {}
        self._origin_lock_ref_counts: dict[str, int] = {}
        self._origin_locks_guard = threading.Lock()
        self._chunk_id_lock = threading.Lock()
        # Temporary workaround for ChromaDB's non-thread-safe telemetry batching.
        # See https://github.com/chroma-core/chroma/issues/6512
        self._collection_lock = threading.Lock()
        self._next_chunk_id: Optional[int] = None

    def upsert(
        self,
        document: Document,
        *,
        skip_if_unchanged: bool = True,
    ) -> WriteResult:
        if not isinstance(document, MarkdownDocument):
            raise ValueError("Only MarkdownDocument is supported for ChromaDBStore")
        if document.chunks is None:
            raise ValueError("Document must be chunked before insertion")
        if len(document.chunks) == 0:
            raise ValueError("Document must contain at least one chunk.")
        if not isinstance(document.origin, str) or not document.origin:
            raise ValueError("document.origin must be a non-empty string for upsert().")

        with self._origin_lock(document.origin), self._collection_lock:
            content_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()

            existing = self.collection.get(
                where={"origin": document.origin},
                include=["metadatas", "documents"],
            )
            existing_ids = list(existing.get("ids") or [])
            replaced_document = None
            incoming_signature = self._incoming_chunk_signature(document)
            if existing_ids and skip_if_unchanged:
                existing_metadatas = list(existing.get("metadatas") or [])
                existing_hash = None
                for metadata in existing_metadatas:
                    if metadata and metadata.get(_CONTENT_HASH_METADATA_KEY):
                        existing_hash = metadata[_CONTENT_HASH_METADATA_KEY]
                        break
                existing_signature = self._existing_chunk_signature(existing)
                if (
                    existing_hash == content_hash
                    and existing_signature == incoming_signature
                ):
                    current_document = (
                        self._snapshot_document_from_existing_if_available(
                            existing,
                            origin=document.origin,
                        )
                    )
                    if current_document is None:
                        current_document = MarkdownDocument(
                            origin=document.origin,
                            content=document.content,
                            chunks=document.chunks,
                            attributes=document.attributes,
                        )
                    return WriteResult(
                        action="skipped",
                        document=current_document,
                    )
            if existing_ids:
                replaced_document = self._snapshot_document_from_existing_if_available(
                    existing,
                    origin=document.origin,
                )

            texts = [chunk.text for chunk in document.chunks]

            ids = []
            chunk_attributes_records = []
            merged_document_attributes: dict[str, Any] = {}
            assigned_chunk_ids = self._allocate_chunk_ids(len(document.chunks))
            for idx, chunk in enumerate(document.chunks):
                resolved_attributes = merge_attribute_values(
                    attributes_spec=self.metadata.attributes_spec,
                    sources=[document.attributes, chunk.attributes],
                )
                ids.append(f"{document.origin}:{idx}")
                chunk_record = {
                    "chunk_id": assigned_chunk_ids[idx],
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "token_count": chunk.token_count,
                    "context": chunk.context,
                    "origin": document.origin,
                    _CONTENT_HASH_METADATA_KEY: content_hash,
                }
                if idx == 0:
                    chunk_record[_CONTENT_TEXT_METADATA_KEY] = document.content
                chunk_record.update(resolved_attributes)
                chunk_attributes_records.append(
                    {k: v for k, v in chunk_record.items() if v is not None}
                )
                for key, value in resolved_attributes.items():
                    if key not in merged_document_attributes:
                        merged_document_attributes[key] = value
                chunk.origin = document.origin

            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=chunk_attributes_records,
            )
            stale_ids = [
                existing_id for existing_id in existing_ids if existing_id not in ids
            ]
            if stale_ids:
                self.collection.delete(ids=stale_ids)
            current_document = MarkdownDocument(
                origin=document.origin,
                content=document.content,
                chunks=document.chunks,
                attributes=merged_document_attributes or None,
            )
            return WriteResult(
                action="replaced" if existing_ids else "inserted",
                document=current_document,
                replaced_document=replaced_document,
            )

    @contextmanager
    def _origin_lock(self, origin: str):
        with self._origin_locks_guard:
            lock = self._origin_locks.get(origin)
            if lock is None:
                lock = threading.Lock()
                self._origin_locks[origin] = lock
            self._origin_lock_ref_counts[origin] = (
                self._origin_lock_ref_counts.get(origin, 0) + 1
            )

        try:
            with lock:
                yield
        finally:
            with self._origin_locks_guard:
                remaining = self._origin_lock_ref_counts.get(origin, 1) - 1
                if remaining <= 0:
                    self._origin_lock_ref_counts.pop(origin, None)
                    if self._origin_locks.get(origin) is lock:
                        self._origin_locks.pop(origin, None)
                else:
                    self._origin_lock_ref_counts[origin] = remaining

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
            If None, defaults to 4 to avoid API rate limiting.
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
            num_workers = 4

        if prepare is None:
            chunker = MarkdownChunker()

            def default_prepare(uri: str) -> Document:
                return chunker.chunk_document(read_as_markdown(uri))

            prepare = default_prepare

        total = len(items) if isinstance(items, Sized) else None

        def do_ingest_work(item: Any) -> None:
            try:
                doc = prepare(item)
                self.upsert(doc)
            except Exception as e:
                raise RuntimeError(f"Failed to ingest '{item}': {e}") from e

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for future in tqdm(
                lazy_map(pool, do_ingest_work, items), total=total, disable=not progress
            ):
                future.result()

    def retrieve(
        self,
        text: str,
        top_k: int,
        *,
        deoverlap: bool = True,
        attributes_filter: Optional[AttributeFilter] = None,
        **kwargs,
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
            original ranges, combines metrics, and aggregates attribute values
            into per-chunk lists in start-order. The `context` value is kept
            from the first chunk in each merged overlap group.
        attributes_filter
            Optional attribute filter as SQL-like string or dict AST.
            Example string: `"tenant = 'docs' AND priority >= 2"`.
            Supports declared attributes plus built-in columns:
            `chunk_id`, `start_index`, `end_index`,
            `token_count`, `context`, and `origin`.
        **kwargs
            Additional arguments passed to ChromaDB's `query()` method.

        Returns
        -------
        Sequence[RetrievedChromaDBMarkdownChunk]
            The retrieved chunks with their relevance metrics.
        """
        if attributes_filter is not None:
            if "where" in kwargs:
                raise ValueError("Use either attributes_filter or where, not both.")
            kwargs["where"] = compile_filter_to_chroma_where(
                attributes_filter,
                allowed_columns=self._filterable_columns(),
            )

        with self._collection_lock:
            results = self.collection.query(
                query_texts=[text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
                **kwargs,
            )

        documents = (results.get("documents") or [[]])[0]
        chunk_attributes_rows = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        output: list[RetrievedChromaDBMarkdownChunk] = []
        for doc_text, chunk_attributes, distance in zip(
            documents, chunk_attributes_rows, distances, strict=False
        ):
            chunk_attributes = chunk_attributes or {}
            user_attributes = {
                key: chunk_attributes.get(key)
                for key in self.metadata.attributes_schema
            }
            start_index = int(chunk_attributes.get("start_index", 0))
            end_index = int(
                chunk_attributes.get("end_index", start_index + len(doc_text))
            )
            token_count = int(chunk_attributes.get("token_count", len(doc_text)))
            metrics = []
            if distance is not None:
                metrics.append(Metric(name="distance", value=distance))
            chunk = RetrievedChromaDBMarkdownChunk(
                text=doc_text,
                start_index=start_index,
                end_index=end_index,
                context=chunk_attributes.get("context"),
                token_count=token_count,
                origin=chunk_attributes.get("origin"),
                chunk_ids=(
                    []
                    if chunk_attributes.get("chunk_id") is None
                    else [int(chunk_attributes["chunk_id"])]
                ),
                metrics=metrics,
                attributes=user_attributes,
            )
            output.append(chunk)

        if deoverlap:
            output = deoverlap_chunks(output, key=lambda c: c.origin)

        return output

    def _get_all_metadatas(self) -> list[dict[str, Any]]:
        all_metadatas: list[dict[str, Any]] = []
        batch_size = 5000
        offset = 0
        while True:
            results = self.collection.get(
                include=["metadatas"], limit=batch_size, offset=offset
            )
            batch = results.get("metadatas") or []
            all_metadatas.extend(batch)
            if len(batch) < batch_size:
                break
            offset += batch_size
        return all_metadatas

    def size(self) -> int:
        with self._collection_lock:
            chunk_attributes_rows = self._get_all_metadatas()
        origins = {
            chunk_attributes.get("origin")
            for chunk_attributes in chunk_attributes_rows
            if chunk_attributes and chunk_attributes.get("origin")
        }
        return len(origins)

    def _filterable_columns(self) -> set[str]:
        return _FILTERABLE_BASE_COLUMNS | set(self.metadata.attributes_schema)

    def _incoming_chunk_signature(
        self, document: MarkdownDocument
    ) -> list[tuple[Any, ...]]:
        signatures: list[tuple[Any, ...]] = []
        attribute_columns = list(self.metadata.attributes_schema)
        for chunk in document.chunks or []:
            resolved_attributes = merge_attribute_values(
                attributes_spec=self.metadata.attributes_spec,
                sources=[document.attributes, chunk.attributes],
            )
            signatures.append(
                (
                    chunk.start_index,
                    chunk.end_index,
                    chunk.context,
                    chunk.token_count,
                    chunk.text,
                    *[resolved_attributes.get(col) for col in attribute_columns],
                )
            )
        signatures.sort(key=self._chunk_signature_sort_key)
        return signatures

    def _existing_chunk_signature(
        self, existing: dict[str, Any]
    ) -> list[tuple[Any, ...]]:
        chunk_texts = list(existing.get("documents") or [])
        chunk_metadatas = list(existing.get("metadatas") or [])
        attribute_columns = list(self.metadata.attributes_schema)
        signatures: list[tuple[Any, ...]] = []
        for chunk_text, metadata in zip(chunk_texts, chunk_metadatas, strict=False):
            metadata = metadata or {}
            start_index = int(metadata.get("start_index", 0))
            signatures.append(
                (
                    start_index,
                    int(metadata.get("end_index", start_index + len(chunk_text))),
                    metadata.get("context"),
                    int(metadata.get("token_count", len(chunk_text))),
                    chunk_text,
                    *[metadata.get(col) for col in attribute_columns],
                )
            )
        signatures.sort(key=self._chunk_signature_sort_key)
        return signatures

    def _chunk_signature_sort_key(self, row: tuple[Any, ...]) -> tuple[str, ...]:
        return tuple(f"{type(value).__name__}:{value!r}" for value in row)

    def _snapshot_document_from_existing_if_available(
        self, existing: dict[str, Any], *, origin: str
    ) -> Optional[MarkdownDocument]:
        try:
            return self._snapshot_document_from_existing(existing, origin=origin)
        except ValueError as exc:
            if f"missing required {_CONTENT_TEXT_METADATA_KEY}" not in str(exc):
                raise
            return None

    def _snapshot_document_from_existing(
        self, existing: dict[str, Any], *, origin: str
    ) -> MarkdownDocument:
        chunk_texts = list(existing.get("documents") or [])
        chunk_metadatas = list(existing.get("metadatas") or [])

        content: str | None = None
        for metadata in chunk_metadatas:
            if metadata and _CONTENT_TEXT_METADATA_KEY in metadata:
                content = metadata[_CONTENT_TEXT_METADATA_KEY]
                break
        if content is None:
            raise ValueError(
                f"Corrupted Chroma store for origin '{origin}': missing required {_CONTENT_TEXT_METADATA_KEY} in chunk metadata"
            )

        chunk_rows: list[tuple[int, int, int, Chunk]] = []
        document_attributes: dict[str, Any] = {}
        for idx, (chunk_text, metadata) in enumerate(
            zip(chunk_texts, chunk_metadatas, strict=False)
        ):
            if metadata is None:
                raise ValueError(
                    f"Corrupted Chroma store for origin '{origin}': missing chunk metadata"
                )
            chunk_id = int(metadata.get("chunk_id", idx))
            start_index = int(metadata.get("start_index", 0))
            end_index = int(metadata.get("end_index", start_index + len(chunk_text)))
            attributes = {
                key: metadata.get(key)
                for key in self.metadata.attributes_schema
                if metadata.get(key) is not None
            }
            for key, value in attributes.items():
                if key not in document_attributes:
                    document_attributes[key] = value
            chunk_rows.append(
                (
                    start_index,
                    end_index,
                    chunk_id,
                    MarkdownChunk(
                        text=chunk_text,
                        start_index=start_index,
                        end_index=end_index,
                        token_count=int(metadata.get("token_count", len(chunk_text))),
                        context=metadata.get("context"),
                        origin=origin,
                        attributes=attributes or None,
                    ),
                )
            )

        chunk_rows.sort(key=lambda row: (row[0], row[1], row[2]))
        chunks = [chunk for _, _, _, chunk in chunk_rows]

        return MarkdownDocument(
            origin=origin,
            content=content,
            chunks=chunks,
            attributes=document_attributes or None,
        )

    def _allocate_chunk_ids(self, count: int) -> list[int]:
        if count <= 0:
            return []
        with self._chunk_id_lock:
            if self._next_chunk_id is None:
                chunk_attributes_rows = self._get_all_metadatas()
                max_chunk_id = -1
                for chunk_attributes in chunk_attributes_rows:
                    if not chunk_attributes:
                        continue
                    value = chunk_attributes.get("chunk_id")
                    if value is None:
                        continue
                    max_chunk_id = max(max_chunk_id, int(value))
                self._next_chunk_id = max_chunk_id + 1
            start = self._next_chunk_id
            self._next_chunk_id += count
            return list(range(start, start + count))
