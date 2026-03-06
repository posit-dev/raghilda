import openai
import json
import hashlib
import threading
import logging
from contextlib import contextmanager
from ._store import BaseStore, WriteResult
from .chunk import MarkdownChunk, RetrievedChunk, Metric
from .document import Document, MarkdownDocument
from typing import Any, Mapping, Optional, Sequence
from dataclasses import dataclass
from ._attributes import (
    AttributeFilter,
    AttributesSchemaSpec,
    AttributeSpec,
    AttributeType,
    AttributeValue,
    attributes_spec_from_json_dict,
    attributes_spec_to_json_dict,
    compile_filter_to_openai_filters,
    merge_attribute_values,
    normalize_attributes_spec,
)

_ATTRIBUTES_SCHEMA_METADATA_KEY = "raghilda_attributes_schema_json"
_INTERNAL_ORIGIN_ATTRIBUTE_KEY = "_raghilda_origin"
_INTERNAL_CONTENT_HASH_ATTRIBUTE_KEY = "_raghilda_content_hash"
_RESERVED_INTERNAL_ATTRIBUTE_KEYS = {
    _INTERNAL_ORIGIN_ATTRIBUTE_KEY,
    _INTERNAL_CONTENT_HASH_ATTRIBUTE_KEY,
}
_OPENAI_MAX_FILE_ATTRIBUTES = 16
_OPENAI_INTERNAL_ATTRIBUTE_COUNT = 2

logger = logging.getLogger(__name__)


def _ensure_openai_user_attribute_limit(user_attribute_count: int) -> None:
    if (
        user_attribute_count + _OPENAI_INTERNAL_ATTRIBUTE_COUNT
        > _OPENAI_MAX_FILE_ATTRIBUTES
    ):
        raise ValueError(
            "OpenAI vector store files support at most 16 total attributes; "
            f"received {user_attribute_count} user attributes plus "
            "2 internal attributes. Use at most 14 user attributes."
        )


@dataclass(repr=False)
class OpenAIMarkdownChunk(MarkdownChunk):
    """MarkdownChunk for OpenAI store."""

    def __init__(
        self,
        text: str,
        start_index: int = 0,
        end_index: Optional[int] = None,
        context=None,
        char_count=None,
        origin=None,
        attributes=None,
    ):
        if char_count is None:
            char_count = len(text)

        if end_index is None:
            end_index = len(text)

        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            char_count=char_count,
            context=context,
            origin=origin,
            attributes=attributes,
        )


@dataclass(repr=False)
class RetrievedOpenAIMarkdownChunk(OpenAIMarkdownChunk, RetrievedChunk):
    """OpenAIMarkdownChunk with retrieval metrics"""

    def __init__(
        self,
        text: str,
        start_index: int = 0,
        end_index: Optional[int] = None,
        context=None,
        char_count=None,
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
            char_count=char_count,
            origin=origin,
            attributes=attributes,
        )

        # Initialize metrics
        if metrics is None:
            metrics = []
        self.metrics = metrics
        if chunk_ids is None:
            chunk_ids = []
        self.chunk_ids = chunk_ids


class OpenAIStore(BaseStore):
    """A vector store backed by OpenAI's Vector Store API.

    OpenAIStore uses OpenAI's hosted vector storage service for document
    storage and retrieval. Documents are uploaded as files and automatically
    chunked and embedded by OpenAI.

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.store import OpenAIStore

    # Create a new store
    store = OpenAIStore.create(name="my-store")

    # Or connect to an existing store
    store = OpenAIStore.connect(store_id="vs_abc123")

    # Insert documents
    from raghilda.document import MarkdownDocument
    doc = MarkdownDocument(content="# Hello\\nWorld", origin="example.md")
    store.upsert(doc)

    # Retrieve similar chunks
    chunks = store.retrieve("greeting", top_k=5)
    ```
    """

    @staticmethod
    def create(
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        *,
        attributes: Optional[AttributesSchemaSpec] = None,
        metadata: Optional[Mapping[str, str]] = None,
        **kwargs,
    ):
        """Create a new OpenAI vector store.

        Parameters
        ----------
        base_url
            Base URL for the OpenAI API.
        api_key
            OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        attributes
            Optional schema for user-defined attribute columns.
            Attribute names use identifier-style syntax.
            OpenAIStore filters only support declared attributes.
        metadata
            Additional metadata to attach to the OpenAI vector store resource.
        **kwargs
            Additional arguments passed to the vector store creation
            (e.g., name, expires_after).

        Returns
        -------
        OpenAIStore
            A newly created store instance.
        """
        attributes_spec = normalize_attributes_spec(
            attributes=attributes,
            reserved_columns=_RESERVED_INTERNAL_ATTRIBUTE_KEYS,
            allow_vector_types=False,
            allow_struct_types=False,
            allow_optional_values=False,
        )
        attributes_schema = {
            key: spec.attribute_type for key, spec in attributes_spec.items()
        }
        _ensure_openai_user_attribute_limit(len(attributes_spec))

        client = openai.Client(api_key=api_key, base_url=base_url)
        api_vector_store_metadata = dict(metadata or {})
        api_vector_store_metadata[_ATTRIBUTES_SCHEMA_METADATA_KEY] = json.dumps(
            attributes_spec_to_json_dict(attributes_spec)
        )
        kwargs["metadata"] = api_vector_store_metadata

        vector_store = client.vector_stores.create(**kwargs)
        return OpenAIStore(
            client,
            vector_store.id,
            attributes_spec=attributes_spec,
            attributes=attributes_schema,
        )

    @staticmethod
    def connect(
        store_id: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
    ):
        """Connect to an existing OpenAI vector store.

        Parameters
        ----------
        store_id
            The ID of the vector store to connect to (e.g., "vs_abc123").
        base_url
            Base URL for the OpenAI API.
        api_key
            OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.

        Returns
        -------
        OpenAIStore
            A connected store instance.
        """
        client = openai.Client(api_key=api_key, base_url=base_url)
        vector_store = client.vector_stores.retrieve(vector_store_id=store_id)
        store_metadata = getattr(vector_store, "metadata", None) or {}
        if _ATTRIBUTES_SCHEMA_METADATA_KEY not in store_metadata:
            raise ValueError(
                f"OpenAI vector store metadata is missing required key '{_ATTRIBUTES_SCHEMA_METADATA_KEY}'."
            )
        resolved_attributes_spec = attributes_spec_from_json_dict(
            json.loads(store_metadata[_ATTRIBUTES_SCHEMA_METADATA_KEY]),
            allow_vector_types=False,
            allow_struct_types=False,
            allow_optional_values=False,
        )
        _ensure_no_reserved_attributes(
            resolved_attributes_spec,
            _RESERVED_INTERNAL_ATTRIBUTE_KEYS,
        )
        resolved_attributes_schema = {
            key: spec.attribute_type for key, spec in resolved_attributes_spec.items()
        }

        return OpenAIStore(
            client,
            store_id,
            attributes_spec=resolved_attributes_spec,
            attributes=resolved_attributes_schema,
        )

    def __init__(
        self,
        client: Any,
        store_id: str,
        *,
        attributes_spec: Optional[Mapping[str, AttributeSpec]] = None,
        attributes: Optional[Mapping[str, AttributeType]] = None,
    ):
        self.client = client
        self.store_id = store_id
        if attributes_spec is not None:
            resolved_spec = dict(attributes_spec)
        elif attributes is not None:
            resolved_spec = normalize_attributes_spec(
                attributes=attributes,
                reserved_columns=_RESERVED_INTERNAL_ATTRIBUTE_KEYS,
                allow_vector_types=False,
                allow_struct_types=False,
                allow_optional_values=False,
            )
        else:
            resolved_spec = {}

        if attributes is not None:
            resolved_schema = dict(attributes)
        else:
            resolved_schema = {
                key: spec.attribute_type for key, spec in resolved_spec.items()
            }
        _ensure_openai_user_attribute_limit(len(resolved_spec))

        self.attributes_spec = resolved_spec
        self.attributes_schema = resolved_schema
        self._origin_locks: dict[str, threading.Lock] = {}
        self._origin_lock_ref_counts: dict[str, int] = {}
        self._origin_locks_guard = threading.Lock()

    def upsert(
        self,
        document: Document,
        *,
        skip_if_unchanged: bool = True,
    ) -> WriteResult:
        # Upload the document content as a file to the vector store
        # create a temporary file, write the content to it, and upload it
        if not isinstance(document, MarkdownDocument):
            raise ValueError("Only MarkdownDocument is supported for OpenAIStore")
        if not isinstance(document.origin, str) or not document.origin:
            raise ValueError("document.origin must be a non-empty string for upsert().")

        if document.chunks is not None:
            raise ValueError("OpenAIStore does not support chunked documents.")

        resolved_attributes = merge_attribute_values(
            attributes_spec=self.attributes_spec,
            sources=[document.attributes],
        )
        user_file_attributes = _normalize_openai_attributes(resolved_attributes)
        _ensure_openai_user_attribute_limit(len(user_file_attributes))

        with self._origin_lock(document.origin):
            content_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            existing_files = [
                vector_store_file
                for vector_store_file in self._iter_vector_store_files()
                if self._matches_existing_origin(vector_store_file, document.origin)
            ]
            matching_files = [
                vector_store_file
                for vector_store_file in existing_files
                if self._openai_file_matches_insert_request(
                    vector_store_file=vector_store_file,
                    expected_content_hash=content_hash,
                    expected_user_attributes=user_file_attributes,
                )
            ]
            replaced_document = None
            if existing_files and skip_if_unchanged and matching_files:
                keep_file = self._select_primary_vector_store_file(matching_files)
                duplicate_files = [
                    vector_store_file
                    for vector_store_file in existing_files
                    if getattr(vector_store_file, "id", None)
                    != getattr(keep_file, "id", None)
                ]
                for vector_store_file in duplicate_files:
                    try:
                        self.client.vector_stores.files.delete(
                            file_id=vector_store_file.id,
                            vector_store_id=self.store_id,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Skipping duplicate managed file cleanup for origin '%s' "
                            "because delete failed for file '%s': %s",
                            document.origin,
                            getattr(vector_store_file, "id", None),
                            exc,
                        )
                current_document = self._snapshot_document_from_file(keep_file)
                if current_document is None:
                    current_document = MarkdownDocument(
                        origin=document.origin,
                        content=document.content,
                        attributes=document.attributes,
                    )
                return WriteResult(
                    action="skipped",
                    document=current_document,
                )

            if existing_files:
                replaced_document = self._snapshot_document_from_file(
                    self._select_primary_vector_store_file(existing_files)
                )

            file_attributes = {
                **user_file_attributes,
                _INTERNAL_ORIGIN_ATTRIBUTE_KEY: document.origin,
                _INTERNAL_CONTENT_HASH_ATTRIBUTE_KEY: content_hash,
            }

            file = (document.origin + ".md", document.content.encode("utf-8"))
            if file_attributes:
                uploaded_file = self.client.vector_stores.files.upload_and_poll(
                    file=file,
                    vector_store_id=self.store_id,
                    attributes=file_attributes,
                )
            else:
                uploaded_file = self.client.vector_stores.files.upload_and_poll(
                    file=file,
                    vector_store_id=self.store_id,
                )
            uploaded_file_id = getattr(uploaded_file, "id", None)
            if uploaded_file_id is None:
                raise ValueError("OpenAI upload response missing file id.")
            if existing_files:
                try:
                    for vector_store_file in existing_files:
                        self.client.vector_stores.files.delete(
                            file_id=vector_store_file.id,
                            vector_store_id=self.store_id,
                        )
                except Exception:
                    try:
                        self.client.vector_stores.files.delete(
                            file_id=uploaded_file_id,
                            vector_store_id=self.store_id,
                        )
                    except Exception:
                        pass
                    raise
            current_document = MarkdownDocument(
                origin=document.origin,
                content=document.content,
                attributes=document.attributes,
            )
            return WriteResult(
                action="replaced" if existing_files else "inserted",
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

    def _openai_file_matches_insert_request(
        self,
        *,
        vector_store_file: Any,
        expected_content_hash: str,
        expected_user_attributes: Mapping[str, str | int | float | bool],
    ) -> bool:
        attributes = dict(getattr(vector_store_file, "attributes", None) or {})
        if (
            attributes.get(_INTERNAL_CONTENT_HASH_ATTRIBUTE_KEY)
            != expected_content_hash
        ):
            return False
        for key in self.attributes_schema:
            if attributes.get(key) != expected_user_attributes.get(key):
                return False
        return True

    def _select_primary_vector_store_file(
        self, vector_store_files: Sequence[Any]
    ) -> Any:
        return max(
            vector_store_files,
            key=lambda vector_store_file: (
                int(getattr(vector_store_file, "created_at", -1) or -1),
                str(getattr(vector_store_file, "id", "")),
            ),
        )

    def retrieve(
        self,
        text: str,
        top_k: int,
        *,
        attributes_filter: Optional[AttributeFilter] = None,
        **kwargs,
    ) -> Sequence[RetrievedOpenAIMarkdownChunk]:
        """Retrieve the most similar chunks to the given text.

        Parameters
        ----------
        text
            The query text to search for.
        top_k
            The maximum number of chunks to return.
        attributes_filter
            Optional attribute filter as SQL-like string or dict AST.
            Supports declared attributes only. Built-in columns such as
            `origin` are not available in OpenAI filters.
        **kwargs
            Additional arguments passed to OpenAI's `vector_stores.search()`.

        Returns
        -------
        Sequence[RetrievedOpenAIMarkdownChunk]
            The retrieved chunks with their relevance metrics.
        """
        if attributes_filter is not None:
            if "filters" in kwargs:
                raise ValueError("Use either attributes_filter or filters, not both.")
            kwargs["filters"] = compile_filter_to_openai_filters(
                attributes_filter,
                allowed_columns=set(self.attributes_schema),
            )

        results = self.client.vector_stores.search(
            vector_store_id=self.store_id,
            query=text,
            max_num_results=top_k,
            **kwargs,
        )

        chunks = []
        for item in results.data:
            chunk_text = "\n\n".join([x.text for x in item.content])
            attribute_values = {
                key: (item.attributes or {}).get(key) for key in self.attributes_schema
            }
            chunk = RetrievedOpenAIMarkdownChunk(
                text=chunk_text,
                metrics=[Metric(name="similarity", value=item.score)],
                origin=(item.attributes or {}).get(_INTERNAL_ORIGIN_ATTRIBUTE_KEY),
                attributes=attribute_values,
            )
            chunks.append(chunk)

        return chunks

    def size(self):
        return self.client.vector_stores.retrieve(
            vector_store_id=self.store_id
        ).file_counts.total

    def _iter_vector_store_files(self):
        page = self.client.vector_stores.files.list(
            vector_store_id=self.store_id,
            limit=100,
        )
        while True:
            for vector_store_file in page.data:
                yield vector_store_file
            if not page.has_next_page():
                break
            page = page.get_next_page()

    def _snapshot_document_from_file(
        self, vector_store_file: Any
    ) -> Optional[MarkdownDocument]:
        attributes = dict(getattr(vector_store_file, "attributes", None) or {})
        origin = self._origin_from_vector_store_file(vector_store_file)
        if not origin:
            origin = getattr(vector_store_file, "id")

        try:
            response = self.client.files.content(file_id=vector_store_file.id)
        except openai.APIError:
            return None
        raw = response.content
        content = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        user_attributes = {
            key: value
            for key, value in attributes.items()
            if key in self.attributes_schema
        }
        return MarkdownDocument(
            origin=origin,
            content=content,
            attributes=user_attributes or None,
        )

    def _origin_from_vector_store_file(self, vector_store_file: Any) -> Optional[str]:
        attributes = dict(getattr(vector_store_file, "attributes", None) or {})
        origin = attributes.get(_INTERNAL_ORIGIN_ATTRIBUTE_KEY)
        if origin:
            return str(origin)
        return None

    def _matches_existing_origin(self, vector_store_file: Any, origin: str) -> bool:
        attributes = dict(getattr(vector_store_file, "attributes", None) or {})
        managed_origin = attributes.get(_INTERNAL_ORIGIN_ATTRIBUTE_KEY)
        return str(managed_origin) == origin if managed_origin else False


def _normalize_openai_attributes(
    attributes: Mapping[str, AttributeValue],
) -> dict[str, str | int | float | bool]:
    out: dict[str, str | int | float | bool] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, str):
            out[key] = value
        elif isinstance(value, int):
            out[key] = value
        elif isinstance(value, float):
            out[key] = value
        else:
            raise ValueError(
                f"Unsupported OpenAI attribute type for '{key}': {type(value).__name__}"
            )
    return out


def _ensure_no_reserved_attributes(
    attributes_spec: Mapping[str, AttributeSpec],
    reserved_keys: set[str],
) -> None:
    for key in attributes_spec:
        if key in reserved_keys:
            raise ValueError(f"Attribute column '{key}' is reserved")
