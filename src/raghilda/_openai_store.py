import openai
import json
from ._store import BaseStore
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


@dataclass(repr=False)
class OpenAIMarkdownChunk(MarkdownChunk):
    """MarkdownChunk for OpenAI store - uses character count as token count"""

    def __init__(
        self,
        text: str,
        start_index: int = 0,
        end_index: Optional[int] = None,
        context=None,
        token_count=None,
        attributes=None,
    ):
        # Compute token_count if not provided (use character count)
        if token_count is None:
            token_count = len(text)

        # Compute end_index if not provided
        if end_index is None:
            end_index = len(text)

        # Initialize parent class
        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            token_count=token_count,
            context=context,
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
        token_count=None,
        metrics=None,
        attributes=None,
    ):
        # Initialize OpenAIMarkdownChunk
        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            context=context,
            token_count=token_count,
            attributes=attributes,
        )

        # Initialize metrics
        if metrics is None:
            metrics = []
        self.metrics = metrics


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
    store.insert(doc)

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
            reserved_columns=set(),
            allow_vector_types=False,
            allow_struct_types=False,
            allow_optional_values=False,
        )
        attributes_schema = {
            key: spec.attribute_type for key, spec in attributes_spec.items()
        }

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
        *,
        attributes: Optional[AttributesSchemaSpec] = None,
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
        attributes
            Optional schema for user-defined attribute columns. If omitted,
            schema is loaded from the vector store metadata when available.
            Attribute names use identifier-style syntax.

        Returns
        -------
        OpenAIStore
            A connected store instance.
        """
        client = openai.Client(api_key=api_key, base_url=base_url)
        vector_store = client.vector_stores.retrieve(vector_store_id=store_id)
        store_metadata = getattr(vector_store, "metadata", None) or {}

        resolved_attributes_spec = normalize_attributes_spec(
            attributes=attributes,
            reserved_columns=set(),
            allow_vector_types=False,
            allow_struct_types=False,
            allow_optional_values=False,
        )
        if not resolved_attributes_spec and store_metadata.get(
            _ATTRIBUTES_SCHEMA_METADATA_KEY
        ):
            resolved_attributes_spec = attributes_spec_from_json_dict(
                json.loads(store_metadata[_ATTRIBUTES_SCHEMA_METADATA_KEY]),
                allow_vector_types=False,
                allow_struct_types=False,
                allow_optional_values=False,
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
                reserved_columns=set(),
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

        self.attributes_spec = resolved_spec
        self.attributes_schema = resolved_schema

    def insert(
        self,
        document: Document,
    ) -> None:
        # Upload the document content as a file to the vector store
        # create a temporary file, write the content to it, and upload it
        if not isinstance(document, MarkdownDocument):
            raise ValueError("Only MarkdownDocument is supported for OpenAIStore")

        if document.chunks is not None:
            for chunk in document.chunks:
                if chunk.attributes:
                    raise ValueError(
                        "OpenAIStore does not support per-chunk attributes; use document-level attributes."
                    )

        resolved_attributes = merge_attribute_values(
            attributes_spec=self.attributes_spec,
            sources=[document.attributes],
        )
        file_attributes = _normalize_openai_attributes(resolved_attributes)

        file = ((document.origin or "") + ".md", document.content.encode("utf-8"))
        if file_attributes:
            self.client.vector_stores.files.upload_and_poll(
                file=file,
                vector_store_id=self.store_id,
                attributes=file_attributes,
            )
        else:
            self.client.vector_stores.files.upload_and_poll(
                file=file,
                vector_store_id=self.store_id,
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
            `doc_id` and `origin` are not available in OpenAI filters.
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
                attributes=attribute_values,
            )
            chunks.append(chunk)

        return chunks

    def size(self):
        return self.client.vector_stores.retrieve(
            vector_store_id=self.store_id
        ).file_counts.total


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
