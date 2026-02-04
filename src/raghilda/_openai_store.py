import openai
from ._store import BaseStore
from .chunk import MarkdownChunk, RetrievedChunk, Metric
from .document import Document, MarkdownDocument
from typing import Optional, Sequence
from dataclasses import dataclass


@dataclass
class OpenAIMarkdownChunk(MarkdownChunk):
    """MarkdownChunk for OpenAI store - uses character count as token count"""

    def __init__(
        self,
        text: str,
        start_index: int = 0,
        end_index: Optional[int] = None,
        context=None,
        token_count=None,
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
        )


@dataclass
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
    ):
        # Initialize OpenAIMarkdownChunk
        super().__init__(
            text=text,
            start_index=start_index,
            end_index=end_index,
            context=context,
            token_count=token_count,
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
        **kwargs,
    ):
        """Create a new OpenAI vector store.

        Parameters
        ----------
        base_url
            Base URL for the OpenAI API.
        api_key
            OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        **kwargs
            Additional arguments passed to the vector store creation
            (e.g., name, expires_after).

        Returns
        -------
        OpenAIStore
            A newly created store instance.
        """
        client = openai.Client(api_key=api_key, base_url=base_url)
        vector_store = client.vector_stores.create(**kwargs)
        return OpenAIStore(client, vector_store.id)

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
        return OpenAIStore(client, store_id)

    def __init__(self, client: openai.Client, store_id: str):
        self.client = client
        self.store_id = store_id

    def insert(self, document: Document) -> None:
        # Upload the document content as a file to the vector store
        # create a temporary file, write the content to it, and upload it
        if not isinstance(document, MarkdownDocument):
            raise ValueError("Only MarkdownDocument is supported for OpenAIStore")

        self.client.vector_stores.files.upload_and_poll(
            file=((document.origin or "") + ".md", document.content.encode("utf-8")),
            vector_store_id=self.store_id,
        )

    def retrieve(
        self, text: str, top_k: int, **kwargs
    ) -> Sequence[RetrievedOpenAIMarkdownChunk]:
        results = self.client.vector_stores.search(
            vector_store_id=self.store_id,
            query=text,
            max_num_results=top_k,
            **kwargs,
        )

        chunks = []
        for item in results.data:
            chunk_text = "\n\n".join([x.text for x in item.content])
            chunk = RetrievedOpenAIMarkdownChunk(
                text=chunk_text,
                metrics=[Metric(name="similarity", value=item.score)],
            )
            chunks.append(chunk)

        return chunks

    def size(self):
        return self.client.vector_stores.retrieve(
            vector_store_id=self.store_id
        ).file_counts.total
