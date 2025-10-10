import openai
from ._store import Store
from ._chunker import MarkdownChunk
from .document import (
    Document,
    RetrievedChunk,
    MarkdownDocument,
    Metric,
)
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


class OpenAIStore(Store):
    @staticmethod
    def create(
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        client = openai.Client(api_key=api_key, base_url=base_url)
        vector_store = client.vector_stores.create(**kwargs)
        return OpenAIStore(client, vector_store.id)

    @staticmethod
    def connect(
        store_id: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
    ):
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
