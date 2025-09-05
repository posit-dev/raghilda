import openai
from ._store import Store
from .document import (
    Document,
    RetrievedChunk,
    MarkdownDocument,
    ChunkedDocument,
    Metric,
)
from typing import Optional, Sequence


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

    def insert(self, document: Document | ChunkedDocument) -> None:
        # Upload the document content as a file to the vector store
        # create a temporary file, write the content to it, and upload it
        if not isinstance(document, MarkdownDocument):
            raise ValueError("Only MarkdownDocument is supported for OpenAIStore")

        self.client.vector_stores.files.upload_and_poll(
            file=((document.origin or "") + ".md", document.content.encode("utf-8")),
            vector_store_id=self.store_id,
        )

    def retrieve(self, text: str, top_k: int) -> Sequence[RetrievedChunk]:
        results = self.client.vector_stores.search(
            vector_store_id=self.store_id, query=text, max_num_results=top_k
        )

        chunks = []
        for item in results.data:
            chunk = RetrievedChunk(
                content="\n\n".join([x.text for x in item.content]),
                metrics=[Metric(name="similarity", value=item.score)],
            )
            chunks.append(chunk)

        return chunks

    def size(self):
        return self.client.vector_stores.retrieve(
            vector_store_id=self.store_id
        ).file_counts.total
