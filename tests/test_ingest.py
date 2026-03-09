from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

from raghilda.chunk import Chunk, MarkdownChunk
from raghilda.document import MarkdownDocument
from raghilda.ingest import Ingestor, ItemError, ingest
from raghilda.store import (
    BaseStore,
    ChromaDBStore,
    DuckDBStore,
    OpenAIStore,
    WriteResult,
)


class _RecordingStore(BaseStore):
    def __init__(self) -> None:
        self.documents: list[MarkdownDocument] = []

    @staticmethod
    def connect(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def create(*args, **kwargs):
        raise NotImplementedError

    def upsert(
        self,
        document,
        *,
        skip_if_unchanged: bool = True,
    ) -> WriteResult:
        assert isinstance(document, MarkdownDocument)
        self.documents.append(document)
        return WriteResult(action="inserted", document=document)

    def retrieve(self, text: str, top_k: int, *args, **kwargs):
        raise NotImplementedError

    def size(self) -> int:
        return len(self.documents)


class _SlowUpsertStore(_RecordingStore):
    def upsert(
        self,
        document,
        *,
        skip_if_unchanged: bool = True,
    ) -> WriteResult:
        assert isinstance(document, MarkdownDocument)
        if document.origin == "slow":
            time.sleep(1.0)
        return super().upsert(document, skip_if_unchanged=skip_if_unchanged)


class _SinglePage:
    def __init__(self):
        self.data = []

    def has_next_page(self):
        return False


class _FakeVectorStoreFiles:
    def __init__(self):
        self.upload_calls = []

    def list(self, **kwargs):
        return _SinglePage()

    def upload_and_poll(self, **kwargs):
        self.upload_calls.append(kwargs)
        return SimpleNamespace(id="file_new")

    def delete(self, **kwargs):
        raise AssertionError("delete should not be called")


def _make_fake_openai_store() -> tuple[OpenAIStore, _FakeVectorStoreFiles]:
    fake_vector_store_files = _FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=SimpleNamespace(content=lambda **kwargs: None),
    )
    return OpenAIStore(client=fake_client, store_id="vs_test"), fake_vector_store_files


def _chunked_doc(origin: str, content: str) -> MarkdownDocument:
    return MarkdownDocument(
        origin=origin,
        content=content,
        chunks=[
            MarkdownChunk(
                start_index=0,
                end_index=len(content),
                text=content,
                char_count=len(content),
            )
        ],
    )


def _write_markdown_files(tmp_path: Path, names: list[str]) -> list[str]:
    paths = []
    for name in names:
        path = tmp_path / f"{name}.md"
        path.write_text(f"# {name}\n\n{name} content\n", encoding="utf-8")
        paths.append(str(path))
    return paths


def test_ingest_list_input(tmp_path):
    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_list",
    )
    paths = _write_markdown_files(tmp_path, ["first", "second"])

    results = ingest(paths, store=store, progress=False)

    assert results.inserted == 2
    assert results.replaced == 0
    assert results.skipped == 0
    assert results.failed == 0
    assert results.pending == 0
    assert store.size() == 2


def test_ingest_result_keeps_item_scoped_write_outcomes():
    store = _RecordingStore()
    items = [
        {"id": "doc1", "text": "First document content"},
        {"id": "doc2", "text": "Second document content"},
    ]

    def custom_prepare(item: dict[str, str]) -> MarkdownDocument:
        return _chunked_doc(origin=item["id"], content=item["text"])

    results = ingest(
        items,
        store=store,
        prepare=custom_prepare,
        num_workers=1,
        progress=False,
    )

    assert [outcome.item for outcome in results.outcomes] == items
    assert [outcome.status for outcome in results.outcomes] == ["inserted", "inserted"]
    assert [write.document.origin for write in results.write_results] == [
        "doc1",
        "doc2",
    ]


def test_ingest_with_generator_uses_bounded_lazy_consumption():
    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_generator",
    )

    consumed_count = 0
    inserted_count = 0
    max_pending = 0
    lock = threading.Lock()

    def tracking_generator():
        nonlocal consumed_count, max_pending
        for i in range(20):
            with lock:
                consumed_count += 1
                pending = consumed_count - inserted_count
                if pending > max_pending:
                    max_pending = pending
            yield {"id": f"doc_{i}", "text": f"Document content {i}"}

    def slow_prepare(item: dict[str, str]) -> MarkdownDocument:
        nonlocal inserted_count
        time.sleep(0.02)
        doc = _chunked_doc(origin=item["id"], content=item["text"])
        with lock:
            inserted_count += 1
        return doc

    results = ingest(
        tracking_generator(),
        store=store,
        prepare=slow_prepare,
        num_workers=2,
        progress=False,
    )

    assert results.inserted == 20
    assert store.size() == 20
    assert max_pending <= 4
    assert results.pending is None


def test_ingestor_run_supports_custom_prepare():
    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_custom_prepare",
    )

    def custom_prepare(item: dict[str, str]) -> MarkdownDocument:
        return _chunked_doc(origin=item["id"], content=item["text"])

    ingestor = Ingestor(prepare=custom_prepare, num_workers=1)
    results = ingestor.run(
        [
            {"id": "doc1", "text": "First document content"},
            {"id": "doc2", "text": "Second document content"},
        ],
        store=store,
        progress=False,
    )

    assert results.inserted == 2
    assert results.failed == 0
    assert store.size() == 2


def test_ingest_aggregates_inserted_replaced_and_skipped():
    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_aggregate",
    )

    def custom_prepare(item: dict[str, str]) -> MarkdownDocument:
        return _chunked_doc(origin=item["id"], content=item["text"])

    results = ingest(
        [
            {"id": "doc1", "text": "First version"},
            {"id": "doc2", "text": "Stable document"},
            {"id": "doc1", "text": "Updated version"},
            {"id": "doc2", "text": "Stable document"},
        ],
        store=store,
        prepare=custom_prepare,
        num_workers=1,
        progress=False,
    )

    assert results.inserted == 2
    assert results.replaced == 1
    assert results.skipped == 1
    assert results.failed == 0
    assert store.size() == 2


def test_ingest_on_error_raise_fails_fast():
    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_raise",
    )

    def prepare(item: str) -> MarkdownDocument:
        if item == "bad":
            raise ValueError("boom")
        return _chunked_doc(origin=item, content=item)

    with pytest.raises(ItemError, match="Failed to ingest 'bad': boom") as exc_info:
        ingest(
            ["first", "bad", "later"],
            store=store,
            prepare=prepare,
            num_workers=1,
            progress=False,
        )

    assert exc_info.value.item == "bad"
    assert isinstance(exc_info.value.error, ValueError)
    assert exc_info.value.partial_results is not None
    assert exc_info.value.partial_results.inserted == 1
    assert exc_info.value.partial_results.failed == 1
    assert exc_info.value.partial_results.pending == 1
    assert store.size() == 1


def test_ingest_on_error_raise_drains_running_prepare_work():
    store = _RecordingStore()

    def prepare(item: str) -> MarkdownDocument:
        if item == "slow":
            time.sleep(1.0)
            return _chunked_doc(origin=item, content=item)
        if item == "bad":
            raise ValueError("boom")
        return _chunked_doc(origin=item, content=item)

    start = time.monotonic()
    with pytest.raises(ItemError, match="Failed to ingest 'bad': boom") as exc_info:
        ingest(
            ["slow", "bad"],
            store=store,
            prepare=prepare,
            num_workers=2,
            progress=False,
        )
    elapsed = time.monotonic() - start

    assert elapsed >= 1.0
    assert exc_info.value.partial_results is not None
    assert exc_info.value.partial_results.inserted == 1
    assert exc_info.value.partial_results.failed == 1
    assert exc_info.value.partial_results.cancelled == 0
    assert exc_info.value.partial_results.pending == 0
    assert store.size() == 1


def test_ingest_on_error_raise_waits_for_inflight_upserts():
    store = _SlowUpsertStore()

    def prepare(item: str) -> MarkdownDocument:
        if item == "bad":
            raise ValueError("boom")
        return _chunked_doc(origin=item, content=item)

    start = time.monotonic()
    with pytest.raises(ItemError, match="Failed to ingest 'bad': boom") as exc_info:
        ingest(
            ["slow", "bad"],
            store=store,
            prepare=prepare,
            num_workers=2,
            progress=False,
        )
    elapsed = time.monotonic() - start

    assert elapsed >= 1.0
    assert exc_info.value.partial_results is not None
    assert exc_info.value.partial_results.inserted == 1
    assert exc_info.value.partial_results.failed == 1
    assert exc_info.value.partial_results.cancelled == 0
    assert store.size() == 1

    time.sleep(0.2)

    assert store.size() == 1


def test_ingest_on_error_skip_collects_item_errors():
    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_skip",
    )

    def prepare(item: str) -> MarkdownDocument:
        if item == "bad":
            raise ValueError("boom")
        return _chunked_doc(origin=item, content=item)

    results = ingest(
        ["first", "bad", "later"],
        store=store,
        prepare=prepare,
        num_workers=1,
        on_error="skip",
        progress=False,
    )

    assert results.inserted == 2
    assert results.failed == 1
    assert len(results.errors) == 1
    assert results.errors[0].item == "bad"
    assert isinstance(results.errors[0].error, ValueError)
    assert [outcome.status for outcome in results.outcomes] == [
        "inserted",
        "failed",
        "inserted",
    ]
    assert store.size() == 2


def test_openai_store_default_prepare_uses_read_without_chunking(tmp_path):
    store, fake_vector_store_files = _make_fake_openai_store()
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nBody text\n", encoding="utf-8")

    results = ingest([str(path)], store=store, progress=False)

    assert results.inserted == 1
    assert results.failed == 0
    assert len(fake_vector_store_files.upload_calls) == 1
    uploaded_name, uploaded_bytes = fake_vector_store_files.upload_calls[0]["file"]
    assert uploaded_name == path.name
    assert b"Body text" in uploaded_bytes


@pytest.mark.parametrize(
    ("origin", "expected_name"),
    [
        ("https://example.com/docs/readme", "readme.md"),
        ("https://example.com/", "example.com.md"),
        (r"C:\docs\guide.txt", "guide.txt.md"),
        ("/", "document.md"),
    ],
)
def test_openai_store_normalizes_managed_filenames(origin: str, expected_name: str):
    store, fake_vector_store_files = _make_fake_openai_store()

    result = store.upsert(
        MarkdownDocument(origin=origin, content="# Title\n\nBody text\n")
    )

    assert result.action == "inserted"
    assert len(fake_vector_store_files.upload_calls) == 1
    uploaded_name, uploaded_bytes = fake_vector_store_files.upload_calls[0]["file"]
    assert uploaded_name == expected_name
    assert b"Body text" in uploaded_bytes


def test_chromadb_store_supports_shared_ingest_default_prepare(tmp_path):
    pytest.importorskip("chromadb")
    from tests.test_chroma_store import DummyEmbeddingFunction

    store = ChromaDBStore.create(
        location=":memory:",
        embed=DummyEmbeddingFunction(),
        overwrite=True,
        name="ingest_chroma_shared",
    )
    paths = _write_markdown_files(tmp_path, ["first", "second"])

    results = ingest(paths, store=store, progress=False)

    assert results.inserted == 2
    assert results.failed == 0
    assert store.size() == 2


def test_chonkie_compatibility_via_shared_ingest(tmp_path):
    chonkie = pytest.importorskip("chonkie")

    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_chonkie",
    )
    path = tmp_path / "doc.md"
    path.write_text(
        "# Test Document\n\nThis is test content for chonkie chunking.",
        encoding="utf-8",
    )
    chunker = getattr(chonkie, "TokenChunker")(chunk_size=50, chunk_overlap=10)

    def prepare(uri: str) -> MarkdownDocument:
        content = Path(uri).read_text(encoding="utf-8")
        chonkie_chunks = chunker.chunk(content)
        return MarkdownDocument(
            content=content,
            origin=uri,
            chunks=[Chunk.from_any(chunk) for chunk in chonkie_chunks],
        )

    results = ingest([str(path)], store=store, prepare=prepare, progress=False)

    assert results.inserted == 1
    assert store.size() == 1
