import inspect
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

import raghilda._ingest as ingest_impl
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


class _SignalingStore(_RecordingStore):
    def __init__(self) -> None:
        super().__init__()
        self.write_started = threading.Event()

    def upsert(
        self,
        document,
        *,
        skip_if_unchanged: bool = True,
    ) -> WriteResult:
        self.write_started.set()
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

    result = ingest(paths, store=store, progress=False)

    assert result.inserted == 2
    assert result.replaced == 0
    assert result.skipped == 0
    assert result.failed == 0
    assert store.size() == 2


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

    result = ingest(
        tracking_generator(),
        store=store,
        prepare=slow_prepare,
        num_workers=2,
        progress=False,
    )

    assert result.inserted == 20
    assert store.size() == 20
    assert max_pending <= 4


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
    result = ingestor.run(
        [
            {"id": "doc1", "text": "First document content"},
            {"id": "doc2", "text": "Second document content"},
        ],
        store=store,
        progress=False,
    )

    assert result.inserted == 2
    assert result.failed == 0
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

    result = ingest(
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

    assert result.inserted == 2
    assert result.replaced == 1
    assert result.skipped == 1
    assert result.failed == 0
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
    assert store.size() == 1


def test_ingest_on_error_raise_does_not_wait_for_running_workers():
    store = _RecordingStore()

    def prepare(item: str) -> MarkdownDocument:
        if item == "slow":
            time.sleep(1.0)
            return _chunked_doc(origin=item, content=item)
        if item == "bad":
            raise ValueError("boom")
        return _chunked_doc(origin=item, content=item)

    start = time.monotonic()
    with pytest.raises(ItemError, match="Failed to ingest 'bad': boom"):
        ingest(
            ["slow", "bad"],
            store=store,
            prepare=prepare,
            num_workers=2,
            progress=False,
        )
    elapsed = time.monotonic() - start

    assert elapsed < 0.5
    assert store.size() == 0

    time.sleep(1.0)

    assert store.size() == 0


def test_ingest_on_error_raise_waits_for_inflight_upserts():
    store = _SlowUpsertStore()

    def prepare(item: str) -> MarkdownDocument:
        if item == "bad":
            raise ValueError("boom")
        return _chunked_doc(origin=item, content=item)

    start = time.monotonic()
    with pytest.raises(ItemError, match="Failed to ingest 'bad': boom"):
        ingest(
            ["slow", "bad"],
            store=store,
            prepare=prepare,
            num_workers=2,
            progress=False,
        )
    elapsed = time.monotonic() - start

    assert elapsed >= 1.0
    assert store.size() == 1

    time.sleep(0.2)

    assert store.size() == 1


def test_ingest_on_error_raise_prevents_post_error_writes_from_registration_gap(
    monkeypatch,
):
    store = _SignalingStore()
    registration_waiting = threading.Event()
    release_registration = threading.Event()
    real_lock = threading.Lock

    class _BlockingRegistrationLock:
        def __init__(self) -> None:
            self._lock = real_lock()
            self._blocked_once = False

        def acquire(self, *args, **kwargs):
            return self._lock.acquire(*args, **kwargs)

        def release(self) -> None:
            self._lock.release()

        def locked(self) -> bool:
            return self._lock.locked()

        def __enter__(self):
            if not self._blocked_once:
                self._blocked_once = True
                registration_waiting.set()
                assert release_registration.wait(timeout=1.0)
            return self._lock.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            return self._lock.__exit__(exc_type, exc_value, traceback)

    def lock_factory():
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if (
            caller is not None
            and caller.f_code.co_name == "ingest"
            and caller.f_code.co_filename.endswith("_ingest.py")
        ):
            return _BlockingRegistrationLock()
        return real_lock()

    monkeypatch.setattr(ingest_impl.threading, "Lock", lock_factory)

    def prepare(item: str) -> MarkdownDocument:
        if item == "bad":
            assert registration_waiting.wait(timeout=1.0)
            raise ValueError("boom")
        return _chunked_doc(origin=item, content=item)

    with pytest.raises(ItemError, match="Failed to ingest 'bad': boom"):
        ingest(
            ["race", "bad"],
            store=store,
            prepare=prepare,
            num_workers=2,
            progress=False,
        )

    assert store.size() == 0
    assert not store.write_started.is_set()

    release_registration.set()

    assert not store.write_started.wait(timeout=0.2)
    assert store.size() == 0


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

    result = ingest(
        ["first", "bad", "later"],
        store=store,
        prepare=prepare,
        num_workers=1,
        on_error="skip",
        progress=False,
    )

    assert result.inserted == 2
    assert result.failed == 1
    assert len(result.errors) == 1
    assert result.errors[0].item == "bad"
    assert isinstance(result.errors[0].error, ValueError)
    assert store.size() == 2


def test_openai_store_default_prepare_uses_read_without_chunking(tmp_path):
    store, fake_vector_store_files = _make_fake_openai_store()
    path = tmp_path / "doc.md"
    path.write_text("# Title\n\nBody text\n", encoding="utf-8")

    result = ingest([str(path)], store=store, progress=False)

    assert result.inserted == 1
    assert result.failed == 0
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

    result = ingest(paths, store=store, progress=False)

    assert result.inserted == 2
    assert result.failed == 0
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

    result = ingest([str(path)], store=store, prepare=prepare, progress=False)

    assert result.inserted == 1
    assert store.size() == 1
