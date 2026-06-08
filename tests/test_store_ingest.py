from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import replace
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

import raghilda._store as store_module
from raghilda.chunker import MarkdownChunker
from raghilda.document import Document, MarkdownDocument
from raghilda.store import (
    BaseStore,
    ChromaDBStore,
    DuckDBStore,
    IngestSummary,
    OpenAIStore,
    WriteResult,
)


class _RecordingStore(BaseStore):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.prepare_thread_ids: dict[str, int] = {}
        self.upsert_thread_ids: dict[str, int] = {}
        self.started_origins: list[str] = []
        self.max_in_flight = 0
        self.in_flight = 0

    @staticmethod
    def connect(*args, **kwargs) -> "_RecordingStore":
        return _RecordingStore()

    @staticmethod
    def create(*args, **kwargs) -> "_RecordingStore":
        return _RecordingStore()

    def upsert(
        self,
        document: Document,
        *,
        skip_if_unchanged: bool = True,
    ) -> WriteResult[Document]:
        origin = document.origin
        assert isinstance(origin, str)
        with self.lock:
            self.started_origins.append(origin)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(0.02 if origin == "doc-1" else 0)
            if origin == "doc-2":
                raise RuntimeError("boom")
            self.upsert_thread_ids[origin] = threading.get_ident()
            action = (
                document.attributes["action"] if document.attributes else "inserted"
            )
            return WriteResult(action=action, document=document)
        finally:
            with self.lock:
                self.in_flight -= 1

    def retrieve(self, text: str, top_k: int, *args, **kwargs):
        return []

    def size(self) -> int:
        return len(self.started_origins)


class _BlockingFailureStore(BaseStore):
    def __init__(self) -> None:
        self.blocking_started = threading.Event()
        self.release_blocked = threading.Event()
        self.blocking_finished = threading.Event()

    @staticmethod
    def connect(*args, **kwargs) -> "_BlockingFailureStore":
        return _BlockingFailureStore()

    @staticmethod
    def create(*args, **kwargs) -> "_BlockingFailureStore":
        return _BlockingFailureStore()

    def upsert(
        self,
        document: Document,
        *,
        skip_if_unchanged: bool = True,
    ) -> WriteResult[Document]:
        del skip_if_unchanged
        origin = document.origin
        assert isinstance(origin, str)
        if origin == "doc-1":
            self.blocking_started.set()
            self.release_blocked.wait(timeout=1.0)
            self.blocking_finished.set()
            return WriteResult(action="inserted", document=document)
        if origin == "doc-2":
            assert self.blocking_started.wait(timeout=1.0)
            raise RuntimeError("boom")
        return WriteResult(action="inserted", document=document)

    def retrieve(self, text: str, top_k: int, *args, **kwargs):
        return []

    def size(self) -> int:
        return 0


class _CancelledSiblingStore(BaseStore):
    def __init__(self) -> None:
        self.allow_failure = threading.Event()
        self.release_cancelled = threading.Event()

    @staticmethod
    def connect(*args, **kwargs) -> "_CancelledSiblingStore":
        return _CancelledSiblingStore()

    @staticmethod
    def create(*args, **kwargs) -> "_CancelledSiblingStore":
        return _CancelledSiblingStore()

    def upsert(
        self,
        document: Document,
        *,
        skip_if_unchanged: bool = True,
    ) -> WriteResult[Document]:
        del skip_if_unchanged
        assert isinstance(document.origin, str)
        if document.origin == "doc-1":
            self.allow_failure.set()
            raise RuntimeError("boom")
        return WriteResult(action="inserted", document=document)

    def retrieve(self, text: str, top_k: int, *args, **kwargs):
        return []

    def size(self) -> int:
        return 0


def test_base_store_ingest_returns_summary_and_applies_prepare_before_upsert() -> None:
    store = _RecordingStore()
    main_thread_id = threading.get_ident()
    documents = [
        MarkdownDocument(
            origin="doc-1", content="# One", attributes={"action": "inserted"}
        ),
        MarkdownDocument(
            origin="doc-3", content="# Three", attributes={"action": "skipped"}
        ),
    ]

    def prepare(document: MarkdownDocument) -> MarkdownDocument:
        assert document.origin is not None
        store.prepare_thread_ids[document.origin] = threading.get_ident()
        return replace(document, content=document.content + "\nprepared")

    summary = store.ingest(documents, prepare=prepare, max_workers=2)

    assert summary == IngestSummary(inserted=1, replaced=0, skipped=1)
    assert set(store.prepare_thread_ids) == {"doc-1", "doc-3"}
    assert set(store.upsert_thread_ids) == {"doc-1", "doc-3"}
    assert set(store.prepare_thread_ids.values()).isdisjoint({main_thread_id})


def test_base_store_ingest_runs_prepare_in_worker_pool_concurrently() -> None:
    store = _RecordingStore()
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    in_prepare = 0
    max_in_prepare = 0

    documents = [
        MarkdownDocument(origin="doc-1", content="# One"),
        MarkdownDocument(origin="doc-3", content="# Three"),
    ]

    def prepare(document: MarkdownDocument) -> MarkdownDocument:
        nonlocal in_prepare, max_in_prepare
        with lock:
            in_prepare += 1
            max_in_prepare = max(max_in_prepare, in_prepare)
        try:
            barrier.wait(timeout=1.0)
            return document
        finally:
            with lock:
                in_prepare -= 1

    summary = store.ingest(documents, prepare=prepare, max_workers=2)

    assert summary == IngestSummary(inserted=2, replaced=0, skipped=0)
    assert max_in_prepare == 2


def test_base_store_ingest_starts_writes_before_input_is_exhausted() -> None:
    class _StreamingStore(BaseStore):
        def __init__(self) -> None:
            self.started = threading.Event()
            self.started_origins: list[str] = []

        @staticmethod
        def connect(*args, **kwargs) -> "_StreamingStore":
            return _StreamingStore()

        @staticmethod
        def create(*args, **kwargs) -> "_StreamingStore":
            return _StreamingStore()

        def upsert(
            self,
            document: Document,
            *,
            skip_if_unchanged: bool = True,
        ) -> WriteResult[Document]:
            del skip_if_unchanged
            assert isinstance(document.origin, str)
            self.started_origins.append(document.origin)
            self.started.set()
            return WriteResult(action="inserted", document=document)

        def retrieve(self, text: str, top_k: int, *args, **kwargs):
            return []

        def size(self) -> int:
            return len(self.started_origins)

    store = _StreamingStore()

    def documents():
        yield MarkdownDocument(origin="doc-1", content="# One")
        assert store.started.wait(timeout=1.0)
        yield MarkdownDocument(origin="doc-2", content="# Two")

    summary = store.ingest(documents(), max_workers=1)

    assert summary == IngestSummary(inserted=2, replaced=0, skipped=0)
    assert store.started_origins == ["doc-1", "doc-2"]


def test_base_store_ingest_raises_on_duplicate_after_streaming_started() -> None:
    store = _RecordingStore()
    documents = [
        MarkdownDocument(origin="dup", content="# One"),
        MarkdownDocument(origin="dup", content="# Two"),
        MarkdownDocument(origin="doc-3", content="# Three"),
    ]

    with pytest.raises(ValueError, match="Duplicate origin during ingest: dup"):
        store.ingest(documents, max_workers=1)

    assert store.started_origins == ["dup"]


def test_base_store_ingest_fails_fast_and_bounds_worker_count() -> None:
    store = _RecordingStore()
    documents = [
        MarkdownDocument(origin="doc-1", content="# One"),
        MarkdownDocument(origin="doc-2", content="# Two"),
        MarkdownDocument(origin="doc-3", content="# Three"),
    ]

    with pytest.raises(RuntimeError, match="boom"):
        store.ingest(documents, max_workers=2)

    assert "doc-3" not in store.started_origins
    assert store.max_in_flight <= 2


def test_base_store_ingest_waits_for_running_workers_before_raising() -> None:
    store = _BlockingFailureStore()
    documents = [
        MarkdownDocument(origin="doc-1", content="# One"),
        MarkdownDocument(origin="doc-2", content="# Two"),
    ]

    def release_blocked() -> None:
        assert store.blocking_started.wait(timeout=1.0)
        time.sleep(0.2)
        store.release_blocked.set()

    releaser = threading.Thread(target=release_blocked)
    releaser.start()

    try:
        with pytest.raises(RuntimeError, match="boom"):
            store.ingest(documents, max_workers=2)

        assert store.release_blocked.is_set()
        assert store.blocking_finished.is_set()
    finally:
        releaser.join()


def test_base_store_ingest_ignores_cancelled_sibling_when_worker_failed(
    monkeypatch,
) -> None:
    class _FakeFuture:
        def __init__(
            self,
            *,
            result: WriteResult[Document] | None = None,
            error: BaseException | None = None,
        ) -> None:
            self._result = result
            self._error = error

        def result(self) -> WriteResult[Document]:
            if self._error is not None:
                raise self._error
            assert self._result is not None
            return self._result

        def cancel(self) -> None:
            return None

    class _FakeExecutor:
        def __init__(self, *, max_workers: int) -> None:
            del max_workers
            self._submissions = [
                _FakeFuture(error=CancelledError()),
                _FakeFuture(error=RuntimeError("boom")),
            ]

        def submit(self, fn, arg):
            del fn, arg
            return self._submissions.pop(0)

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            del wait, cancel_futures
            return None

    def fake_wait(pending, return_when):
        del pending, return_when
        return (
            [
                _FakeFuture(error=CancelledError()),
                _FakeFuture(error=RuntimeError("boom")),
            ],
            set(),
        )

    monkeypatch.setattr(store_module, "ThreadPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(store_module, "wait", fake_wait)

    store = _RecordingStore()
    documents = [
        MarkdownDocument(origin="doc-1", content="# One"),
        MarkdownDocument(origin="doc-2", content="# Two"),
    ]

    with pytest.raises(RuntimeError, match="boom"):
        store.ingest(documents, max_workers=2)


def test_base_store_ingest_propagates_worker_cancelled_error() -> None:
    store = _RecordingStore()
    documents = [MarkdownDocument(origin="doc-1", content="# One")]

    def prepare(document: MarkdownDocument) -> MarkdownDocument:
        del document
        raise CancelledError("prepare cancelled")

    with pytest.raises(CancelledError, match="prepare cancelled"):
        store.ingest(documents, prepare=prepare, max_workers=1)


def test_postgresql_store_ingest_serializes_upsert_calls() -> None:
    pytest.importorskip("psycopg2")
    from raghilda._postgres_store import PostgreSQLStore

    store = PostgreSQLStore.__new__(PostgreSQLStore)
    store._ingest_upsert_lock = threading.Lock()
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def upsert(
        document: Document,
        *,
        skip_if_unchanged: bool = True,
    ) -> WriteResult[Document]:
        del skip_if_unchanged
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        try:
            time.sleep(0.02)
            return WriteResult(action="inserted", document=document)
        finally:
            with lock:
                in_flight -= 1

    store.upsert = upsert  # type: ignore[method-assign]
    documents = [
        MarkdownDocument(origin="doc-1", content="# One"),
        MarkdownDocument(origin="doc-2", content="# Two"),
    ]

    summary = store.ingest(documents, max_workers=2)

    assert summary == IngestSummary(inserted=2, replaced=0, skipped=0)
    assert max_in_flight == 1


def test_duckdb_store_ingest_prepares_chunked_documents() -> None:
    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="duckdb_ingest",
    )
    documents = [
        MarkdownDocument(origin="doc-1", content="# One\n\nHello"),
        MarkdownDocument(origin="doc-2", content="# Two\n\nWorld"),
    ]

    summary = store.ingest(
        documents,
        prepare=MarkdownChunker(chunk_size=32, target_overlap=0).chunk,
        max_workers=2,
    )

    assert summary == IngestSummary(inserted=2, replaced=0, skipped=0)
    assert store.size() == 2


def test_chromadb_store_ingest_prepares_chunked_documents(tmp_path: Path) -> None:
    store = ChromaDBStore.create(
        location=tmp_path / "chroma",
        overwrite=True,
        name="chroma_ingest",
        embed=None,
    )
    documents = [
        MarkdownDocument(origin="doc-1", content="# One\n\nHello"),
        MarkdownDocument(origin="doc-2", content="# Two\n\nWorld"),
    ]

    summary = store.ingest(
        documents,
        prepare=MarkdownChunker(chunk_size=32, target_overlap=0).chunk,
        max_workers=2,
    )

    assert summary == IngestSummary(inserted=2, replaced=0, skipped=0)
    assert store.size() == 2


class _SinglePage:
    def __init__(self, data: list[Any]):
        self.data = data

    def has_next_page(self) -> bool:
        return False


class _FakeVectorStoreFiles:
    def __init__(self) -> None:
        self.uploads: list[dict[str, Any]] = []

    def list(self, **kwargs):
        return _SinglePage([])

    def upload_and_poll(self, **kwargs):
        self.uploads.append(kwargs)
        return SimpleNamespace(id=f"file-{len(self.uploads)}")

    def delete(self, **kwargs):
        raise AssertionError("delete should not be called")


def test_openai_store_ingest_accepts_markdown_documents_without_prepare() -> None:
    vector_store_files = _FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=vector_store_files),
    )
    store = OpenAIStore(client=fake_client, store_id="vs_test")
    documents = [
        MarkdownDocument(origin="doc-1", content="# One"),
        MarkdownDocument(origin="doc-2", content="# Two"),
    ]

    summary = store.ingest(documents, max_workers=2)

    assert summary == IngestSummary(inserted=2, replaced=0, skipped=0)
    assert len(vector_store_files.uploads) == 2
