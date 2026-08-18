import inspect
from types import SimpleNamespace

import pytest

import raghilda.crawl as crawl_module
import raghilda.store as store_module
from raghilda.chunk import MarkdownChunk
from raghilda.crawl import (
    BaseCrawler,
    CloudflareCrawler,
    CrawlScope,
    DirectoryCrawler,
    FetchedSource,
    WebCrawler,
)
from raghilda.document import Document, MarkdownDocument
from raghilda.store import (
    ChromaDBStore,
    DuckDBStore,
    IngestSummary,
    OpenAIStore,
    WriteResult,
)


def test_document_uses_origin_field_not_id():
    doc = Document(content="hello")
    assert hasattr(doc, "origin")
    assert doc.origin is None
    assert not hasattr(doc, "id")


def test_store_api_uses_upsert_and_ingest_not_insert():
    assert hasattr(DuckDBStore, "upsert")
    assert hasattr(ChromaDBStore, "upsert")
    assert hasattr(OpenAIStore, "upsert")
    assert hasattr(DuckDBStore, "ingest")
    assert hasattr(ChromaDBStore, "ingest")
    assert hasattr(OpenAIStore, "ingest")
    assert not hasattr(DuckDBStore, "insert")
    assert not hasattr(ChromaDBStore, "insert")
    assert not hasattr(OpenAIStore, "insert")


def test_store_exports_write_and_ingest_results_not_insert_result():
    assert WriteResult is store_module.WriteResult
    assert IngestSummary is store_module.IngestSummary
    assert not hasattr(store_module, "InsertResult")


def test_store_exports_postgres_store_when_dependency_is_installed():
    pytest.importorskip("psycopg2")

    assert hasattr(store_module, "PostgreSQLStore")
    assert "PostgreSQLStore" in store_module.__all__


def test_crawl_exports_public_crawler_types():
    assert crawl_module.BaseCrawler is BaseCrawler
    assert crawl_module.CrawlScope is CrawlScope
    assert crawl_module.DirectoryCrawler is DirectoryCrawler
    assert crawl_module.WebCrawler is WebCrawler
    assert crawl_module.CloudflareCrawler is CloudflareCrawler
    assert crawl_module.FetchedSource is FetchedSource


def test_crawl_scope_owns_traversal_policy() -> None:
    assert tuple(inspect.signature(CrawlScope).parameters) == (
        "roots",
        "include_patterns",
        "exclude_patterns",
        "depth",
        "limit",
        "include_types",
        "exclude_types",
        "include_external_links",
        "include_subdomains",
    )


def test_crawler_constructors_keep_backend_and_cache_configuration_only() -> None:
    assert tuple(inspect.signature(DirectoryCrawler).parameters) == (
        "cache_dir",
        "max_workers",
    )
    assert tuple(inspect.signature(WebCrawler).parameters) == (
        "session",
        "cache_dir",
        "cache_stale_after",
        "max_workers",
    )
    assert tuple(inspect.signature(CloudflareCrawler).parameters) == (
        "account_id",
        "api_token",
        "cache_dir",
        "session",
        "source",
        "render",
        "cache_stale_after",
        "modified_since",
        "poll_interval",
        "max_poll_attempts",
        "max_workers",
        "base_url",
    )


def test_openai_upsert_rejects_chunked_document():
    class _SinglePage:
        def __init__(self):
            self.data = []

        def has_next_page(self):
            return False

    class FakeVectorStoreFiles:
        def list(self, **kwargs):
            return _SinglePage()

        def upload_and_poll(self, **kwargs):
            raise AssertionError("upload_and_poll should not be called")

        def delete(self, **kwargs):
            raise AssertionError("delete should not be called")

    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=FakeVectorStoreFiles()),
        files=SimpleNamespace(content=lambda **kwargs: None),
    )
    store = OpenAIStore(client=fake_client, store_id="vs_test")

    doc = MarkdownDocument(origin="doc", content="hello")
    doc = doc.to_chunked(
        [
            MarkdownChunk(
                text="hello",
                start_index=0,
                end_index=5,
                char_count=5,
            )
        ]
    )

    with pytest.raises(TypeError, match="does not support chunked documents"):
        store.upsert(doc)
