from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fnmatch
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import socketserver
import threading
from typing import Any
import unicodedata

import pytest
import raghilda.crawl as crawl_module
from raghilda.crawl import (
    CrawlScope,
    CloudflareCrawler,
    DirectoryCrawler,
    FetchedSource,
    WebCrawler,
)
from raghilda.document import MarkdownDocument

_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def _write(tmp_path: Path, relative: str, contents: str) -> Path:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def _expected_cache_base(key: str) -> str:
    value = unicodedata.normalize("NFC", key)
    value = value.replace("://", "__")
    value = value.replace("\\", "_")
    value = value.replace("/", "_")
    value = re.sub(r'[\x00-\x1f<>:"|?*]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip(" ._")

    if not value:
        value = "entry"

    root = value.split(".", 1)[0].rstrip(" .").upper()
    if root in _WINDOWS_RESERVED:
        value = f"_{value}"

    if len(value) > 180:
        head = 180 // 2 - 2
        tail = 180 - head - 2
        value = f"{value[:head]}..{value[-tail:]}"

    value = value.rstrip(" .")
    stem = value or "entry"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"{stem}--{digest}"


def test_directory_crawler_discovers_and_converts_markdown_documents(
    tmp_path: Path,
) -> None:
    markdown = _write(tmp_path, "docs/readme.md", "# Hello\n\nDirectory crawler")
    _write(tmp_path, "docs/skip.py", "print('skip')")
    notebook = _write(
        tmp_path,
        "docs/notebook.ipynb",
        json.dumps(
            {
                "cells": [],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
    )

    crawler = DirectoryCrawler()
    scope = CrawlScope(
        roots=[tmp_path],
        depth=3,
        include_patterns=[r".*/docs/.*"],
        exclude_patterns=[r".*/skip\.py$"],
        include_types=["markdown", "jupyter-notebook"],
    )

    origins = list(crawler.origins(scope, progress=False))

    assert markdown.resolve().as_uri() in origins
    assert notebook.resolve().as_uri() in origins
    assert all(not origin.endswith("skip.py") for origin in origins)

    source = crawler.fetch_raw(markdown.resolve().as_uri())
    assert isinstance(source, FetchedSource)
    assert source.origin == markdown.resolve().as_uri()
    assert source.body_path == markdown.resolve()
    assert source.status_code is None

    fetched_markdown = crawler.fetch_markdown(markdown.resolve().as_uri())
    assert fetched_markdown == MarkdownDocument(
        origin=markdown.resolve().as_uri(),
        content="# Hello\n\nDirectory crawler",
    )


def test_directory_crawler_convert_override_receives_fetched_source(
    tmp_path: Path,
) -> None:
    markdown = _write(tmp_path, "docs/readme.md", "# Hello\n\nDirectory crawler")
    seen: list[FetchedSource] = []

    crawler = DirectoryCrawler()

    converted = crawler.fetch_markdown(
        markdown.resolve().as_uri(),
        convert=lambda source: _record_directory_conversion(source, seen),
    )

    assert [item.origin for item in seen] == [markdown.resolve().as_uri()]
    assert converted == MarkdownDocument(
        origin=markdown.resolve().as_uri(),
        content="# Converted\n",
    )


def test_directory_crawler_cache_dir_uses_hashed_file_pair(
    tmp_path: Path,
) -> None:
    markdown = _write(tmp_path, "docs/readme.md", "# Hello\n")
    cache_dir = tmp_path / "cache"
    crawler = DirectoryCrawler(cache_dir=cache_dir)

    origin = markdown.resolve().as_uri()
    document = crawler.fetch_markdown(origin)

    base = _expected_cache_base(origin)
    metadata_path = cache_dir / f"{base}.metadata.json"
    content_path = cache_dir / f"{base}.md"
    assert document == MarkdownDocument(origin=origin, content="# Hello\n")
    assert sorted(path.name for path in cache_dir.iterdir()) == [
        content_path.name,
        metadata_path.name,
    ]
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == {
        "content_path": content_path.name,
        "key": origin,
        "metadata": {
            "mtime_ns": markdown.stat().st_mtime_ns,
            "origin": origin,
            "source_hash": hashlib.sha256(markdown.read_bytes()).hexdigest(),
        },
    }


def test_directory_crawler_cache_dir_true_uses_default_backend_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    markdown = _write(tmp_path, "docs/readme.md", "# Hello\n")
    monkeypatch.chdir(tmp_path)
    crawler = DirectoryCrawler(cache_dir=True)

    origin = markdown.resolve().as_uri()
    crawler.fetch_markdown(origin)

    cache_dir = tmp_path / ".raghilda" / "cache" / "directory"
    base = _expected_cache_base(origin)
    assert sorted(path.name for path in cache_dir.iterdir()) == [
        f"{base}.md",
        f"{base}.metadata.json",
    ]


def _record_directory_conversion(
    source: FetchedSource, seen: list[FetchedSource]
) -> MarkdownDocument:
    seen.append(source)
    return MarkdownDocument(origin=source.origin, content="# Converted\n")


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class _RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        routes = self.server.routes  # type: ignore[attr-defined]
        route = routes[path]
        self.server.requests.append(  # type: ignore[attr-defined]
            {"path": path, "headers": dict(self.headers.items())}
        )
        if route["etag"] and self.headers.get("If-None-Match") == route["etag"]:
            self.send_response(304)
            self.send_header("ETag", route["etag"])
            self.end_headers()
            return

        body = route["body"].encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", route["content_type"])
        self.send_header("Content-Length", str(len(body)))
        if route["etag"]:
            self.send_header("ETag", route["etag"])
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


class _FakeWebResponse:
    def __init__(
        self,
        *,
        body: str,
        url: str,
        content_type: str = "text/html; charset=utf-8",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.content = body.encode("utf-8")
        self.headers = {"Content-Type": content_type, **(headers or {})}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        assert self.status_code < 400


class _FakeWebSession:
    def __init__(self, routes: dict[str, dict[str, Any]]) -> None:
        self.routes = routes
        self.requests: list[tuple[str, dict[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeWebResponse:
        del timeout
        self.requests.append((url, headers))
        route = self.routes[url]
        return _FakeWebResponse(
            body=route["body"],
            url=route.get("resolved_url", url),
            content_type=route.get("content_type", "text/html; charset=utf-8"),
            status_code=route.get("status_code", 200),
            headers=route.get("headers"),
        )


@contextmanager
def _serve(routes: dict[str, dict[str, str | None]]):
    server = _ThreadingHTTPServer(("127.0.0.1", 0), _RequestHandler)
    server.routes = routes  # type: ignore[attr-defined]
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_web_crawler_discovers_origins_and_revalidates_cache(tmp_path: Path) -> None:
    with _serve(
        {
            "/": {
                "body": """
                <html><body>
                  <main>
                    <a href="/guide">Guide</a>
                    <a href="/skip">Skip</a>
                    <a href="http://external.test/path">External</a>
                  </main>
                </body></html>
                """,
                "content_type": "text/html; charset=utf-8",
                "etag": "root-v1",
            },
            "/guide": {
                "body": "<html><body><main><h1>Guide</h1><p>Hello</p></main></body></html>",
                "content_type": "text/html; charset=utf-8",
                "etag": "guide-v1",
            },
            "/skip": {
                "body": "<html><body><main><h1>Skip</h1></main></body></html>",
                "content_type": "text/html; charset=utf-8",
                "etag": "skip-v1",
            },
        }
    ) as server:
        root_url = f"http://127.0.0.1:{server.server_port}/"
        crawler = WebCrawler(
            cache_dir=tmp_path / "cache",
            cache_stale_after=timedelta(seconds=0),
        )
        scope = CrawlScope(
            roots=[root_url],
            depth=1,
            include_patterns=[rf"^{re.escape(root_url)}.*"],
            exclude_patterns=[r".*/skip$"],
        )

        origins = list(crawler.origins(scope, progress=False))

        assert root_url in origins
        assert f"{root_url}guide" in origins
        assert all(not origin.endswith("/skip") for origin in origins)
        assert all("external.test" not in origin for origin in origins)

        first = crawler.fetch_raw(root_url)
        second = crawler.fetch_raw(root_url)
        third = crawler.fetch_raw(root_url, cache_force_refresh=True)
        server_requests = getattr(server, "requests")
        root_requests = [
            request for request in server_requests if request["path"] == "/"
        ]

        assert first.body_path == second.body_path == third.body_path
        assert second.revalidated_at is not None
        assert root_requests[-2]["headers"]["If-None-Match"] == "root-v1"
        assert "If-None-Match" not in root_requests[-1]["headers"]

        guide_doc = crawler.fetch_markdown(f"{root_url}guide")
        assert guide_doc.origin == f"{root_url}guide"
        assert "Guide" in guide_doc.content


def test_web_crawler_resolves_relative_links_from_redirect_target(
    tmp_path: Path,
) -> None:
    session: Any = _FakeWebSession(
        {
            "https://example.com/docs": {
                "body": '<html><body><a href="page">Page</a></body></html>',
                "resolved_url": "https://example.com/docs/",
            },
            "https://example.com/docs/page": {
                "body": "<html><body><main>Page</main></body></html>",
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "redirect-cache",
        session=session,
    )
    scope = CrawlScope(roots=["https://example.com/docs"], depth=1)

    origins = list(crawler.origins(scope, progress=False))

    assert "https://example.com/docs" in origins
    assert "https://example.com/docs/page" in origins
    assert "https://example.com/page" not in origins


def test_web_crawler_follows_links_after_redirect_to_different_host(
    tmp_path: Path,
) -> None:
    session: Any = _FakeWebSession(
        {
            "https://example.com": {
                "body": '<html><body><a href="/about">About</a></body></html>',
                "resolved_url": "https://www.example.com/landing",
            },
            "https://www.example.com/about": {
                "body": "<html><body><main>About</main></body></html>",
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "redirect-host-cache",
        session=session,
    )
    scope = CrawlScope(roots=["https://example.com"], depth=1)

    origins = list(crawler.origins(scope, progress=False))

    assert "https://example.com" in origins
    assert "https://www.example.com/about" in origins


def test_web_crawler_include_subdomains_stays_within_requested_host_tree(
    tmp_path: Path,
) -> None:
    root = "https://docs.example.co.uk/start"
    allowed = "https://api.docs.example.co.uk/page"
    disallowed_parent = "https://example.co.uk/root"
    disallowed_sibling = "https://other.co.uk/page"
    session: Any = _FakeWebSession(
        {
            root: {
                "body": (
                    f'<html><body><a href="{allowed}">Allowed</a>'
                    f'<a href="{disallowed_parent}">Parent</a>'
                    f'<a href="{disallowed_sibling}">Sibling</a></body></html>'
                ),
            },
            allowed: {"body": "<html><body><main>Allowed</main></body></html>"},
            disallowed_parent: {
                "body": "<html><body><main>Parent</main></body></html>"
            },
            disallowed_sibling: {
                "body": "<html><body><main>Sibling</main></body></html>"
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "subdomain-cache",
        session=session,
    )
    scope = CrawlScope(
        roots=[root],
        depth=1,
        include_subdomains=True,
    )

    origins = list(crawler.origins(scope, progress=False))

    assert root in origins
    assert allowed in origins
    assert disallowed_parent not in origins
    assert disallowed_sibling not in origins


def test_web_crawler_discovers_matching_descendants_from_filtered_seed(
    tmp_path: Path,
) -> None:
    with _serve(
        {
            "/": {
                "body": '<html><body><a href="/docs/guide">Guide</a></body></html>',
                "content_type": "text/html; charset=utf-8",
                "etag": None,
            },
            "/docs/guide": {
                "body": "<html><body><main>Guide</main></body></html>",
                "content_type": "text/html; charset=utf-8",
                "etag": None,
            },
        }
    ) as server:
        root_url = f"http://127.0.0.1:{server.server_port}/"
        crawler = WebCrawler(
            cache_dir=tmp_path / "filtered-seed-cache",
        )
        scope = CrawlScope(
            roots=[root_url],
            depth=1,
            include_patterns=[rf"^{re.escape(root_url)}docs/.*"],
        )

        origins = list(crawler.origins(scope, progress=False))

        assert root_url not in origins
        assert f"{root_url}docs/guide" in origins


def test_web_crawler_accepts_crawl_scope_for_roots_and_patterns(
    tmp_path: Path,
) -> None:
    with _serve(
        {
            "/": {
                "body": '<html><body><a href="/docs/guide">Guide</a></body></html>',
                "content_type": "text/html; charset=utf-8",
                "etag": None,
            },
            "/docs/guide": {
                "body": "<html><body><main>Guide</main></body></html>",
                "content_type": "text/html; charset=utf-8",
                "etag": None,
            },
        }
    ) as server:
        root_url = f"http://127.0.0.1:{server.server_port}/"
        crawler = WebCrawler(cache_dir=tmp_path / "scope-cache")
        scope = CrawlScope(
            roots=[root_url],
            depth=1,
            include_patterns=[rf"^{re.escape(root_url)}docs/.*"],
        )

        origins = list(crawler.origins(scope, progress=False))
        documents = list(crawler.markdown_documents(scope, progress=False))

        assert origins == [f"{root_url}docs/guide"]
        assert documents == [
            MarkdownDocument(origin=f"{root_url}docs/guide", content="Guide")
        ]


def test_web_markdown_documents_reuses_refreshed_sources(
    tmp_path: Path,
) -> None:
    with _serve(
        {
            "/": {
                "body": "<html><body><main>Root</main></body></html>",
                "content_type": "text/html; charset=utf-8",
                "etag": None,
            }
        }
    ) as server:
        root_url = f"http://127.0.0.1:{server.server_port}/"
        crawler = WebCrawler(
            cache_dir=tmp_path / "markdown-docs-cache",
        )
        scope = CrawlScope(roots=[root_url], depth=0)

        documents = list(crawler.markdown_documents(scope, cache_force_refresh=True))
        root_requests = [
            request for request in getattr(server, "requests") if request["path"] == "/"
        ]

        assert documents == [MarkdownDocument(origin=root_url, content="Root")]
        assert len(root_requests) == 1


def test_web_crawler_fetches_same_depth_frontier_concurrently(tmp_path: Path) -> None:
    root = "https://example.com/docs"
    first = "https://example.com/docs/one"
    second = "https://example.com/docs/two"

    class _ConcurrentWebSession:
        def __init__(self) -> None:
            self.requests: list[tuple[str, dict[str, str]]] = []
            self._lock = threading.Lock()
            self._barrier = threading.Barrier(2)
            self.in_flight = 0
            self.max_in_flight = 0

        def get(
            self, url: str, headers: dict[str, str], timeout: float
        ) -> _FakeWebResponse:
            del timeout
            with self._lock:
                self.requests.append((url, headers))
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                if url == root:
                    return _FakeWebResponse(
                        body=(
                            f'<html><body><a href="{first}">One</a>'
                            f'<a href="{second}">Two</a></body></html>'
                        ),
                        url=url,
                    )
                if url in {first, second}:
                    self._barrier.wait(timeout=1.0)
                    return _FakeWebResponse(
                        body="<html><body><main>Child</main></body></html>",
                        url=url,
                    )
                raise AssertionError(f"Unexpected url: {url}")
            finally:
                with self._lock:
                    self.in_flight -= 1

    session: Any = _ConcurrentWebSession()
    crawler = WebCrawler(
        cache_dir=tmp_path / "frontier-cache",
        session=session,
        max_workers=2,
    )
    scope = CrawlScope(roots=[root], depth=1)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [root, first, second]
    assert session.max_in_flight == 2


def test_web_crawler_treats_304_revalidation_as_fresh_cache_hit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    with _serve(
        {
            "/": {
                "body": "<html><body><main>Cached</main></body></html>",
                "content_type": "text/html; charset=utf-8",
                "etag": "root-v1",
            }
        }
    ) as server:
        root_url = f"http://127.0.0.1:{server.server_port}/"
        times = iter(
            [
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 2, 500000, tzinfo=timezone.utc),
            ]
        )
        monkeypatch.setattr(crawl_module, "_utcnow", lambda: next(times))
        crawler = WebCrawler(
            cache_dir=tmp_path / "fresh-cache",
            cache_stale_after=timedelta(seconds=1),
        )

        first = crawler.fetch_raw(root_url)
        second = crawler.fetch_raw(root_url)
        third = crawler.fetch_raw(root_url)
        root_requests = [
            request for request in getattr(server, "requests") if request["path"] == "/"
        ]

        assert first.body_path == second.body_path == third.body_path
        assert second.revalidated_at is not None
        assert len(root_requests) == 2
        assert root_requests[1]["headers"]["If-None-Match"] == "root-v1"


def test_web_crawler_cache_dir_uses_hashed_file_pair(
    tmp_path: Path,
) -> None:
    with _serve(
        {
            "/": {
                "body": "<html><body><main>Root</main></body></html>",
                "content_type": "text/html; charset=utf-8",
                "etag": None,
            }
        }
    ) as server:
        root_url = f"http://127.0.0.1:{server.server_port}/"
        cache_dir = tmp_path / "cache"
        crawler = WebCrawler(cache_dir=cache_dir)

        document = crawler.fetch_markdown(root_url)

        base = _expected_cache_base(root_url)
        metadata_path = cache_dir / f"{base}.metadata.json"
        content_path = cache_dir / f"{base}.html"
        assert document == MarkdownDocument(origin=root_url, content="Root")
        assert sorted(path.name for path in cache_dir.iterdir()) == [
            content_path.name,
            metadata_path.name,
        ]
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert record["key"] == root_url
        assert record["content_path"] == content_path.name
        assert record["metadata"]["content_type"] == "text/html; charset=utf-8"
        assert record["metadata"]["origin"] == root_url


def test_web_crawler_cache_dir_true_uses_default_backend_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with _serve(
        {
            "/": {
                "body": "<html><body><main>Root</main></body></html>",
                "content_type": "text/html; charset=utf-8",
                "etag": None,
            }
        }
    ) as server:
        root_url = f"http://127.0.0.1:{server.server_port}/"
        crawler = WebCrawler(cache_dir=True)

        crawler.fetch_markdown(root_url)

    cache_dir = tmp_path / ".raghilda" / "cache" / "web"
    base = _expected_cache_base(root_url)
    assert sorted(path.name for path in cache_dir.iterdir()) == [
        f"{base}.html",
        f"{base}.metadata.json",
    ]


def test_web_crawler_disambiguates_colliding_sanitized_cache_prefixes(
    tmp_path: Path,
) -> None:
    first_origin = "https://example.com/docs/page"
    second_origin = "https://example.com/docs:page"
    third_origin = "https://example.com/docs?page"
    session: Any = _FakeWebSession(
        {
            first_origin: {"body": "<html><body><main>One</main></body></html>"},
            second_origin: {"body": "<html><body><main>Two</main></body></html>"},
            third_origin: {"body": "<html><body><main>Three</main></body></html>"},
        }
    )
    cache_dir = tmp_path / "collision-cache"
    crawler = WebCrawler(cache_dir=cache_dir, session=session)

    crawler.fetch_raw(first_origin)
    crawler.fetch_raw(second_origin)
    crawler.fetch_raw(third_origin)

    first_base = _expected_cache_base(first_origin)
    second_base = _expected_cache_base(second_origin)
    third_base = _expected_cache_base(third_origin)
    cached_names = {path.name for path in cache_dir.iterdir()}
    assert {
        f"{first_base}.html",
        f"{first_base}.metadata.json",
        f"{second_base}.html",
        f"{second_base}.metadata.json",
        f"{third_base}.html",
        f"{third_base}.metadata.json",
    }.issubset(cached_names)
    assert len(cached_names) == 6

    second_session: Any = _FakeWebSession(
        {
            first_origin: {"body": "<html><body><main>One</main></body></html>"},
            second_origin: {"body": "<html><body><main>Two</main></body></html>"},
            third_origin: {"body": "<html><body><main>Three</main></body></html>"},
        }
    )
    second_crawler = WebCrawler(cache_dir=cache_dir, session=second_session)

    assert second_crawler.fetch_raw(first_origin).body_path.exists()
    assert second_crawler.fetch_raw(second_origin).body_path.exists()
    assert second_crawler.fetch_raw(third_origin).body_path.exists()
    assert second_session.requests == []


def test_web_crawler_cache_writes_for_different_keys_do_not_contend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_origin = "https://example.com/docs/one"
    second_origin = "https://example.com/docs/two"
    session: Any = _FakeWebSession(
        {
            first_origin: {"body": "<html><body><main>One</main></body></html>"},
            second_origin: {"body": "<html><body><main>Two</main></body></html>"},
        }
    )
    cache_dir = tmp_path / "concurrency-cache"
    crawler = WebCrawler(cache_dir=cache_dir, session=session)

    first_content_path = cache_dir / f"{_expected_cache_base(first_origin)}.html"
    second_content_path = cache_dir / f"{_expected_cache_base(second_origin)}.html"
    first_started = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    errors: list[BaseException] = []
    original_write_content = crawl_module._FilesystemCrawlerCache._write_content

    def blocking_write_content(
        self,
        path: Path,
        content: bytes | str | Path,
    ) -> None:
        if path == first_content_path and not first_started.is_set():
            first_started.set()
            assert release_first.wait(timeout=2.0)
        original_write_content(self, path, content)
        if path == second_content_path:
            second_finished.set()

    monkeypatch.setattr(
        crawl_module._FilesystemCrawlerCache,
        "_write_content",
        blocking_write_content,
    )

    def fetch(origin: str) -> None:
        try:
            crawler.fetch_raw(origin)
        except BaseException as exc:
            errors.append(exc)

    first_thread = threading.Thread(target=fetch, args=(first_origin,))
    second_thread = threading.Thread(target=fetch, args=(second_origin,))
    first_thread.start()
    assert first_started.wait(timeout=1.0)
    second_thread.start()
    try:
        assert second_finished.wait(timeout=1.0)
    finally:
        release_first.set()
        first_thread.join(timeout=1.0)
        second_thread.join(timeout=1.0)

    assert errors == []


def test_web_crawler_uses_magika_when_no_explicit_ext_is_available(
    tmp_path: Path,
) -> None:
    origin = "https://example.com/download"
    session: Any = _FakeWebSession(
        {
            origin: {
                "body": "<html><body><main>Download</main></body></html>",
                "content_type": "application/octet-stream",
            }
        }
    )
    cache_dir = tmp_path / "magika-cache"
    crawler = WebCrawler(cache_dir=cache_dir, session=session)

    source = crawler.fetch_raw(origin)

    base = _expected_cache_base(origin)
    assert source.body_path == cache_dir / f"{base}.html"


def test_web_crawler_falls_back_to_raw_when_magika_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    origin = "https://example.com/download"
    session: Any = _FakeWebSession(
        {
            origin: {
                "body": "opaque payload",
                "content_type": "application/octet-stream",
            }
        }
    )
    cache_dir = tmp_path / "raw-cache"
    monkeypatch.setattr(crawl_module, "_MAGIKA", None)
    crawler = WebCrawler(cache_dir=cache_dir, session=session)

    source = crawler.fetch_raw(origin)

    base = _expected_cache_base(origin)
    assert source.body_path == cache_dir / f"{base}.raw"


class _CloudflareResponse:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.status_code = 200

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        return


class _CloudflareSession:
    def __init__(self) -> None:
        self.post_calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        self.get_calls: list[tuple[str, dict[str, Any] | None]] = []
        self._poll_count = 0

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _CloudflareResponse:
        self.post_calls.append((url, json, headers))
        return _CloudflareResponse({"success": True, "result": "job-123"})

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        timeout: float,
    ) -> _CloudflareResponse:
        self.get_calls.append((url, params))
        self._poll_count += 1
        if self._poll_count == 1:
            return _CloudflareResponse(
                {"success": True, "result": {"id": "job-123", "status": "running"}}
            )
        if params == {"limit": 1}:
            return _CloudflareResponse(
                {"success": True, "result": {"id": "job-123", "status": "completed"}}
            )
        return _CloudflareResponse(
            {
                "success": True,
                "result": {
                    "id": "job-123",
                    "status": "completed",
                    "records": [
                        {
                            "url": "https://example.com/docs",
                            "status": "completed",
                            "markdown": "# Docs\n",
                            "metadata": {
                                "status": 200,
                                "title": "Docs",
                                "url": "https://example.com/docs",
                            },
                        },
                        {
                            "url": "https://example.com/docs/page",
                            "status": "completed",
                            "markdown": "## Page\n",
                            "metadata": {
                                "status": 200,
                                "title": "Page",
                                "url": "https://example.com/docs/page",
                            },
                        },
                    ],
                },
            }
        )


class _ParameterizedCloudflareSession:
    def __init__(self) -> None:
        self.post_calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        self.get_calls: list[tuple[str, dict[str, Any] | None]] = []
        self._jobs: dict[str, dict[str, Any]] = {}

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> _CloudflareResponse:
        del timeout
        job_id = f"job-{len(self.post_calls) + 1}"
        self.post_calls.append((url, json, headers))
        self._jobs[job_id] = json
        return _CloudflareResponse({"success": True, "result": job_id})

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        timeout: float,
    ) -> _CloudflareResponse:
        del headers, timeout
        self.get_calls.append((url, params))
        job_id = url.rsplit("/", 1)[-1]
        payload = self._jobs[job_id]
        if params == {"limit": 1}:
            return _CloudflareResponse(
                {"success": True, "result": {"id": job_id, "status": "completed"}}
            )
        records = [
            {
                "url": payload["url"],
                "status": "completed",
                "markdown": "# Docs\n",
                "metadata": {
                    "status": 200,
                    "title": "Docs",
                    "url": payload["url"],
                },
            }
        ]
        if payload["depth"] > 0:
            records.append(
                {
                    "url": f"{payload['url']}/page",
                    "status": "completed",
                    "markdown": "## Page\n",
                    "metadata": {
                        "status": 200,
                        "title": "Page",
                        "url": f"{payload['url']}/page",
                    },
                }
            )
        return _CloudflareResponse(
            {
                "success": True,
                "result": {
                    "id": job_id,
                    "status": "completed",
                    "records": records,
                },
            }
        )


class _DiscoveryFilteringCloudflareSession(_ParameterizedCloudflareSession):
    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
        timeout: float,
    ) -> _CloudflareResponse:
        response = super().get(url, headers=headers, params=params, timeout=timeout)
        if params == {"limit": 1}:
            return response

        payload = self._jobs[url.rsplit("/", 1)[-1]]
        include_patterns = payload["options"].get("includePatterns", [])
        exclude_patterns = payload["options"].get("excludePatterns", [])
        records = response.json()["result"]["records"]
        filtered_records = [
            record
            for record in records
            if (
                (
                    not include_patterns
                    or any(
                        fnmatch.fnmatchcase(record["url"], pattern)
                        for pattern in include_patterns
                    )
                )
                and not any(
                    fnmatch.fnmatchcase(record["url"], pattern)
                    for pattern in exclude_patterns
                )
            )
        ]
        return _CloudflareResponse(
            {
                "success": True,
                "result": {
                    "id": url.rsplit("/", 1)[-1],
                    "status": "completed",
                    "records": filtered_records,
                },
            }
        )


def test_cloudflare_crawler_polls_job_and_uses_markdown_records(
    tmp_path: Path,
) -> None:
    session = _CloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-cache",
        render=False,
        session=session,
        poll_interval=0,
    )
    scope = CrawlScope(
        roots=["https://example.com/docs"],
        depth=2,
        limit=25,
        include_patterns=["https://example.com/docs/**"],
        exclude_patterns=["https://example.com/docs/archive/**"],
        include_external_links=True,
        include_subdomains=True,
    )

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [
        "https://example.com/docs",
        "https://example.com/docs/page",
    ]
    assert len(session.post_calls) == 1
    post_url, payload, headers = session.post_calls[0]
    assert post_url.endswith("/accounts/account-123/browser-rendering/crawl")
    assert headers["Authorization"] == "Bearer token-123"
    assert payload["formats"] == ["markdown"]
    assert payload["depth"] == 2
    assert payload["limit"] == 25
    assert payload["render"] is False
    assert payload["options"]["includePatterns"] == ["https://example.com/docs/**"]
    assert payload["options"]["excludePatterns"] == [
        "https://example.com/docs/archive/**"
    ]
    assert payload["options"]["includeExternalLinks"] is True
    assert payload["options"]["includeSubdomains"] is True

    page_source = crawler.fetch_raw("https://example.com/docs/page")
    assert page_source.status_code == 200
    assert page_source.markdown_path is not None
    assert page_source.markdown_path.read_text(encoding="utf-8") == "## Page\n"

    page_doc = crawler.fetch_markdown("https://example.com/docs/page")
    assert page_doc == MarkdownDocument(
        origin="https://example.com/docs/page",
        content="## Page\n",
    )
    assert len(session.post_calls) == 1


def test_cloudflare_crawler_accepts_crawl_scope_for_roots_and_patterns(
    tmp_path: Path,
) -> None:
    session = _ParameterizedCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-scope-cache",
        session=session,
        poll_interval=0,
    )
    scope = CrawlScope(
        roots=["https://example.com/docs"],
        depth=1,
        include_patterns=["https://example.com/docs/**"],
    )

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [
        "https://example.com/docs",
        "https://example.com/docs/page",
    ]
    assert session.post_calls[0][1]["depth"] == 1
    assert session.post_calls[0][1]["options"]["includePatterns"] == [
        "https://example.com/docs/**"
    ]


def test_cloudflare_crawler_cache_key_includes_crawl_parameters(
    tmp_path: Path,
) -> None:
    session = _ParameterizedCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-cache",
        session=session,
        poll_interval=0,
    )
    scope = CrawlScope(
        roots=["https://example.com/docs"],
        depth=2,
        limit=25,
    )

    source = crawler.fetch_raw("https://example.com/docs")
    origins = list(crawler.origins(scope, progress=False))

    assert source.origin == "https://example.com/docs"
    assert origins == [
        "https://example.com/docs",
        "https://example.com/docs/page",
    ]
    assert len(session.post_calls) == 2
    assert session.post_calls[0][1]["depth"] == 0
    assert session.post_calls[0][1]["limit"] == 1
    assert session.post_calls[1][1]["depth"] == 2
    assert session.post_calls[1][1]["limit"] == 25


def test_cloudflare_crawler_rechecks_stale_in_memory_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session = _ParameterizedCloudflareSession()
    times = iter(
        [
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
        ]
    )
    monkeypatch.setattr(crawl_module, "_utcnow", lambda: next(times))
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-cache",
        cache_stale_after=timedelta(seconds=1),
        session=session,
        poll_interval=0,
    )
    scope = CrawlScope(roots=["https://example.com/docs"], depth=2)

    origins = list(crawler.origins(scope, progress=False))
    page_source = crawler.fetch_raw("https://example.com/docs/page")

    assert origins == [
        "https://example.com/docs",
        "https://example.com/docs/page",
    ]
    assert page_source.origin == "https://example.com/docs/page"
    assert len(session.post_calls) == 2
    assert session.post_calls[1][1]["url"] == "https://example.com/docs/page"
    assert session.post_calls[1][1]["depth"] == 0
    assert session.post_calls[1][1]["limit"] == 1


def test_cloudflare_fetch_raw_ignores_discovery_patterns_for_explicit_origin(
    tmp_path: Path,
) -> None:
    session = _DiscoveryFilteringCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-cache",
        session=session,
        poll_interval=0,
    )

    source = crawler.fetch_raw("https://example.com/docs")

    assert source.origin == "https://example.com/docs"
    assert source.status_code == 200
    assert "includePatterns" not in session.post_calls[0][1]["options"]


def test_cloudflare_fetch_raw_reuses_cache_directory_across_instances(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cloudflare-cache"
    first_session = _ParameterizedCloudflareSession()
    first_crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=cache,
        session=first_session,
        poll_interval=0,
    )

    first = first_crawler.fetch_raw("https://example.com/docs")

    second_session = _ParameterizedCloudflareSession()
    second_crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=cache,
        session=second_session,
        poll_interval=0,
    )

    second = second_crawler.fetch_raw("https://example.com/docs")

    assert first.body_path.read_text(encoding="utf-8") == "# Docs\n"
    assert second.body_path.read_text(encoding="utf-8") == "# Docs\n"
    assert second.status_code == 200
    assert len(first_session.post_calls) == 1
    assert second_session.post_calls == []


def test_cloudflare_crawler_cache_dir_uses_hashed_file_pair(
    tmp_path: Path,
) -> None:
    session = _ParameterizedCloudflareSession()
    cache_dir = tmp_path / "cloudflare-cache"
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=cache_dir,
        session=session,
        poll_interval=0,
    )

    source = crawler.fetch_raw("https://example.com/docs")

    base = _expected_cache_base(source.origin)
    metadata_path = cache_dir / f"{base}.metadata.json"
    content_path = cache_dir / f"{base}.md"
    assert source.body_path.read_text(encoding="utf-8") == "# Docs\n"
    assert sorted(path.name for path in cache_dir.iterdir()) == [
        content_path.name,
        metadata_path.name,
    ]
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert record["key"] == source.origin
    assert record["content_path"] == content_path.name
    assert record["metadata"]["record"]["url"] == source.origin


def test_cloudflare_crawler_cache_dir_true_uses_default_backend_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    session = _ParameterizedCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=True,
        session=session,
        poll_interval=0,
    )

    source = crawler.fetch_raw("https://example.com/docs")

    cache_dir = tmp_path / ".raghilda" / "cache" / "cloudflare"
    base = _expected_cache_base(source.origin)
    assert sorted(path.name for path in cache_dir.iterdir()) == [
        f"{base}.md",
        f"{base}.metadata.json",
    ]


def test_cloudflare_fetch_raw_scopes_cache_to_account_id(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cloudflare-cache"
    first_session = _ParameterizedCloudflareSession()
    first_crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=cache,
        session=first_session,
        poll_interval=0,
    )
    first_crawler.fetch_raw("https://example.com/docs")

    second_session = _ParameterizedCloudflareSession()
    second_crawler = CloudflareCrawler(
        account_id="account-456",
        api_token="token-123",
        cache_dir=cache,
        session=second_session,
        poll_interval=0,
    )

    second_crawler.fetch_raw("https://example.com/docs")

    assert len(first_session.post_calls) == 1
    assert len(second_session.post_calls) == 1
    assert "/accounts/account-456/" in second_session.post_calls[0][0]


def test_cloudflare_fetch_raw_scopes_cache_to_api_base(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cloudflare-cache"
    first_session = _ParameterizedCloudflareSession()
    first_crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=cache,
        session=first_session,
        poll_interval=0,
        base_url="https://prod.example/api",
    )
    first_crawler.fetch_raw("https://example.com/docs")

    second_session = _ParameterizedCloudflareSession()
    second_crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=cache,
        session=second_session,
        poll_interval=0,
        base_url="https://staging.example/api",
    )

    second_crawler.fetch_raw("https://example.com/docs")

    assert len(first_session.post_calls) == 1
    assert len(second_session.post_calls) == 1
    assert second_session.post_calls[0][0].startswith(
        "https://staging.example/api/accounts/account-123/"
    )


def test_cloudflare_crawler_applies_limit_across_all_roots(
    tmp_path: Path,
) -> None:
    session = _ParameterizedCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-cache",
        session=session,
        poll_interval=0,
    )
    scope = CrawlScope(
        roots=["https://example.com/docs-a", "https://example.com/docs-b"],
        limit=1,
    )

    origins = list(crawler.origins(scope, progress=False))

    assert origins == ["https://example.com/docs-a"]
    assert len(session.post_calls) == 1


def test_directory_crawler_counts_file_roots_toward_limit(tmp_path: Path) -> None:
    first = _write(tmp_path, "a.md", "# First")
    second = _write(tmp_path, "b.md", "# Second")
    crawler = DirectoryCrawler()
    scope = CrawlScope(roots=[first, second], limit=1)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [first.resolve().as_uri()]


def test_directory_crawler_accepts_crawl_scope_for_roots_and_patterns(
    tmp_path: Path,
) -> None:
    docs = _write(tmp_path, "docs/readme.md", "# Hello")
    _write(tmp_path, "notes/todo.md", "# Skip")
    crawler = DirectoryCrawler()
    scope = CrawlScope(
        roots=[tmp_path],
        depth=1,
        include_patterns=[r".*/docs/.*"],
    )

    origins = list(crawler.origins(scope, progress=False))
    documents = list(crawler.markdown_documents(scope, progress=False))

    assert origins == [docs.resolve().as_uri()]
    assert documents == [
        MarkdownDocument(origin=docs.resolve().as_uri(), content="# Hello")
    ]


def test_directory_crawler_returns_no_origins_when_limit_is_zero(
    tmp_path: Path,
) -> None:
    markdown = _write(tmp_path, "a.md", "# First")
    crawler = DirectoryCrawler()
    scope = CrawlScope(roots=[markdown], limit=0)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == []


def test_directory_crawler_fetch_markdown_force_refresh_rebuilds_cached_markdown(
    tmp_path: Path,
) -> None:
    markdown = _write(tmp_path, "docs/readme.md", "# Hello")
    cache = tmp_path / "cache"
    crawler = DirectoryCrawler(cache_dir=cache)

    origin = markdown.resolve().as_uri()
    first = crawler.fetch_markdown(origin)
    cached_markdown = next(
        path for path in cache.iterdir() if not path.name.endswith(".metadata.json")
    )
    cached_markdown.write_text("# Stale\n", encoding="utf-8")

    refreshed = crawler.fetch_markdown(origin, cache_force_refresh=True)

    assert first.content == "# Hello"
    assert refreshed.content == "# Hello"


def test_directory_crawler_markdown_documents_force_refresh_rebuilds_cache(
    tmp_path: Path,
) -> None:
    markdown = _write(tmp_path, "docs/readme.md", "# Hello")
    cache = tmp_path / "cache"
    crawler = DirectoryCrawler(cache_dir=cache)
    root = tmp_path / "docs"
    scope = CrawlScope(roots=[root])

    documents = list(crawler.markdown_documents(scope, progress=False))
    cached_markdown = next(
        path for path in cache.iterdir() if not path.name.endswith(".metadata.json")
    )
    cached_markdown.write_text("# Stale\n", encoding="utf-8")

    refreshed = list(
        crawler.markdown_documents(
            scope,
            progress=False,
            cache_force_refresh=True,
        )
    )

    assert documents == [
        MarkdownDocument(origin=markdown.resolve().as_uri(), content="# Hello")
    ]
    assert refreshed == [
        MarkdownDocument(origin=markdown.resolve().as_uri(), content="# Hello")
    ]


def test_directory_crawler_markdown_documents_converts_in_parallel(
    tmp_path: Path,
) -> None:
    first = _write(tmp_path, "docs/a.md", "# First")
    second = _write(tmp_path, "docs/b.md", "# Second")
    crawler = DirectoryCrawler(max_workers=2)
    scope = CrawlScope(roots=[tmp_path / "docs"])
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def convert(source: FetchedSource) -> MarkdownDocument:
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        try:
            barrier.wait(timeout=1.0)
            return MarkdownDocument(
                origin=source.origin,
                content=source.body_path.read_text(encoding="utf-8"),
            )
        finally:
            with lock:
                in_flight -= 1

    documents = list(crawler.markdown_documents(scope, progress=False, convert=convert))

    assert documents == [
        MarkdownDocument(origin=first.resolve().as_uri(), content="# First"),
        MarkdownDocument(origin=second.resolve().as_uri(), content="# Second"),
    ]
    assert max_in_flight == 2


def test_directory_crawler_reopens_origins_with_uri_escaped_characters(
    tmp_path: Path,
) -> None:
    root = tmp_path / "My Docs"
    markdown = _write(root, "read me.md", "# Hello")
    crawler = DirectoryCrawler()

    origin = next(crawler.origins(CrawlScope(roots=[root]), progress=False))
    document = crawler.fetch_markdown(origin)

    assert "%20" in origin
    assert document == MarkdownDocument(
        origin=markdown.resolve().as_uri(),
        content="# Hello",
    )


def test_directory_crawler_accepts_percent_escaped_file_uri_roots(
    tmp_path: Path,
) -> None:
    root = tmp_path / "My Docs"
    markdown = _write(root, "read me.md", "# Hello")
    crawler = DirectoryCrawler()

    origins = list(
        crawler.origins(CrawlScope(roots=[root.resolve().as_uri()]), progress=False)
    )

    assert origins == [markdown.resolve().as_uri()]


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific file URI handling")
def test_directory_crawler_round_trips_windows_file_uris(
    tmp_path: Path,
) -> None:
    root = tmp_path / "My Docs"
    markdown = _write(root, "read me.md", "# Hello")
    crawler = DirectoryCrawler()

    root_uri = root.resolve().as_uri()
    origin = markdown.resolve().as_uri()

    origins = list(crawler.origins(CrawlScope(roots=[root_uri]), progress=False))
    source = crawler.fetch_raw(origin)

    assert origins == [origin]
    assert source.origin == origin
    assert source.body_path == markdown.resolve()


def test_web_crawler_returns_no_origins_or_requests_when_limit_is_zero(
    tmp_path: Path,
) -> None:
    with _serve(
        {
            "/": {
                "body": "<html><body><main>Root</main></body></html>",
                "content_type": "text/html; charset=utf-8",
                "etag": None,
            }
        }
    ) as server:
        root_url = f"http://127.0.0.1:{server.server_port}/"
        crawler = WebCrawler(
            cache_dir=tmp_path / "zero-limit-cache",
        )
        scope = CrawlScope(roots=[root_url], depth=0, limit=0)

        origins = list(crawler.origins(scope, progress=False))

        assert origins == []
        assert getattr(server, "requests") == []
