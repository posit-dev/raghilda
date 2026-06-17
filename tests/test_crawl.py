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
        include_patterns=["**/docs/**"],
        exclude_patterns=["**/skip.py"],
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
        root_origin = root_url.rstrip("/")
        crawler = WebCrawler(
            cache_dir=tmp_path / "cache",
            cache_stale_after=timedelta(seconds=0),
        )
        scope = CrawlScope(
            roots=[root_url],
            depth=1,
            include_patterns=[f"{root_origin}/**"],
            exclude_patterns=["**/skip"],
        )

        origins = list(crawler.origins(scope, progress=False))

        assert root_origin in origins
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


def test_web_crawler_include_subdomains_uses_redirect_scope(
    tmp_path: Path,
) -> None:
    root = "http://example.com"
    page = "https://example.com/page"
    session: Any = _FakeWebSession(
        {
            root: {
                "body": '<html><body><a href="/page">Page</a></body></html>',
                "resolved_url": "https://example.com/landing",
            },
            page: {
                "body": "<html><body><main>Page</main></body></html>",
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "redirect-subdomain-cache",
        session=session,
    )
    scope = CrawlScope(roots=[root], depth=1, include_subdomains=True)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [root, page]


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


def test_web_crawler_include_subdomains_keeps_original_scope_host(
    tmp_path: Path,
) -> None:
    root = "https://docs.example.com/start"
    api = "https://api.docs.example.com/page"
    cdn = "https://cdn.docs.example.com/asset"
    session: Any = _FakeWebSession(
        {
            root: {
                "body": f'<html><body><a href="{api}">API</a></body></html>',
            },
            api: {
                "body": f'<html><body><a href="{cdn}">CDN</a></body></html>',
            },
            cdn: {"body": "<html><body><main>CDN</main></body></html>"},
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "subdomain-root-host-cache",
        session=session,
    )
    scope = CrawlScope(
        roots=[root],
        depth=2,
        include_subdomains=True,
    )

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [root, api, cdn]


def test_web_crawler_excludes_same_host_different_port_by_default(
    tmp_path: Path,
) -> None:
    root = "http://127.0.0.1:8000"
    other_port = "http://127.0.0.1:9000/page"
    session: Any = _FakeWebSession(
        {
            root: {
                "body": f'<html><body><a href="{other_port}">Other</a></body></html>',
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "same-host-port-cache",
        session=session,
    )
    scope = CrawlScope(roots=[root], depth=1)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [root]
    assert session.requests == [(root, {})]


def test_web_crawler_include_subdomains_excludes_same_host_different_port(
    tmp_path: Path,
) -> None:
    root = "http://127.0.0.1:8000"
    other_port = "http://127.0.0.1:9000/page"
    session: Any = _FakeWebSession(
        {
            root: {
                "body": f'<html><body><a href="{other_port}">Other</a></body></html>',
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "subdomain-same-host-port-cache",
        session=session,
    )
    scope = CrawlScope(roots=[root], depth=1, include_subdomains=True)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [root]
    assert session.requests == [(root, {})]


def test_web_crawler_treats_explicit_default_port_as_same_origin(
    tmp_path: Path,
) -> None:
    root = "http://example.com"
    explicit_root = "http://example.com:80"
    child = "http://example.com/about"
    session: Any = _FakeWebSession(
        {
            root: {
                "body": f'<html><body><a href="{child}">About</a></body></html>',
            },
            child: {
                "body": "<html><body><main>About</main></body></html>",
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "default-port-cache",
        session=session,
    )
    scope = CrawlScope(roots=[explicit_root], depth=1)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [root, child]


def test_web_crawler_deduplicates_explicit_default_port_variants(
    tmp_path: Path,
) -> None:
    root = "https://example.com"
    session: Any = _FakeWebSession(
        {
            root: {
                "body": "<html><body><main>Root</main></body></html>",
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "default-port-variant-cache",
        session=session,
    )
    scope = CrawlScope(roots=[root, "https://example.com:443"], depth=0)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [root]
    assert session.requests == [(root, {})]


def test_web_crawler_normalizes_uppercase_url_schemes(tmp_path: Path) -> None:
    origin = "http://example.com"
    page = "https://example.com/page"
    session: Any = _FakeWebSession(
        {
            origin: {
                "body": (
                    '<html><body><a href="HTTPS://example.com/page">'
                    "Page</a></body></html>"
                ),
            },
            page: {
                "body": "<html><body><main>Page</main></body></html>",
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "uppercase-scheme-cache",
        session=session,
    )
    scope = CrawlScope(
        roots=["HTTP://example.com"], depth=1, include_external_links=True
    )

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [origin, page]


def test_web_crawler_preserves_url_credentials(tmp_path: Path) -> None:
    origin = "https://user:pass@example.com/private"
    session: Any = _FakeWebSession(
        {
            origin: {
                "body": "<html><body><main>Private</main></body></html>",
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "credential-url-cache",
        session=session,
    )

    source = crawler.fetch_raw(origin)

    assert source.origin == origin
    assert session.requests == [(origin, {})]


def test_web_crawler_discovers_urls_from_xml_sitemap(tmp_path: Path) -> None:
    sitemap = "https://example.com/sitemap.xml"
    page = "https://example.com/docs/page"
    session: Any = _FakeWebSession(
        {
            sitemap: {
                "body": (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                    f"<url><loc>{page}</loc></url>"
                    "</urlset>"
                ),
                "content_type": "application/xml",
            },
            page: {
                "body": "<html><body><main>Page</main></body></html>",
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "sitemap-cache",
        session=session,
    )
    scope = CrawlScope(roots=[sitemap], depth=1)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [sitemap, page]


def test_web_crawler_allows_later_in_scope_occurrence_of_same_url(
    tmp_path: Path,
) -> None:
    first_root = "https://alpha.example.com/start"
    second_root = "https://docs.example.com/start"
    shared = "https://api.docs.example.com/page"
    session: Any = _FakeWebSession(
        {
            first_root: {
                "body": f'<html><body><a href="{shared}">Shared</a></body></html>',
            },
            second_root: {
                "body": f'<html><body><a href="{shared}">Shared</a></body></html>',
            },
            shared: {"body": "<html><body><main>Shared</main></body></html>"},
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "multi-root-visited-cache",
        session=session,
    )
    scope = CrawlScope(
        roots=[first_root, second_root],
        depth=1,
        include_subdomains=True,
    )

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [first_root, second_root, shared]


def test_web_crawler_revisits_shared_page_for_broader_subdomain_scope(
    tmp_path: Path,
) -> None:
    narrow_root = "https://api.docs.example.com/start"
    broad_root = "https://docs.example.com/start"
    shared = "https://api.docs.example.com/shared"
    sibling = "https://cdn.docs.example.com/asset"
    session: Any = _FakeWebSession(
        {
            narrow_root: {
                "body": f'<html><body><a href="{shared}">Shared</a></body></html>',
            },
            broad_root: {
                "body": f'<html><body><a href="{shared}">Shared</a></body></html>',
            },
            shared: {
                "body": f'<html><body><a href="{sibling}">Sibling</a></body></html>',
            },
            sibling: {"body": "<html><body><main>Sibling</main></body></html>"},
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "multi-root-subdomain-cache",
        session=session,
    )
    scope = CrawlScope(
        roots=[narrow_root, broad_root],
        depth=2,
        include_subdomains=True,
    )

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [narrow_root, broad_root, shared, sibling]


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
            include_patterns=[f"{root_url}docs/**"],
        )

        origins = list(crawler.origins(scope, progress=False))

        assert root_url not in origins
        assert f"{root_url}docs/guide" in origins


def test_web_crawler_does_not_fetch_excluded_origins(tmp_path: Path) -> None:
    root = "https://example.com"
    admin = "https://example.com/admin"
    session: Any = _FakeWebSession(
        {
            root: {
                "body": f'<html><body><a href="{admin}">Admin</a></body></html>',
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "excluded-origin-cache",
        session=session,
    )
    scope = CrawlScope(
        roots=[root],
        depth=1,
        exclude_patterns=["**/admin"],
    )

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [root]
    assert session.requests == [(root, {})]


def test_web_crawler_deduplicates_root_url_with_and_without_slash(
    tmp_path: Path,
) -> None:
    root = "https://example.com"
    session: Any = _FakeWebSession(
        {
            root: {
                "body": '<html><body><a href="/">Root</a></body></html>',
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "root-slash-cache",
        session=session,
    )
    scope = CrawlScope(roots=[root], depth=1)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [root]
    assert session.requests == [(root, {})]


def test_web_crawler_deduplicates_root_scope_variants(tmp_path: Path) -> None:
    root = "https://example.com"
    session: Any = _FakeWebSession(
        {
            root: {
                "body": "<html><body><main>Root</main></body></html>",
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "root-variant-cache",
        session=session,
    )
    scope = CrawlScope(roots=[root, f"{root}/"], depth=0)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [root]
    assert session.requests == [(root, {})]


def test_web_crawler_deduplicates_queried_root_scope_variants(
    tmp_path: Path,
) -> None:
    root = "https://example.com?x=1"
    session: Any = _FakeWebSession(
        {
            root: {
                "body": "<html><body><main>Root</main></body></html>",
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "queried-root-variant-cache",
        session=session,
    )
    scope = CrawlScope(roots=[root, "https://example.com/?x=1"], depth=0)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [root]
    assert session.requests == [(root, {})]


def test_web_crawler_normalizes_root_links_from_non_root_pages(
    tmp_path: Path,
) -> None:
    root = "https://example.com"
    page = "https://example.com/docs"
    session: Any = _FakeWebSession(
        {
            page: {
                "body": '<html><body><a href="/">Root</a></body></html>',
            },
            root: {
                "body": "<html><body><main>Root</main></body></html>",
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "non-root-root-link-cache",
        session=session,
    )
    scope = CrawlScope(roots=[page], depth=1)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [page, root]
    assert session.requests == [(page, {}), (root, {})]


def test_web_crawler_skips_links_with_malformed_ports(tmp_path: Path) -> None:
    root = "https://example.com"
    session: Any = _FakeWebSession(
        {
            root: {
                "body": (
                    '<html><body><a href="http://example.com:bad/path">'
                    "Bad</a></body></html>"
                ),
            },
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "bad-port-cache",
        session=session,
    )
    scope = CrawlScope(roots=[root], depth=1)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [root]
    assert session.requests == [(root, {})]


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
            include_patterns=[f"{root_url}docs/**"],
        )

        origins = list(crawler.origins(scope, progress=False))
        documents = list(crawler.markdown_documents(scope, progress=False))

        assert origins == [f"{root_url}docs/guide"]
        assert documents == [
            MarkdownDocument(origin=f"{root_url}docs/guide", content="Guide")
        ]


def test_web_crawler_single_star_does_not_cross_path_separator(
    tmp_path: Path,
) -> None:
    with _serve(
        {
            "/": {
                "body": (
                    '<html><body><a href="/docs/guide">Guide</a>'
                    '<a href="/docs/guide/intro">Intro</a></body></html>'
                ),
                "content_type": "text/html; charset=utf-8",
                "etag": None,
            },
            "/docs/guide": {
                "body": "<html><body><main>Guide</main></body></html>",
                "content_type": "text/html; charset=utf-8",
                "etag": None,
            },
            "/docs/guide/intro": {
                "body": "<html><body><main>Intro</main></body></html>",
                "content_type": "text/html; charset=utf-8",
                "etag": None,
            },
        }
    ) as server:
        root_url = f"http://127.0.0.1:{server.server_port}/"
        crawler = WebCrawler(cache_dir=tmp_path / "single-star-cache")
        # `*` matches within a path segment only; `docs/guide/intro` is excluded.
        scope = CrawlScope(
            roots=[root_url],
            depth=2,
            include_patterns=[f"{root_url}docs/*"],
        )

        origins = list(crawler.origins(scope, progress=False))

        assert origins == [f"{root_url}docs/guide"]


def test_web_crawler_accepts_compiled_regex_pattern(tmp_path: Path) -> None:
    root = "https://example.com"
    guide = "https://example.com/guide"
    admin = "https://example.com/admin"
    session: Any = _FakeWebSession(
        {
            root: {
                "body": (
                    f'<html><body><a href="{guide}">Guide</a>'
                    f'<a href="{admin}">Admin</a></body></html>'
                ),
            },
            guide: {"body": "<html><body><main>Guide</main></body></html>"},
            admin: {"body": "<html><body><main>Admin</main></body></html>"},
        }
    )
    crawler = WebCrawler(
        cache_dir=tmp_path / "regex-pattern-cache",
        session=session,
    )
    # A pre-compiled regex is the escape hatch and uses `search` semantics.
    scope = CrawlScope(
        roots=[root],
        depth=1,
        include_patterns=[re.compile(r"/guide$")],
    )

    origins = list(crawler.origins(scope, progress=False))

    assert guide in origins
    assert admin not in origins


def test_glob_pattern_matchers_segment_semantics() -> None:
    single = crawl_module._compile_pattern_matchers(["https://example.com/docs/*"])
    deep = crawl_module._compile_pattern_matchers(["https://example.com/docs/**"])

    def matches(matchers, url: str) -> bool:
        return crawl_module._matches_patterns(
            url, include_matchers=matchers, exclude_matchers=[]
        )

    # `*` stays within a single path segment.
    assert matches(single, "https://example.com/docs/page")
    assert not matches(single, "https://example.com/docs/page/sub")
    # `**` crosses path separators, and a trailing `/**` also matches the bare parent.
    assert matches(deep, "https://example.com/docs/page/sub")
    assert matches(deep, "https://example.com/docs")


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
        root_origin = root_url.rstrip("/")
        crawler = WebCrawler(
            cache_dir=tmp_path / "markdown-docs-cache",
        )
        scope = CrawlScope(roots=[root_url], depth=0)

        documents = list(crawler.markdown_documents(scope, cache_force_refresh=True))
        root_requests = [
            request for request in getattr(server, "requests") if request["path"] == "/"
        ]

        assert documents == [MarkdownDocument(origin=root_origin, content="Root")]
        assert len(root_requests) == 1


def test_web_markdown_documents_reuses_immediately_stale_discovery_cache(
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
        root_origin = root_url.rstrip("/")
        crawler = WebCrawler(
            cache_dir=tmp_path / "stale-markdown-docs-cache",
            cache_stale_after=timedelta(seconds=0),
        )
        scope = CrawlScope(roots=[root_url], depth=0)

        documents = list(crawler.markdown_documents(scope, progress=False))
        root_requests = [
            request for request in getattr(server, "requests") if request["path"] == "/"
        ]

        assert documents == [MarkdownDocument(origin=root_origin, content="Root")]
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
        root_origin = root_url.rstrip("/")
        cache_dir = tmp_path / "cache"
        crawler = WebCrawler(cache_dir=cache_dir)

        document = crawler.fetch_markdown(root_url)

        base = _expected_cache_base(root_origin)
        metadata_path = cache_dir / f"{base}.metadata.json"
        content_path = cache_dir / f"{base}.html"
        assert document == MarkdownDocument(origin=root_origin, content="Root")
        assert sorted(path.name for path in cache_dir.iterdir()) == [
            content_path.name,
            metadata_path.name,
        ]
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert record["key"] == root_origin
        assert record["content_path"] == content_path.name
        assert record["metadata"]["content_type"] == "text/html; charset=utf-8"
        assert record["metadata"]["origin"] == root_origin


def test_web_crawler_rejects_cache_metadata_content_path_outside_cache(
    tmp_path: Path,
) -> None:
    origin = "https://example.com/poison"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    outside = tmp_path / "outside.html"
    outside.write_text("<html><body>Poison</body></html>", encoding="utf-8")
    base = _expected_cache_base(origin)
    metadata_path = cache_dir / f"{base}.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "key": origin,
                "content_path": "../outside.html",
                "metadata": {
                    "origin": origin,
                    "resolved_origin": origin,
                    "content_type": "text/html",
                    "status_code": 200,
                    "etag": None,
                    "last_modified": None,
                    "type_label": "html",
                    "fetched_at": "2026-01-01T00:00:00+00:00",
                    "revalidated_at": None,
                },
            }
        ),
        encoding="utf-8",
    )
    session: Any = _FakeWebSession(
        {
            origin: {
                "body": "<html><body><main>Fresh</main></body></html>",
            }
        }
    )
    crawler = WebCrawler(cache_dir=cache_dir, session=session)

    source = crawler.fetch_raw(origin)

    assert source.body_path.parent == cache_dir
    assert source.body_path != outside
    assert session.requests == [(origin, {})]


def test_web_crawler_rejects_cache_metadata_with_mismatched_key(
    tmp_path: Path,
) -> None:
    origin = "https://example.com/requested"
    stale_origin = "https://example.com/stale"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    base = _expected_cache_base(origin)
    content_path = cache_dir / f"{base}.html"
    content_path.write_text("<html><body>Stale</body></html>", encoding="utf-8")
    metadata_path = cache_dir / f"{base}.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "key": stale_origin,
                "content_path": content_path.name,
                "metadata": {
                    "origin": stale_origin,
                    "resolved_origin": stale_origin,
                    "content_type": "text/html",
                    "status_code": 200,
                    "etag": None,
                    "last_modified": None,
                    "type_label": "html",
                    "fetched_at": "2026-01-01T00:00:00+00:00",
                    "revalidated_at": None,
                },
            }
        ),
        encoding="utf-8",
    )
    session: Any = _FakeWebSession(
        {
            origin: {
                "body": "<html><body><main>Fresh</main></body></html>",
            }
        }
    )
    crawler = WebCrawler(cache_dir=cache_dir, session=session)

    source = crawler.fetch_raw(origin)

    assert source.origin == origin
    assert source.body_path.read_text(encoding="utf-8") != (
        "<html><body>Stale</body></html>"
    )
    assert session.requests == [(origin, {})]


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
        root_origin = root_url.rstrip("/")
        crawler = WebCrawler(cache_dir=True)

        crawler.fetch_markdown(root_url)

    cache_dir = tmp_path / ".raghilda" / "cache" / "web"
    base = _expected_cache_base(root_origin)
    assert sorted(path.name for path in cache_dir.iterdir()) == [
        f"{base}.html",
        f"{base}.metadata.json",
    ]


def test_web_crawler_relative_cache_dir_is_anchored_at_construction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    origin = "https://example.com/page"
    session: Any = _FakeWebSession(
        {
            origin: {
                "body": "<html><body><main>Page</main></body></html>",
            }
        }
    )
    crawler = WebCrawler(cache_dir="cache", session=session)
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    source = crawler.fetch_raw(origin)

    assert source.body_path.parent == tmp_path / "cache"


def test_web_crawler_scopes_fresh_cache_hits_to_custom_session(
    tmp_path: Path,
) -> None:
    origin = "https://example.com/private"
    cache_dir = tmp_path / "session-cache"
    first_session: Any = _FakeWebSession(
        {
            origin: {
                "body": "<html><body><main>First</main></body></html>",
            },
        }
    )
    second_session: Any = _FakeWebSession(
        {
            origin: {
                "body": "<html><body><main>Second</main></body></html>",
            },
        }
    )
    first_crawler = WebCrawler(cache_dir=cache_dir, session=first_session)
    second_crawler = WebCrawler(cache_dir=cache_dir, session=second_session)

    first = first_crawler.fetch_raw(origin)
    first_body = first.body_path.read_text(encoding="utf-8")
    second = second_crawler.fetch_raw(origin)

    assert "First" in first_body
    assert "Second" in second.body_path.read_text(encoding="utf-8")
    assert second_session.requests == [(origin, {})]


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
    assert second_session.requests == [
        (first_origin, {}),
        (second_origin, {}),
        (third_origin, {}),
    ]


def test_web_crawler_refresh_deletes_only_exact_cache_base(tmp_path: Path) -> None:
    first_origin = "https://example.com"
    first_base = _expected_cache_base(first_origin)
    second_origin = f"https://example.com--{first_base.rsplit('--', 1)[1]}.child"
    session: Any = _FakeWebSession(
        {
            first_origin: {"body": "<html><body><main>One</main></body></html>"},
            second_origin: {"body": "<html><body><main>Two</main></body></html>"},
        }
    )
    crawler = WebCrawler(cache_dir=tmp_path / "exact-delete-cache", session=session)

    crawler.fetch_raw(first_origin)
    crawler.fetch_raw(second_origin)
    crawler.fetch_raw(first_origin, cache_force_refresh=True)
    session.requests.clear()
    crawler.fetch_raw(second_origin)

    assert session.requests == []


def test_web_crawler_refresh_replaces_cached_body_atomically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    origin = "https://example.com"
    session: Any = _FakeWebSession(
        {
            origin: {
                "body": "<html><body><main>First</main></body></html>",
            },
        }
    )
    crawler = WebCrawler(cache_dir=tmp_path / "atomic-cache", session=session)
    first = crawler.fetch_raw(origin)
    session.routes[origin]["body"] = "<html><body><main>Second</main></body></html>"
    replacements: list[tuple[Path, Path]] = []
    replace = crawl_module.os.replace

    def track_replace(src: str | Path, dst: str | Path) -> None:
        replacements.append((Path(src), Path(dst)))
        replace(src, dst)

    monkeypatch.setattr(crawl_module.os, "replace", track_replace)

    second = crawler.fetch_raw(origin, cache_force_refresh=True)

    assert first.body_path == second.body_path
    assert second.body_path.read_text(encoding="utf-8") == (
        "<html><body><main>Second</main></body></html>"
    )
    assert replacements[-1][1] == second.body_path


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


def test_web_crawler_type_filters_use_sniffed_cache_extension(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _FakeMagikaOutput:
        label = "html"
        extensions = ["html"]

    class _FakeMagikaResult:
        output = _FakeMagikaOutput()

    class _FakeMagika:
        def identify_bytes(self, content: bytes) -> _FakeMagikaResult:
            assert content.startswith(b"<html>")
            return _FakeMagikaResult()

        def identify_path(self, path: Path) -> _FakeMagikaResult:
            assert path.suffix == ".html"
            return _FakeMagikaResult()

    origin = "https://example.com/download"
    session: Any = _FakeWebSession(
        {
            origin: {
                "body": "<html><body><main>Download</main></body></html>",
                "content_type": "application/octet-stream",
            }
        }
    )
    monkeypatch.setattr(crawl_module, "_MAGIKA", _FakeMagika())
    crawler = WebCrawler(cache_dir=tmp_path / "sniffed-type-cache", session=session)
    scope = CrawlScope(roots=[origin], depth=0, include_types=["html"])

    origins = list(crawler.origins(scope, progress=False))
    source = crawler.fetch_raw(origin)

    assert origins == [origin]
    assert source.metadata == {
        "etag": None,
        "last_modified": None,
        "type_label": "html",
    }


def test_web_crawler_prefers_content_type_over_misleading_url_suffix(
    tmp_path: Path,
) -> None:
    origin = "https://example.com/README.md"
    session: Any = _FakeWebSession(
        {
            origin: {
                "body": "<html><body><main>Rendered Readme</main></body></html>",
                "content_type": "text/html; charset=utf-8",
            }
        }
    )
    cache_dir = tmp_path / "content-type-cache"
    crawler = WebCrawler(cache_dir=cache_dir, session=session)

    source = crawler.fetch_raw(origin)
    document = crawler.fetch_markdown(origin)

    base = _expected_cache_base(origin)
    assert source.body_path == cache_dir / f"{base}.html"
    assert document == MarkdownDocument(origin=origin, content="Rendered Readme")


def test_web_crawler_prefers_text_content_type_over_url_suffix(
    tmp_path: Path,
) -> None:
    origin = "https://example.com/plain.html"
    session: Any = _FakeWebSession(
        {
            origin: {
                "body": "plain text",
                "content_type": "text/plain; charset=utf-8",
            }
        }
    )
    cache_dir = tmp_path / "text-content-type-cache"
    crawler = WebCrawler(cache_dir=cache_dir, session=session)
    scope = CrawlScope(roots=[origin], depth=0, include_types=["text"])

    origins = list(crawler.origins(scope, progress=False))
    source = crawler.fetch_raw(origin)

    assert origins == [origin]
    assert source.body_path == cache_dir / f"{_expected_cache_base(origin)}.txt"
    assert (source.metadata or {})["type_label"] == "text"


def test_web_crawler_preserves_reserved_escapes_in_requested_origin(
    tmp_path: Path,
) -> None:
    origin = "https://example.com/a%2Fb"
    session: Any = _FakeWebSession(
        {
            origin: {
                "body": "<html><body><main>Escaped</main></body></html>",
            }
        }
    )
    cache_dir = tmp_path / "escaped-cache"
    crawler = WebCrawler(cache_dir=cache_dir, session=session)

    source = crawler.fetch_raw(origin)

    assert session.requests == [(origin, {})]
    assert source.origin == origin
    assert source.body_path == cache_dir / f"{_expected_cache_base(origin)}.html"


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


class _OverlappingLimitedCloudflareSession(_ParameterizedCloudflareSession):
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
                "url": "https://example.com/shared",
                "status": "completed",
                "markdown": "# Shared\n",
                "metadata": {
                    "status": 200,
                    "title": "Shared",
                    "url": "https://example.com/shared",
                },
            }
        ]
        if payload["url"] == "https://example.com/root-b":
            records.append(
                {
                    "url": "https://example.com/root-b/unique",
                    "status": "completed",
                    "markdown": "# Unique\n",
                    "metadata": {
                        "status": 200,
                        "title": "Unique",
                        "url": "https://example.com/root-b/unique",
                    },
                }
            )
        if "limit" in payload:
            records = records[: payload["limit"]]
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


class _TrailingSlashCloudflareSession(_ParameterizedCloudflareSession):
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
        records = response.json()["result"]["records"]
        records[0]["url"] = f"{payload['url'].rstrip('/')}/"
        records[0]["metadata"]["url"] = records[0]["url"]
        return response


class _OutOfScopeCloudflareSession(_ParameterizedCloudflareSession):
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
        job_id = url.rsplit("/", 1)[-1]
        payload = self._jobs[job_id]
        root = payload["url"]
        records = [
            {
                "url": root,
                "status": "completed",
                "markdown": "# Root\n",
                "metadata": {
                    "status": 200,
                    "title": "Root",
                    "url": root,
                },
            },
            {
                "url": "https://example.com/page",
                "status": "completed",
                "markdown": "# Page\n",
                "metadata": {
                    "status": 200,
                    "title": "Page",
                    "url": "https://example.com/page",
                },
            },
            {
                "url": "https://docs.example.com/page",
                "status": "completed",
                "markdown": "# Subdomain\n",
                "metadata": {
                    "status": 200,
                    "title": "Subdomain",
                    "url": "https://docs.example.com/page",
                },
            },
            {
                "url": "https://external.test/page",
                "status": "completed",
                "markdown": "# External\n",
                "metadata": {
                    "status": 200,
                    "title": "External",
                    "url": "https://external.test/page",
                },
            },
        ]
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


class _ExternalFirstCloudflareSession(_ParameterizedCloudflareSession):
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
        job_id = url.rsplit("/", 1)[-1]
        root = self._jobs[job_id]["url"]
        return _CloudflareResponse(
            {
                "success": True,
                "result": {
                    "id": job_id,
                    "status": "completed",
                    "records": [
                        {
                            "url": "https://external.test/page",
                            "status": "completed",
                            "markdown": "# External\n",
                            "metadata": {
                                "status": 200,
                                "title": "External",
                                "url": "https://external.test/page",
                            },
                        },
                        {
                            "url": root,
                            "status": "completed",
                            "markdown": "# Root\n",
                            "metadata": {
                                "status": 200,
                                "title": "Root",
                                "url": root,
                            },
                        },
                    ],
                },
            }
        )


class _RedirectCloudflareSession(_ParameterizedCloudflareSession):
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
        job_id = url.rsplit("/", 1)[-1]
        root = self._jobs[job_id]["url"]
        final_url = f"{root.rstrip('/')}/landing"
        return _CloudflareResponse(
            {
                "success": True,
                "result": {
                    "id": job_id,
                    "status": "completed",
                    "records": [
                        {
                            "url": final_url,
                            "status": "completed",
                            "markdown": "# Landing\n",
                            "metadata": {
                                "status": 200,
                                "title": "Landing",
                                "url": final_url,
                            },
                        }
                    ],
                },
            }
        )


class _CrossOriginRedirectCloudflareSession(_ParameterizedCloudflareSession):
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
        job_id = url.rsplit("/", 1)[-1]
        final_url = "https://example.com/landing"
        return _CloudflareResponse(
            {
                "success": True,
                "result": {
                    "id": job_id,
                    "status": "completed",
                    "records": [
                        {
                            "url": final_url,
                            "status": "completed",
                            "markdown": "# Landing\n",
                            "metadata": {
                                "status": 200,
                                "title": "Landing",
                                "url": final_url,
                            },
                        }
                    ],
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


def test_cloudflare_markdown_documents_reuses_immediately_stale_discovery_cache(
    tmp_path: Path,
) -> None:
    session = _ParameterizedCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-stale-cache",
        session=session,
        cache_stale_after=timedelta(seconds=0),
        poll_interval=0,
    )
    scope = CrawlScope(roots=["https://example.com"], depth=0)

    documents = list(crawler.markdown_documents(scope, progress=False))

    assert documents == [
        MarkdownDocument(origin="https://example.com", content="# Docs\n")
    ]
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


def test_cloudflare_crawler_filters_returned_records_to_web_scope(
    tmp_path: Path,
) -> None:
    session = _OutOfScopeCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-returned-scope-cache",
        session=session,
        poll_interval=0,
    )
    scope = CrawlScope(
        roots=["https://example.com/root"],
        depth=1,
        include_external_links=False,
        include_subdomains=False,
    )

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [
        "https://example.com/root",
        "https://example.com/page",
    ]


def test_cloudflare_crawler_does_not_treat_external_first_record_as_seed(
    tmp_path: Path,
) -> None:
    session = _ExternalFirstCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-external-first-cache",
        session=session,
        poll_interval=0,
    )
    scope = CrawlScope(
        roots=["https://example.com/root"],
        depth=1,
        limit=1,
        include_external_links=False,
        include_subdomains=False,
    )

    origins = list(crawler.origins(scope, progress=False))

    assert origins == ["https://example.com/root"]


def test_cloudflare_markdown_documents_keeps_cross_origin_redirected_seed(
    tmp_path: Path,
) -> None:
    session = _CrossOriginRedirectCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-cross-origin-seed-cache",
        session=session,
        poll_interval=0,
    )
    scope = CrawlScope(
        roots=["http://example.com"],
        depth=0,
        include_external_links=False,
        include_subdomains=False,
    )

    documents = list(crawler.markdown_documents(scope, progress=False))

    assert documents == [
        MarkdownDocument(
            origin="https://example.com/landing",
            content="# Landing\n",
        )
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


def test_cloudflare_fetch_raw_accepts_redirected_record(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cloudflare-redirect-cache"
    session = _RedirectCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=cache,
        session=session,
        poll_interval=0,
    )

    source = crawler.fetch_raw("https://example.com")

    assert source.origin == "https://example.com"
    assert source.resolved_origin == "https://example.com/landing"
    assert source.body_path.read_text(encoding="utf-8") == "# Landing\n"
    assert len(session.post_calls) == 1

    cached_session = _RedirectCloudflareSession()
    cached_crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=cache,
        session=cached_session,
        poll_interval=0,
    )
    cached_source = cached_crawler.fetch_raw("https://example.com")

    assert cached_source.resolved_origin == "https://example.com/landing"
    assert cached_session.post_calls == []


def test_cloudflare_fetch_raw_accepts_cross_origin_redirected_record(
    tmp_path: Path,
) -> None:
    session = _CrossOriginRedirectCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-cross-origin-redirect-cache",
        session=session,
        poll_interval=0,
    )

    source = crawler.fetch_raw("http://example.com")

    assert source.origin == "http://example.com"
    assert source.resolved_origin == "https://example.com/landing"
    assert source.body_path.read_text(encoding="utf-8") == "# Landing\n"


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


def test_cloudflare_origins_reuses_root_cache_directory_across_instances(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cloudflare-cache"
    scope = CrawlScope(roots=["https://example.com/docs"], depth=1)
    first_session = _ParameterizedCloudflareSession()
    first_crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=cache,
        session=first_session,
        poll_interval=0,
    )

    first_origins = list(first_crawler.origins(scope, progress=False))

    second_session = _ParameterizedCloudflareSession()
    second_crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=cache,
        session=second_session,
        poll_interval=0,
    )

    second_origins = list(second_crawler.origins(scope, progress=False))
    page_source = second_crawler.fetch_raw("https://example.com/docs/page")

    assert first_origins == [
        "https://example.com/docs",
        "https://example.com/docs/page",
    ]
    assert second_origins == first_origins
    assert page_source.origin == "https://example.com/docs/page"
    assert len(first_session.post_calls) == 1
    assert second_session.post_calls == []


def test_cloudflare_markdown_documents_canonicalizes_record_urls(
    tmp_path: Path,
) -> None:
    session = _TrailingSlashCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-record-url-cache",
        session=session,
        poll_interval=0,
    )
    scope = CrawlScope(roots=["https://example.com"], depth=0)

    documents = list(crawler.markdown_documents(scope, progress=False))

    assert documents == [
        MarkdownDocument(origin="https://example.com", content="# Docs\n")
    ]


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


def test_cloudflare_crawler_deduplicates_roots_before_counting_limit(
    tmp_path: Path,
) -> None:
    session = _ParameterizedCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-dedupe-cache",
        session=session,
        poll_interval=0,
    )
    scope = CrawlScope(
        roots=[
            "https://example.com/docs-a",
            "https://example.com/docs-a",
            "https://example.com/docs-b",
        ],
        depth=0,
        limit=2,
    )

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [
        "https://example.com/docs-a",
        "https://example.com/docs-b",
    ]
    assert [call[1]["url"] for call in session.post_calls] == [
        "https://example.com/docs-a",
        "https://example.com/docs-b",
    ]


def test_cloudflare_crawler_applies_limit_after_deduplication(
    tmp_path: Path,
) -> None:
    session = _OverlappingLimitedCloudflareSession()
    crawler = CloudflareCrawler(
        account_id="account-123",
        api_token="token-123",
        cache_dir=tmp_path / "cloudflare-overlap-cache",
        session=session,
        poll_interval=0,
    )
    scope = CrawlScope(
        roots=[
            "https://example.com/root-a",
            "https://example.com/root-b",
        ],
        limit=2,
    )

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [
        "https://example.com/shared",
        "https://example.com/root-b/unique",
    ]
    assert "limit" not in session.post_calls[1][1]


def test_directory_crawler_counts_file_roots_toward_limit(tmp_path: Path) -> None:
    first = _write(tmp_path, "a.md", "# First")
    second = _write(tmp_path, "b.md", "# Second")
    crawler = DirectoryCrawler()
    scope = CrawlScope(roots=[first, second], limit=1)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [first.resolve().as_uri()]


def test_directory_crawler_deduplicates_roots_before_counting_limit(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    first = _write(docs, "a.md", "# First")
    second = _write(docs, "b.md", "# Second")
    crawler = DirectoryCrawler()
    scope = CrawlScope(roots=[first, docs], limit=2)

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [
        first.resolve().as_uri(),
        second.resolve().as_uri(),
    ]


def test_directory_crawler_applies_limit_without_prewalking_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _write(tmp_path, "a.md", "# First")
    _write(tmp_path, "z/b.md", "# Second")
    crawler = DirectoryCrawler()

    def fail_rglob(self: Path, pattern: str):
        del self, pattern
        raise AssertionError("DirectoryCrawler should not prewalk with rglob")

    monkeypatch.setattr(Path, "rglob", fail_rglob)

    origins = list(
        crawler.origins(CrawlScope(roots=[tmp_path], depth=0, limit=1), progress=False)
    )

    assert origins == [first.resolve().as_uri()]


def test_directory_crawler_does_not_follow_symlinked_directories_outside_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    inside = _write(root, "inside.md", "# Inside")
    external_dir = tmp_path / "external"
    outside = _write(external_dir, "outside.md", "# Outside")
    link = root / "linked"
    try:
        link.symlink_to(external_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlink creation failed: {exc}")
    crawler = DirectoryCrawler()

    origins = list(crawler.origins(CrawlScope(roots=[root], depth=2), progress=False))

    assert origins == [inside.resolve().as_uri()]
    assert outside.resolve().as_uri() not in origins


def test_directory_crawler_skips_type_sniffing_without_type_filters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    document = _write(tmp_path, "extensionless", "# Document")

    class _FailingMagika:
        def identify_path(self, path: Path):
            raise AssertionError(f"Unexpected type sniff for {path}")

    monkeypatch.setattr(crawl_module, "_MAGIKA", _FailingMagika())
    crawler = DirectoryCrawler()

    origins = list(crawler.origins(CrawlScope(roots=[tmp_path]), progress=False))

    assert origins == [document.resolve().as_uri()]


def test_directory_crawler_coerces_scalar_patterns_and_types(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    readme = _write(docs, "readme.md", "# Readme")
    _write(docs, "skip.py", "print('skip')")
    _write(tmp_path, "notes.md", "# Notes")
    crawler = DirectoryCrawler()
    scope = CrawlScope(
        roots=[tmp_path],
        include_patterns=r".*/docs/.*",
        include_types="markdown",
        exclude_types="python",
    )

    origins = list(crawler.origins(scope, progress=False))

    assert origins == [readme.resolve().as_uri()]


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


def test_directory_crawler_fetch_markdown_refreshes_when_file_changes(
    tmp_path: Path,
) -> None:
    markdown = _write(tmp_path, "docs/readme.md", "# Hello")
    cache = tmp_path / "cache"
    crawler = DirectoryCrawler(cache_dir=cache)

    origin = markdown.resolve().as_uri()
    first = crawler.fetch_markdown(origin)
    markdown.write_text("# Updated\n", encoding="utf-8")

    refreshed = crawler.fetch_markdown(origin)

    assert first == MarkdownDocument(origin=origin, content="# Hello")
    assert refreshed == MarkdownDocument(origin=origin, content="# Updated\n")


def test_directory_crawler_excludes_own_cache_files_from_directory_walk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    markdown = _write(tmp_path, "docs/readme.md", "# Hello")
    monkeypatch.chdir(tmp_path)
    crawler = DirectoryCrawler(cache_dir=True)
    scope = CrawlScope(roots=[tmp_path])

    documents = list(crawler.markdown_documents(scope, progress=False))
    origins = list(crawler.origins(scope, progress=False))

    assert documents == [
        MarkdownDocument(origin=markdown.resolve().as_uri(), content="# Hello")
    ]
    assert origins == [markdown.resolve().as_uri()]


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


def test_directory_crawler_accepts_windows_drive_letter_string_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "C:\\docs"
    markdown = _write(root, "readme.md", "# Hello")
    monkeypatch.chdir(tmp_path)
    crawler = DirectoryCrawler()

    origins = list(crawler.origins(CrawlScope(roots=["C:\\docs"]), progress=False))

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


def test_web_crawler_does_not_fetch_extra_root_once_limit_is_reached(
    tmp_path: Path,
) -> None:
    with _serve(
        {
            "/first": {
                "body": "<html><body><main>First</main></body></html>",
                "content_type": "text/html; charset=utf-8",
                "etag": None,
            },
            "/second": {
                "body": "<html><body><main>Second</main></body></html>",
                "content_type": "text/html; charset=utf-8",
                "etag": None,
            },
        }
    ) as server:
        root_url = f"http://127.0.0.1:{server.server_port}"
        crawler = WebCrawler(
            cache_dir=tmp_path / "limit-cache",
            max_workers=2,
        )
        scope = CrawlScope(
            roots=[f"{root_url}/first", f"{root_url}/second"],
            depth=0,
            limit=1,
        )

        origins = list(crawler.origins(scope, progress=False))
        requests = [request["path"] for request in getattr(server, "requests")]

        assert origins == [f"{root_url}/first"]
        assert requests == ["/first"]
