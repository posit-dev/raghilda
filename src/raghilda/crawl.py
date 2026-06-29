from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TypeVar
import threading
import unicodedata
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
from urllib.request import url2pathname

import requests

from .document import MarkdownDocument
from .read import _convert_to_markdown
from .scrape import _extract_links

try:
    from magika import Magika
except ImportError:  # pragma: no cover - optional at runtime
    Magika = None

__all__ = [
    "BaseCrawler",
    "CrawlScope",
    "FetchedSource",
    "WebCrawler",
    "DirectoryCrawler",
    "CloudflareCrawler",
]

_TYPE_ALIASES = {
    ".htm": "html",
    ".html": "html",
    ".ipynb": "jupyter-notebook",
    ".markdown": "markdown",
    ".md": "markdown",
    ".pdf": "pdf",
    ".py": "python",
    ".rst": "rst",
    ".txt": "text",
}
_CONTENT_TYPE_LABELS = {
    "application/json": "json",
    "application/pdf": "pdf",
    "application/xml": "xml",
    "text/html": "html",
    "text/markdown": "markdown",
    "text/plain": "text",
    "text/x-python": "python",
    "text/xml": "xml",
}
_MAGIKA_LABELS = {
    "html": "html",
    "ipynb": "jupyter-notebook",
    "markdown": "markdown",
    "pdf": "pdf",
    "python": "python",
    "rst": "rst",
    "txt": "text",
}
_TERMINAL_CLOUDFLARE_STATUSES = {
    "cancelled_by_user",
    "cancelled_due_to_limits",
    "cancelled_due_to_timeout",
    "completed",
    "errored",
}
_MAGIKA = Magika() if Magika is not None else None
_DEFAULT_CRAWL_DEPTH = 100_000

RootInput = str | Path
RootsInput = RootInput | Sequence[RootInput]
PatternInput = str | re.Pattern[str]
PatternsInput = PatternInput | Sequence[PatternInput] | None
CacheValue = tuple[Path | None, dict[str, Any] | None]
CacheEntry = tuple[str, Path | None, dict[str, Any] | None]
WebOriginKey = tuple[str, str, int | None]
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


@dataclass(frozen=True)
class CrawlScope:
    """Declarative description of what a crawler should discover.

    A `CrawlScope` is the traversal policy shared by every crawler. It names
    the starting points and the rules used to decide which origins are followed
    and yielded. The same scope can be reused across `DirectoryCrawler`,
    `WebCrawler`, and `CloudflareCrawler`, though a few fields are interpreted
    slightly differently per backend.

    Parameters
    ----------
    roots
        Starting files, directories, or URLs. May be a single value or a
        sequence of values.
    include_patterns
        Patterns that an origin must match to be yielded. A `str` is treated as
        a glob: `*` matches any run of characters except `/`, and `**` matches
        across `/` (a trailing `/**` also matches the bare parent, so `/docs/**`
        matches `/docs` too). Pass a compiled `re.Pattern` to match by regular
        expression instead. Accepts a single pattern or a sequence; when `None`,
        all origins are allowed.
    exclude_patterns
        Patterns that drop an origin from the crawl, taking precedence over
        `include_patterns`. Uses the same glob-or-`re.Pattern` syntax.
    depth
        Number of link or directory levels to follow beyond the roots. `0` means
        only the roots themselves. When `None`, traversal is effectively
        unbounded. Must be non-negative.
    limit
        Maximum number of origins to yield. When `None`, no limit is applied.
        Must be non-negative.
    include_types
        Type labels to include, such as `"html"`, `"markdown"`, `"pdf"`,
        `"python"`, or `"text"`. When `None` or empty, all types are allowed.
    exclude_types
        Type labels to skip. Takes precedence over `include_types`.
    include_external_links
        Allow origins outside the root origin (a different scheme, host, or
        port). Defaults to `False`.
    include_subdomains
        Allow origins on subdomains of the root host. Defaults to `False`.
    """

    roots: RootsInput
    include_patterns: PatternsInput = None
    exclude_patterns: PatternsInput = None
    depth: int | None = None
    limit: int | None = None
    include_types: Sequence[str] | None = None
    exclude_types: Sequence[str] | None = None
    include_external_links: bool = False
    include_subdomains: bool = False

    def __post_init__(self) -> None:
        if self.depth is not None:
            assert self.depth >= 0
        if self.limit is not None:
            assert self.limit >= 0


@dataclass(frozen=True)
class FetchedSource:
    """A fetched source document and its metadata, prior to conversion.

    A `FetchedSource` is the intermediate result returned by a crawler's
    `fetch_raw()`. It points at the raw body on disk and carries the metadata
    needed to convert it to a `MarkdownDocument`. Custom `convert` callables
    passed to `fetch_markdown()` or `markdown_documents()` receive an instance
    of this class.

    Parameters
    ----------
    origin
        The canonical origin the source was requested from (a `file://` URI for
        local files, an `http(s)` URL for web sources).
    body_path
        Filesystem path to the raw fetched body. For local files this is the
        file itself; for web and Cloudflare sources it is a cached copy.
    resolved_origin
        The final origin after any redirects, when it differs from `origin`.
    content_type
        The reported MIME type, such as `"text/html"`, when available.
    status_code
        The HTTP status code for web sources, when available.
    metadata
        Backend-specific metadata, such as the detected `type_label`, validators
        (`etag`, `last_modified`), and source hashes.
    fetched_at
        When the source body was fetched, when known.
    revalidated_at
        When a cached body was last revalidated against the server, when known.
    markdown_path
        Filesystem path to already-converted Markdown, when the backend produced
        or cached it. `None` when conversion has not run.
    """

    origin: str
    body_path: Path
    resolved_origin: str | None = None
    content_type: str | None = None
    status_code: int | None = None
    metadata: dict[str, Any] | None = None
    fetched_at: datetime | None = None
    revalidated_at: datetime | None = None
    markdown_path: Path | None = None


@dataclass(frozen=True)
class _CloudflareRootCacheEntry:
    fetched_at: datetime
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class _CloudflareRecordCacheEntry:
    fetched_at: datetime
    record: dict[str, Any]


@dataclass(frozen=True)
class _ResolvedCrawlScope:
    roots: list[RootInput]
    include_patterns: list[PatternInput]
    exclude_patterns: list[PatternInput]
    include_matchers: list[Callable[[str], bool]]
    exclude_matchers: list[Callable[[str], bool]]
    depth: int
    limit: int | None
    include_types: set[str]
    exclude_types: set[str]
    include_external_links: bool
    include_subdomains: bool


@dataclass
class _EntryLockState:
    lock: threading.RLock
    users: int = 0


class _FilesystemCrawlerCache:
    """
    Filesystem-backed cache rooted at one directory.

    Each logical key is stored as:
        <root>/<sanitized-key>--<hash>.metadata.json
        <root>/<sanitized-key>--<hash><ext>

    The metadata file is the source of truth and stores:
        {
            "key": <original unsanitized key>,
            "content_path": <basename of content file, or null>,
            "metadata": <user metadata dict, or null>,
        }
    """

    _METADATA_SUFFIX = ".metadata.json"
    _HASH_LEN = 12
    _MAX_STEM_LEN = 180

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

    def __init__(self, root: Path | None) -> None:
        """Create a filesystem-backed cache rooted at one directory."""
        self.root = root
        self._entry_locks_guard = threading.Lock()
        self._entry_locks: dict[str, _EntryLockState] = {}
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def fetch(self, key: str) -> CacheValue | None:
        """
        Return the materialized cache entry for one key, if present.

        This method does not lock for normal reads. If it encounters a broken
        metadata file, it triggers a locked re-check and best-effort cleanup,
        then returns None.
        """
        if self.root is None:
            return None

        metadata_path = self._metadata_path_for_key(key)
        if not metadata_path.exists():
            return None

        record = self._read_record(metadata_path)
        if record is None:
            self._cleanup_broken_metadata_path(metadata_path)
            return None
        if record["key"] != key:
            self._cleanup_mismatched_metadata_path(metadata_path, key)
            return None

        content_path: Path | None = None
        content_name = record["content_path"]
        if content_name is not None:
            candidate = self.root / content_name
            if candidate.exists():
                content_path = candidate

        return content_path, record["metadata"]

    def upsert(
        self,
        key: str,
        *,
        content: bytes | str | Path | None,
        metadata: Mapping[str, Any] | None,
        content_ext: str | None,
    ) -> CacheValue | None:
        """
        Create or replace one cache entry.

        Semantics:
        - content=None means no content file for this entry
        - metadata=None means no user metadata for this entry
        - the metadata sidecar is always written, unless both are None
        - (content=None, metadata=None) deletes the entry and returns None
        """
        if self.root is None:
            return None

        if content is None and metadata is None:
            self.delete(key)
            return None

        base = self._base_for_key(key)
        metadata_path = self.root / f"{base}{self._METADATA_SUFFIX}"
        stored_metadata = dict(metadata) if metadata is not None else None
        new_content_path: Path | None = None
        new_content_name: str | None = None
        if content is not None:
            ext = self._choose_content_ext(
                content=content,
                content_ext=content_ext,
            )
            new_content_path = self.root / f"{base}{ext}"
            new_content_name = new_content_path.name
        record = {
            "key": key,
            "content_path": new_content_name,
            "metadata": stored_metadata,
        }
        keep = {metadata_path.name}
        if new_content_name is not None:
            keep.add(new_content_name)

        with self._locked_base(base):
            if metadata_path.exists() and self._read_record(metadata_path) is None:
                self._delete_base_files_locked(base)

            if content is not None:
                assert new_content_path is not None
                self._write_content(new_content_path, content)

            self._write_json(metadata_path, record)
            self._delete_extra_base_files_locked(base, keep=keep)

            return new_content_path, stored_metadata

    def delete(self, key: str) -> int:
        """
        Delete one cache entry.

        Returns the number of files removed.
        """
        if self.root is None:
            return 0

        base = self._base_for_key(key)
        with self._locked_base(base):
            return self._delete_base_files_locked(base)

    def entries(self) -> Iterable[CacheEntry]:
        """
        Yield all cache entries currently described by metadata files.

        This method does not lock for normal reads. Broken metadata files are
        re-checked under the write lock and cleaned up if still invalid.
        """
        if self.root is None:
            return

        for metadata_path in sorted(self.root.glob(f"*{self._METADATA_SUFFIX}")):
            record = self._read_record(metadata_path)
            if record is None:
                self._cleanup_broken_metadata_path(metadata_path)
                continue

            content_path: Path | None = None
            content_name = record["content_path"]
            if content_name is not None:
                candidate = self.root / content_name
                if candidate.exists():
                    content_path = candidate

            yield record["key"], content_path, record["metadata"]

    def _metadata_path_for_key(self, key: str) -> Path:
        """Return the deterministic metadata path for one logical key."""
        assert self.root is not None
        return self.root / f"{self._base_for_key(key)}{self._METADATA_SUFFIX}"

    def _base_for_key(self, key: str) -> str:
        """Build the shared basename for the metadata file and content file."""
        return f"{self._sanitize_stem(key)}--{self._hash_fragment(key)}"

    def _hash_fragment(self, key: str) -> str:
        """Return a stable hash fragment of the original key."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[: self._HASH_LEN]

    def _sanitize_stem(self, key: str) -> str:
        """Make the key visible in the filename, but safe enough for Windows."""
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
        if root in self._WINDOWS_RESERVED:
            value = f"_{value}"

        if len(value) > self._MAX_STEM_LEN:
            head = self._MAX_STEM_LEN // 2 - 2
            tail = self._MAX_STEM_LEN - head - 2
            value = f"{value[:head]}..{value[-tail:]}"

        value = value.rstrip(" .")
        return value or "entry"

    def _choose_content_ext(
        self,
        *,
        content: bytes | str | Path,
        content_ext: str | None,
    ) -> str:
        """Choose the content file extension."""
        ext = self._normalize_ext(content_ext)
        if ext is not None:
            return ext

        if isinstance(content, Path):
            ext = self._normalize_ext(content.suffix)
            if ext is not None:
                return ext

        ext = self._infer_ext_with_magika(content)
        if ext is not None:
            return ext

        if isinstance(content, str):
            return ".txt"

        return ".raw"

    def _infer_ext_with_magika(self, content: bytes | str | Path) -> str | None:
        """Best-effort extension inference using Magika."""
        if _MAGIKA is None:
            return None

        if isinstance(content, Path):
            if not content.exists():
                return None
            result = _MAGIKA.identify_path(content)
        elif isinstance(content, str):
            result = _MAGIKA.identify_bytes(content.encode("utf-8"))
        else:
            result = _MAGIKA.identify_bytes(content)

        extensions = getattr(result.output, "extensions", None)
        if not extensions:
            return None
        return self._normalize_ext(extensions[0])

    def _normalize_ext(self, ext: str | None) -> str | None:
        """Normalize an extension string into a safe canonical form."""
        if ext is None:
            return None

        ext = ext.strip()
        if not ext:
            return None

        if not ext.startswith("."):
            ext = "." + ext

        parts = [part for part in ext.split(".") if part]
        if not parts:
            return None

        cleaned: list[str] = []
        for part in parts:
            token = re.sub(r"[^A-Za-z0-9_-]+", "", part)
            if token:
                cleaned.append(token.lower())

        if not cleaned:
            return None

        return "".join(f".{part}" for part in cleaned)

    def _read_record(self, path: Path) -> dict[str, Any] | None:
        """Read and validate one metadata JSON file."""
        try:
            with path.open("r", encoding="utf-8") as handle:
                obj = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(obj, dict):
            return None

        key = obj.get("key")
        content_path = obj.get("content_path")
        metadata = obj.get("metadata")

        if not isinstance(key, str):
            return None
        if content_path is not None and not isinstance(content_path, str):
            return None
        if content_path is not None:
            if content_path in {"", ".", ".."}:
                return None
            if Path(content_path).name != content_path or "\\" in content_path:
                return None
        if metadata is not None and not isinstance(metadata, dict):
            return None

        return {
            "key": key,
            "content_path": content_path,
            "metadata": metadata,
        }

    def _cleanup_broken_metadata_path(self, metadata_path: Path) -> None:
        """Best-effort cleanup for a broken metadata file."""
        if self.root is None:
            return
        if not metadata_path.name.endswith(self._METADATA_SUFFIX):
            return

        base = metadata_path.name[: -len(self._METADATA_SUFFIX)]
        with self._locked_base(base):
            if not metadata_path.exists():
                return
            if self._read_record(metadata_path) is not None:
                return

            self._delete_base_files_locked(base)

    def _cleanup_mismatched_metadata_path(
        self,
        metadata_path: Path,
        key: str,
    ) -> None:
        """Best-effort cleanup for a metadata file stored under the wrong key."""
        if self.root is None:
            return

        base = self._base_for_key(key)
        with self._locked_base(base):
            if not metadata_path.exists():
                return
            record = self._read_record(metadata_path)
            if record is not None and record["key"] == key:
                return

            self._delete_base_files_locked(base)

    @contextmanager
    def _locked_base(self, base: str) -> Iterator[None]:
        state = self._acquire_entry_lock_state(base)
        state.lock.acquire()
        try:
            yield
        finally:
            self._release_entry_lock_state(base, state)

    def _acquire_entry_lock_state(self, base: str) -> _EntryLockState:
        with self._entry_locks_guard:
            state = self._entry_locks.get(base)
            if state is None:
                state = _EntryLockState(lock=threading.RLock())
                self._entry_locks[base] = state
            state.users += 1
            return state

    def _release_entry_lock_state(self, base: str, state: _EntryLockState) -> None:
        state.lock.release()
        with self._entry_locks_guard:
            current = self._entry_locks.get(base)
            assert current is state
            state.users -= 1
            if state.users == 0:
                del self._entry_locks[base]

    def _delete_base_files_locked(self, base: str) -> int:
        """Delete all files belonging to one logical base."""
        assert self.root is not None

        deleted = 0
        for path in self.root.iterdir():
            if not self._belongs_to_base(path.name, base):
                continue
            if not path.is_file():
                continue
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                pass
        return deleted

    def _delete_extra_base_files_locked(self, base: str, *, keep: set[str]) -> None:
        """Delete stale files for one base, keeping the current pair."""
        assert self.root is not None

        for path in self.root.iterdir():
            if not self._belongs_to_base(path.name, base):
                continue
            if not path.is_file():
                continue
            if path.name in keep:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _belongs_to_base(self, name: str, base: str) -> bool:
        if name == f"{base}{self._METADATA_SUFFIX}":
            return True
        prefix = f"{base}."
        if not name.startswith(prefix):
            return False
        return "--" not in name[len(prefix) :]

    def _write_content(self, content_path: Path, content: bytes | str | Path) -> None:
        if isinstance(content, Path):
            if content == content_path:
                return

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=content_path.parent,
                prefix=f".{content_path.name}.",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                if isinstance(content, bytes):
                    handle.write(content)
                elif isinstance(content, str):
                    handle.write(content.encode("utf-8"))
                elif isinstance(content, Path):
                    with content.open("rb") as source:
                        shutil.copyfileobj(source, handle)
                else:
                    raise TypeError(f"Unsupported content type: {type(content)!r}")
            os.replace(temporary_path, content_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _write_json(self, path: Path, obj: Mapping[str, Any]) -> None:
        """Write metadata JSON directly to its destination path."""
        text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)


class _DirectoryCrawlerCache(_FilesystemCrawlerCache):
    pass


class _WebCrawlerCache(_FilesystemCrawlerCache):
    pass


class _CloudflareCrawlerCache(_FilesystemCrawlerCache):
    pass


def _map_ordered(
    items: Iterable[TInput],
    *,
    max_workers: int,
    fn: Callable[[TInput], TOutput],
) -> Iterator[TOutput]:
    assert max_workers >= 1
    iterator = iter(items)
    if max_workers == 1:
        for item in iterator:
            yield fn(item)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending: deque[Any] = deque()
        while len(pending) < max_workers:
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(executor.submit(fn, item))

        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(fn, item))


class BaseCrawler(ABC):
    """Abstract base class for crawlers.

    A crawler discovers source documents for a `CrawlScope` and converts them
    into `MarkdownDocument` objects ready for chunking and ingestion. All
    crawlers expose the same four public methods, so a scope and the
    surrounding workflow can be reused across backends.

    Subclasses provide a concrete discovery and fetching strategy:

    - `DirectoryCrawler`: walk local files and directories.
    - `WebCrawler`: fetch pages directly over HTTP and follow links.
    - `CloudflareCrawler`: delegate discovery and rendering to Cloudflare's
      Browser Rendering API.

    Attributes
    ----------
    max_workers
        Number of worker threads used to fetch and convert sources
        concurrently in `markdown_documents()`.
    """

    max_workers: int

    @abstractmethod
    def origins(
        self,
        scope: CrawlScope,
        *,
        progress: bool = True,
        cache_force_refresh: bool = False,
    ) -> Iterator[str]:
        """Discover the origins matched by a scope.

        Parameters
        ----------
        scope
            The `CrawlScope` describing what to crawl.
        progress
            Whether to display crawl progress, when the backend supports it.
        cache_force_refresh
            When `True`, bypass any cached discovery results and re-crawl.

        Returns
        -------
        Iterator[str]
            A lazy iterator of source origins, in discovery order. Each origin
            is unique within a single call.
        """
        pass

    @abstractmethod
    def fetch_raw(
        self,
        origin: str,
        *,
        cache_force_refresh: bool = False,
    ) -> FetchedSource:
        """Fetch the raw body and metadata for one origin.

        Parameters
        ----------
        origin
            The origin to fetch, as produced by `origins()`.
        cache_force_refresh
            When `True`, bypass any cached body and re-fetch from the source.

        Returns
        -------
        FetchedSource
            The fetched source, with its body path and metadata. The raw body
            is not yet converted to Markdown.
        """
        pass

    def fetch_markdown(
        self,
        origin: str,
        *,
        convert: Callable[[FetchedSource], MarkdownDocument] | None = None,
        cache_force_refresh: bool = False,
    ) -> MarkdownDocument:
        """Fetch one origin and convert it to a Markdown document.

        Parameters
        ----------
        origin
            The origin to fetch and convert.
        convert
            Optional callable that turns a `FetchedSource` into a
            `MarkdownDocument`. When omitted, the crawler's default conversion
            is used. Use this to apply custom cleanup; keep chunking in
            `store.ingest(prepare=...)` rather than in the converter.
        cache_force_refresh
            When `True`, bypass any cached body and re-fetch from the source.

        Returns
        -------
        MarkdownDocument
            The converted document for `origin`.
        """
        source = self.fetch_raw(origin, cache_force_refresh=cache_force_refresh)
        converter = convert or self._default_convert
        return converter(source)

    def _fetch_markdown_after_origin_discovery(
        self,
        origin: str,
        *,
        convert: Callable[[FetchedSource], MarkdownDocument] | None = None,
    ) -> MarkdownDocument:
        source = self._fetch_raw_after_origin_discovery(origin)
        converter = convert or self._default_convert
        return converter(source)

    def _fetch_raw_after_origin_discovery(self, origin: str) -> FetchedSource:
        return self.fetch_raw(origin, cache_force_refresh=False)

    def markdown_documents(
        self,
        scope: CrawlScope,
        *,
        convert: Callable[[FetchedSource], MarkdownDocument] | None = None,
        progress: bool = True,
        cache_force_refresh: bool = False,
    ) -> Iterator[MarkdownDocument]:
        """Discover and convert all sources matched by a scope.

        This is the primary entry point for crawling. It combines `origins()`
        and `fetch_markdown()`, fetching and converting sources concurrently
        using up to `max_workers` threads while preserving discovery order. The
        result is intended to be passed directly to `store.ingest()`.

        Parameters
        ----------
        scope
            The `CrawlScope` describing what to crawl.
        convert
            Optional callable that turns a `FetchedSource` into a
            `MarkdownDocument`. When omitted, the crawler's default conversion
            is used.
        progress
            Whether to display crawl progress, when the backend supports it.
        cache_force_refresh
            When `True`, bypass cached discovery and bodies and re-crawl.

        Returns
        -------
        Iterator[MarkdownDocument]
            A lazy iterator of converted documents, in discovery order.
        """
        origins = self.origins(
            scope,
            progress=progress,
            cache_force_refresh=cache_force_refresh,
        )
        yield from _map_ordered(
            origins,
            max_workers=self.max_workers,
            fn=lambda origin: self._fetch_markdown_after_origin_discovery(
                origin,
                convert=convert,
            ),
        )

    def _default_convert(self, source: FetchedSource) -> MarkdownDocument:
        raise NotImplementedError


class DirectoryCrawler(BaseCrawler):
    """Crawl local files and optionally cache converted markdown.

    Use a `DirectoryCrawler` for local Markdown, notebooks, PDFs, text files,
    and other formats supported by `read_as_markdown()`. Directory traversal
    always reads the current filesystem state. The cache stores converted
    Markdown per file origin and is reused only when the current file hash and
    modification time still match the cached metadata. When the cache directory
    lives inside a crawled root, the crawler skips its own cache files.

    Parameters
    ----------
    cache_dir
        Where to cache converted Markdown. `None` (default) disables caching.
        `True` uses `.raghilda/cache/directory` under the current working
        directory. A string or `Path` uses that location.
    max_workers
        Number of worker threads used to convert files concurrently in
        `markdown_documents()`. Must be at least 1. Default is 1.

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.crawl import CrawlScope, DirectoryCrawler

    crawler = DirectoryCrawler(cache_dir=True, max_workers=4)
    scope = CrawlScope(
        roots=["docs"],
        depth=3,
        include_patterns=[r".*\\.(md|qmd|ipynb|pdf)$"],
    )

    for document in crawler.markdown_documents(scope):
        print(document.origin)
    ```
    """

    def __init__(
        self,
        *,
        cache_dir: bool | str | Path | None = None,
        max_workers: int = 1,
    ) -> None:
        assert max_workers >= 1
        self.cache_dir = _resolve_cache_dir(
            cache_dir,
            backend_name="directory",
            default_factory=lambda: None,
        )
        self.max_workers = max_workers
        self._cache = _DirectoryCrawlerCache(self.cache_dir)

    def origins(
        self,
        scope: CrawlScope,
        *,
        progress: bool = True,
        cache_force_refresh: bool = False,
    ) -> Iterator[str]:
        """Discover local file origins matched by a scope.

        Walks each root in the scope, descending up to `scope.depth`
        directory levels, and yields a `file://` URI for every file that
        passes the scope's pattern and type filters. Symlinked directories and
        the crawler's own cache directory are skipped. Directory traversal
        always reflects the current filesystem state, so `progress` and
        `cache_force_refresh` have no effect here.

        Parameters
        ----------
        scope
            The `CrawlScope` describing what to crawl.
            `roots` may be directories or individual files.
        progress
            Unused; accepted for interface compatibility.
        cache_force_refresh
            Unused; accepted for interface compatibility.

        Returns
        -------
        Iterator[str]
            A lazy iterator of unique `file://` origins, in sorted traversal
            order.
        """
        del progress, cache_force_refresh
        resolved_scope = _resolve_crawl_scope(scope)
        if resolved_scope.limit == 0:
            return
        cache_root = self.cache_dir.resolve() if self.cache_dir is not None else None
        count = 0
        yielded_origins: set[str] = set()
        for root in resolved_scope.roots:
            path = _to_directory_path(root)
            assert path.exists(), f"Root does not exist: {path}"
            if path.is_file():
                resolved_path = path.resolve()
                if cache_root is not None and resolved_path.is_relative_to(cache_root):
                    continue
                origin = resolved_path.as_uri()
                if origin in yielded_origins:
                    continue
                if self._include_path(
                    path,
                    origin,
                    include_matchers=resolved_scope.include_matchers,
                    exclude_matchers=resolved_scope.exclude_matchers,
                    include_types=resolved_scope.include_types,
                    exclude_types=resolved_scope.exclude_types,
                ):
                    yielded_origins.add(origin)
                    yield origin
                    count += 1
                    if (
                        resolved_scope.limit is not None
                        and count >= resolved_scope.limit
                    ):
                        return
                continue
            for file_path in _iter_directory_files(
                path,
                max_depth=resolved_scope.depth,
            ):
                resolved_file_path = file_path.resolve()
                if cache_root is not None and resolved_file_path.is_relative_to(
                    cache_root
                ):
                    continue
                origin = resolved_file_path.as_uri()
                if origin in yielded_origins:
                    continue
                if not self._include_path(
                    file_path,
                    origin,
                    include_matchers=resolved_scope.include_matchers,
                    exclude_matchers=resolved_scope.exclude_matchers,
                    include_types=resolved_scope.include_types,
                    exclude_types=resolved_scope.exclude_types,
                ):
                    continue
                yielded_origins.add(origin)
                yield origin
                count += 1
                if resolved_scope.limit is not None and count >= resolved_scope.limit:
                    return

    def fetch_raw(
        self,
        origin: str,
        *,
        cache_force_refresh: bool = False,
    ) -> FetchedSource:
        """Read one local file origin and return its source metadata.

        The returned `FetchedSource` points at the
        file on disk and records its size, modification time, content hash, and
        detected type label. When caching is enabled and the file is unchanged
        since the last conversion, the cached Markdown path is attached so that
        conversion can be skipped.

        Parameters
        ----------
        origin
            A `file://` URI (or local path) identifying an existing file.
        cache_force_refresh
            When `True`, ignore any cached Markdown for this file so it will
            be reconverted.

        Returns
        -------
        FetchedSource
            The source description for the file at `origin`.
        """
        path = _path_from_file_origin(origin).resolve()
        assert path.is_file(), f"File origin must exist: {origin}"
        canonical_origin = path.as_uri()
        content_type = mimetypes.guess_type(path.name)[0]
        type_label = _detect_type_label(path=path, content_type=content_type)
        source_hash = _sha256_path(path)
        markdown_path: Path | None = None
        if self.cache_dir is not None and not cache_force_refresh:
            cached_entry = self._cache.fetch(canonical_origin)
            if cached_entry is not None:
                cached_markdown_path, cached_meta = cached_entry
                if (
                    cached_markdown_path is not None
                    and cached_meta is not None
                    and (
                        cached_meta.get("source_hash") == source_hash
                        and cached_meta.get("mtime_ns") == path.stat().st_mtime_ns
                    )
                ):
                    markdown_path = cached_markdown_path
        return FetchedSource(
            origin=canonical_origin,
            resolved_origin=canonical_origin,
            content_type=content_type,
            status_code=None,
            metadata={
                "mtime_ns": path.stat().st_mtime_ns,
                "size": path.stat().st_size,
                "source_hash": source_hash,
                "type_label": type_label,
            },
            fetched_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
            body_path=path,
            markdown_path=markdown_path,
        )

    def markdown_documents(
        self,
        scope: CrawlScope,
        *,
        convert: Callable[[FetchedSource], MarkdownDocument] | None = None,
        progress: bool = True,
        cache_force_refresh: bool = False,
    ) -> Iterator[MarkdownDocument]:
        """Discover and convert all local files matched by a scope.

        Combines `origins()` and `fetch_markdown()`, converting files
        concurrently using up to `max_workers` threads while preserving
        traversal order. When caching is enabled, converted Markdown is reused
        for files whose hash and modification time are unchanged.

        Parameters
        ----------
        scope
            The `CrawlScope` describing what to crawl.
        convert
            Optional callable that turns a `FetchedSource` into a
            `MarkdownDocument`. When omitted, the default conversion is used.
        progress
            Unused; accepted for interface compatibility.
        cache_force_refresh
            When `True`, reconvert files even when cached Markdown is present.

        Returns
        -------
        Iterator[MarkdownDocument]
            A lazy iterator of converted documents, in traversal order.
        """
        origins = self.origins(
            scope,
            progress=progress,
            cache_force_refresh=cache_force_refresh,
        )
        yield from _map_ordered(
            origins,
            max_workers=self.max_workers,
            fn=lambda origin: self.fetch_markdown(
                origin,
                convert=convert,
                cache_force_refresh=cache_force_refresh,
            ),
        )

    def _default_convert(self, source: FetchedSource) -> MarkdownDocument:
        if source.markdown_path is not None and source.markdown_path.exists():
            markdown = source.markdown_path.read_text(encoding="utf-8")
            return MarkdownDocument(origin=source.origin, content=markdown)

        type_label = (source.metadata or {}).get("type_label")
        if type_label == "markdown":
            markdown = source.body_path.read_text(encoding="utf-8")
        else:
            markdown = _convert_to_markdown(str(source.body_path))

        if self.cache_dir is not None:
            self._cache.upsert(
                source.origin,
                content=markdown,
                metadata={
                    "origin": source.origin,
                    "mtime_ns": (source.metadata or {}).get("mtime_ns"),
                    "source_hash": (source.metadata or {}).get("source_hash"),
                },
                content_ext=".md",
            )

        return MarkdownDocument(origin=source.origin, content=markdown)

    def _include_path(
        self,
        path: Path,
        origin: str,
        *,
        include_matchers: Sequence[Callable[[str], bool]],
        exclude_matchers: Sequence[Callable[[str], bool]],
        include_types: set[str],
        exclude_types: set[str],
    ) -> bool:
        if not _matches_patterns(
            origin,
            include_matchers=include_matchers,
            exclude_matchers=exclude_matchers,
        ):
            return False
        if not include_types and not exclude_types:
            return True
        label = _detect_type_label(
            path=path, content_type=mimetypes.guess_type(path.name)[0]
        )
        return _matches_types(
            label,
            include_types=include_types,
            exclude_types=exclude_types,
        )


class WebCrawler(BaseCrawler):
    """Crawl a website by fetching pages directly over HTTP.

    A `WebCrawler` starts from one or more root URLs, fetches each page with
    `requests`, follows discovered links up to `scope.depth`, and yields
    matching pages as `MarkdownDocument` objects. Link following is constrained
    by the scope's patterns, types, and the `include_external_links` /
    `include_subdomains` flags.

    Fetched response bodies are cached on disk. When `cache_stale_after` is
    set, fresh cached responses are reused and stale ones are revalidated with
    `ETag` / `Last-Modified` headers when the server provides them. Pass
    `cache_force_refresh=True` to any method to bypass the cache for a run.

    Parameters
    ----------
    session
        A `requests.Session` to use for requests. When omitted, a new session
        is created. A caller-supplied session also scopes the cache so entries
        are not shared across sessions.
    cache_dir
        Where to cache fetched bodies. `None` (default) uses a temporary
        directory. `True` uses `.raghilda/cache/web` under the current working
        directory. A string or `Path` uses that location.
    cache_stale_after
        How long a cached body stays fresh before it must be revalidated. When
        `None` (default), cached bodies are always considered fresh.
    max_workers
        Number of worker threads used to fetch pages concurrently. Must be at
        least 1. Default is 1.

    Examples
    --------
    ```{python}
    #| eval: false
    from datetime import timedelta

    from raghilda.crawl import CrawlScope, WebCrawler

    crawler = WebCrawler(cache_dir=True, cache_stale_after=timedelta(days=1))
    scope = CrawlScope(
        roots=["https://quarto.org/docs/guide/"],
        depth=2,
        include_patterns=[r"^https://quarto\\.org/docs/guide/"],
        include_types=["html"],
    )

    for document in crawler.markdown_documents(scope):
        print(document.origin)
    ```
    """

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        cache_dir: bool | str | Path | None = None,
        cache_stale_after: timedelta | None = None,
        max_workers: int = 1,
    ) -> None:
        assert max_workers >= 1
        self.session = requests.Session() if session is None else session
        self._cache_context = None if session is None else f"session:{id(self.session)}"
        self.cache_dir = _resolve_cache_dir(
            cache_dir,
            backend_name="web",
            default_factory=lambda: Path(
                tempfile.mkdtemp(prefix="raghilda-web-cache-")
            ),
        )
        self.cache_stale_after = cache_stale_after
        self.max_workers = max_workers
        self._cache = _WebCrawlerCache(self.cache_dir)

    def origins(
        self,
        scope: CrawlScope,
        *,
        progress: bool = True,
        cache_force_refresh: bool = False,
    ) -> Iterator[str]:
        """Discover web origins reachable from the scope's roots.

        Performs a breadth-first crawl: each root is fetched, its links are
        extracted and canonicalized, and the frontier expands one level per
        `scope.depth` until the depth or `scope.limit` is reached. Origins
        outside the root scope are dropped unless `include_external_links` or
        `include_subdomains` allow them. Only origins passing the scope's
        pattern and type filters are yielded.

        Parameters
        ----------
        scope
            The `CrawlScope` describing what to crawl.
            `roots` must be `http` or `https` URLs.
        progress
            Unused; accepted for interface compatibility.
        cache_force_refresh
            When `True`, re-fetch pages instead of using cached bodies while
            discovering links.

        Returns
        -------
        Iterator[str]
            A lazy iterator of unique canonical URLs, in crawl order.
        """
        del progress
        resolved_scope = _resolve_crawl_scope(scope)
        if resolved_scope.limit == 0:
            return
        visited: set[tuple[str, WebOriginKey, str]] = set()
        yielded_origins: set[str] = set()
        yielded = 0
        frontier: list[tuple[str, WebOriginKey, str]] = []

        for root in resolved_scope.roots:
            canonical_root = _canonicalize_web_url(str(root))
            assert canonical_root is not None
            parsed = urlparse(canonical_root)
            assert parsed.scheme in {"http", "https"}
            root_host = parsed.hostname or ""
            frontier.append(
                (canonical_root, _web_origin_key(canonical_root), root_host)
            )

        current_depth = 0
        while frontier:
            batch: list[tuple[str, WebOriginKey, str]] = []
            for origin, scope_origin, root_host in frontier:
                visit_key = (origin, scope_origin, root_host)
                if visit_key in visited:
                    continue
                if not self._allow_origin(
                    origin,
                    scope_origin,
                    root_host,
                    include_external_links=resolved_scope.include_external_links,
                    include_subdomains=resolved_scope.include_subdomains,
                ):
                    continue
                if any(matcher(origin) for matcher in resolved_scope.exclude_matchers):
                    continue
                visited.add(visit_key)
                batch.append((origin, scope_origin, root_host))

            next_frontier: list[tuple[str, WebOriginKey, str]] = []
            offset = 0
            while offset < len(batch):
                remaining = (
                    None
                    if resolved_scope.limit is None
                    else resolved_scope.limit - yielded
                )
                if remaining == 0:
                    return
                chunk_size = len(batch) - offset
                if remaining is not None:
                    chunk_size = min(chunk_size, remaining)
                window = batch[offset : offset + chunk_size]
                fetched_sources = _map_ordered(
                    window,
                    max_workers=min(self.max_workers, len(window)),
                    fn=lambda item: (
                        item,
                        self.fetch_raw(
                            item[0],
                            cache_force_refresh=cache_force_refresh,
                        ),
                    ),
                )
                for (origin, scope_origin, root_host), source in fetched_sources:
                    type_label = (source.metadata or {}).get("type_label")
                    matches_patterns = _matches_patterns(
                        origin,
                        include_matchers=resolved_scope.include_matchers,
                        exclude_matchers=resolved_scope.exclude_matchers,
                    )
                    matches_types = _matches_types(
                        type_label,
                        include_types=resolved_scope.include_types,
                        exclude_types=resolved_scope.exclude_types,
                    )
                    if (
                        matches_patterns
                        and matches_types
                        and origin not in yielded_origins
                    ):
                        yield origin
                        yielded_origins.add(origin)
                        yielded += 1
                        if (
                            resolved_scope.limit is not None
                            and yielded >= resolved_scope.limit
                        ):
                            return
                    if current_depth >= resolved_scope.depth:
                        continue

                    text = _read_text(source.body_path)
                    resolved_origin = source.resolved_origin or origin
                    resolved_origin_key = _web_origin_key(resolved_origin)
                    origin_key = _web_origin_key(origin)
                    child_root_host = root_host
                    if (
                        resolved_scope.include_subdomains
                        and resolved_origin_key == origin_key
                    ):
                        child_scope_origin = scope_origin
                    else:
                        child_scope_origin = resolved_origin_key
                        child_root_host = (
                            urlparse(resolved_origin).hostname or root_host
                        )
                    for link in sorted(_extract_links(text)):
                        canonical = _canonicalize_web_url(link, base=resolved_origin)
                        if canonical is None:
                            continue
                        parsed = urlparse(canonical)
                        if parsed.scheme not in {"http", "https"}:
                            continue
                        next_frontier.append(
                            (canonical, child_scope_origin, child_root_host)
                        )
                offset += chunk_size
            frontier = next_frontier
            current_depth += 1

    def fetch_raw(
        self,
        origin: str,
        *,
        cache_force_refresh: bool = False,
    ) -> FetchedSource:
        """Fetch one URL over HTTP and return its source metadata.

        A fresh cached body is returned without a network request. A stale
        cached body is revalidated with `If-None-Match` / `If-Modified-Since`
        headers; on a `304 Not Modified` the cached body is reused. Otherwise
        the body is downloaded, cached, and its content type and type label are
        recorded on the returned source.

        Parameters
        ----------
        origin
            The URL to fetch. It is canonicalized before use and must be an
            `http` or `https` URL.
        cache_force_refresh
            When `True`, ignore any cached body and re-fetch from the server.

        Returns
        -------
        FetchedSource
            The fetched source, with its cached body path and metadata.
        """
        canonical_origin = _canonicalize_web_url(origin)
        assert canonical_origin is not None
        parsed = urlparse(canonical_origin)
        assert parsed.scheme in {"http", "https"}

        cached_entry = self._cache.fetch(canonical_origin)
        body_path: Path | None = None
        cached_meta: dict[str, Any] | None = None
        if cached_entry is not None:
            body_path, cached_meta = cached_entry
        has_cache = (
            body_path is not None
            and cached_meta is not None
            and self._cache_context_matches(cached_meta)
        )
        now = _utcnow()

        if has_cache and not cache_force_refresh:
            assert cached_meta is not None
            assert body_path is not None
            if self._is_fresh(cached_meta, now):
                return self._source_from_meta(cached_meta, body_path=body_path)

        headers: dict[str, str] = {}
        if has_cache and not cache_force_refresh:
            assert cached_meta is not None
            etag = cached_meta.get("etag")
            last_modified = cached_meta.get("last_modified")
            if etag:
                headers["If-None-Match"] = etag
            if last_modified:
                headers["If-Modified-Since"] = last_modified

        response = self.session.get(canonical_origin, headers=headers, timeout=30.0)
        if response.status_code == 304 and has_cache:
            assert cached_meta is not None
            assert body_path is not None
            cached_meta["revalidated_at"] = now.isoformat()
            cached_entry = self._cache.upsert(
                canonical_origin,
                content=body_path,
                metadata=cached_meta,
                content_ext=None,
            )
            assert cached_entry is not None
            body_path, cached_meta = cached_entry
            assert body_path is not None
            assert cached_meta is not None
            return self._source_from_meta(cached_meta, body_path=body_path)

        response.raise_for_status()
        content_type = response.headers.get("Content-Type")
        resolved_origin = (
            _canonicalize_web_url(response.url, base=canonical_origin) or response.url
        )
        type_label = _detect_type_label(
            path=_type_hint_path(canonical_origin, content_type=content_type),
            content_type=content_type,
        )
        meta = {
            "origin": canonical_origin,
            "resolved_origin": resolved_origin,
            "content_type": content_type,
            "status_code": response.status_code,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "type_label": type_label,
            "fetched_at": now.isoformat(),
            "revalidated_at": None,
            "cache_context": self._cache_context,
        }
        cached_entry = self._cache.upsert(
            canonical_origin,
            content=response.content,
            metadata=meta,
            content_ext=_known_body_suffix(
                canonical_origin,
                content_type=content_type,
            ),
        )
        assert cached_entry is not None
        body_path, meta = cached_entry
        assert body_path is not None
        assert meta is not None
        actual_type_label = _detect_type_label(
            path=body_path,
            content_type=content_type,
        )
        if actual_type_label != meta.get("type_label"):
            meta["type_label"] = actual_type_label
            cached_entry = self._cache.upsert(
                canonical_origin,
                content=body_path,
                metadata=meta,
                content_ext=None,
            )
            assert cached_entry is not None
            body_path, meta = cached_entry
            assert body_path is not None
            assert meta is not None
        return self._source_from_meta(meta, body_path=body_path)

    def _fetch_raw_after_origin_discovery(self, origin: str) -> FetchedSource:
        canonical_origin = _canonicalize_web_url(origin)
        assert canonical_origin is not None
        cached_entry = self._cache.fetch(canonical_origin)
        assert cached_entry is not None
        body_path, cached_meta = cached_entry
        assert body_path is not None
        assert cached_meta is not None
        return self._source_from_meta(cached_meta, body_path=body_path)

    def _default_convert(self, source: FetchedSource) -> MarkdownDocument:
        type_label = (source.metadata or {}).get("type_label")
        if type_label == "markdown":
            markdown = _read_text(source.body_path)
        else:
            path_for_conversion = source.body_path
            if source.body_path.suffix == "":
                suffix = _body_suffix(
                    source.origin,
                    content_type=source.content_type,
                )
                with tempfile.NamedTemporaryFile(
                    prefix="raghilda-convert-",
                    suffix=suffix,
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    temporary_file.write(source.body_path.read_bytes())
                try:
                    path_for_conversion = temporary_path
                    markdown = _convert_to_markdown(str(path_for_conversion))
                finally:
                    temporary_path.unlink(missing_ok=True)
            else:
                markdown = _convert_to_markdown(str(path_for_conversion))
        return MarkdownDocument(origin=source.origin, content=markdown)

    def _source_from_meta(
        self,
        meta: dict[str, Any],
        *,
        body_path: Path,
    ) -> FetchedSource:
        return FetchedSource(
            origin=meta["origin"],
            resolved_origin=meta.get("resolved_origin"),
            content_type=meta.get("content_type"),
            status_code=meta.get("status_code"),
            metadata={
                "etag": meta.get("etag"),
                "last_modified": meta.get("last_modified"),
                "type_label": meta.get("type_label"),
            },
            fetched_at=_parse_datetime(meta.get("fetched_at")),
            revalidated_at=_parse_datetime(meta.get("revalidated_at")),
            body_path=body_path,
        )

    def _is_fresh(self, cached_meta: dict[str, Any], now: datetime) -> bool:
        if self.cache_stale_after is None:
            return True
        timestamps = [
            _parse_datetime(cached_meta.get("fetched_at")),
            _parse_datetime(cached_meta.get("revalidated_at")),
        ]
        freshest_cache_time = max(
            (timestamp for timestamp in timestamps if timestamp is not None),
            default=None,
        )
        if freshest_cache_time is None:
            return False
        return now - freshest_cache_time <= self.cache_stale_after

    def _cache_context_matches(self, cached_meta: dict[str, Any]) -> bool:
        return cached_meta.get("cache_context") == self._cache_context

    def _allow_origin(
        self,
        origin: str,
        scope_origin: WebOriginKey,
        root_host: str,
        *,
        include_external_links: bool,
        include_subdomains: bool,
    ) -> bool:
        return _allow_web_origin(
            origin,
            scope_origin,
            root_host,
            include_external_links=include_external_links,
            include_subdomains=include_subdomains,
        )


class CloudflareCrawler(BaseCrawler):
    """Crawl a website using Cloudflare's Browser Rendering API.

    A `CloudflareCrawler` delegates page discovery, JavaScript rendering, and
    Markdown extraction to Cloudflare. It submits a crawl job, polls until the
    job completes, retrieves the rendered Markdown record for each discovered
    page, and yields them as `MarkdownDocument` objects. This is useful for
    single-page applications and other sites whose content only appears after
    client-side rendering.

    For Cloudflare crawls, `include_patterns` and `exclude_patterns` accept the
    same glob strings (such as `"https://example.com/docs/**"`) or compiled
    `re.Pattern` objects as the other crawlers. Glob strings are forwarded to
    Cloudflare's crawl request; regex patterns are enforced locally on the
    returned records. `include_external_links` / `include_subdomains` are
    passed through to the crawl request. Cached entries are invalidated
    automatically when `render`, `source`, or `modified_since` change between
    runs.

    Parameters
    ----------
    account_id
        The Cloudflare account ID that owns the Browser Rendering subscription.
    api_token
        A Cloudflare API token with Browser Rendering permissions. Sent as a
        bearer token on each request.
    cache_dir
        Where to cache crawl results and page records. `None` (default) uses a
        temporary directory. `True` uses `.raghilda/cache/cloudflare` under the
        current working directory. A string or `Path` uses that location.
    session
        A `requests.Session` to use for API calls. When omitted, a new session
        is created.
    source
        How Cloudflare discovers pages: `"all"` (default) combines available
        methods, `"sitemap"` reads the site's sitemap, and `"crawl"` follows
        links from the rendered DOM.
    render
        Whether Cloudflare executes JavaScript before extracting content.
        Defaults to `True`. Set to `False` for server-rendered sites to
        reduce crawl time and API usage.
    cache_stale_after
        How long cached results stay fresh. When stale, the crawler requests
        updated content with a `maxAge` hint. When `None` (default), cached
        entries never expire.
    modified_since
        Restrict the crawl to pages modified after this Unix timestamp, for
        incremental refreshes. When `None` (default), all pages are eligible.
    poll_interval
        Seconds to wait between job-status polls. Default is `5.0`.
    max_poll_attempts
        Maximum number of status polls before a `TimeoutError` is raised.
        Default is `60` (a five-minute window at the default interval).
    max_workers
        Number of worker threads used to materialize page records
        concurrently. Must be at least 1. Default is 1.
    base_url
        Base URL of the Cloudflare API. Defaults to the public endpoint and is
        rarely overridden outside of testing.

    Examples
    --------
    ```{python}
    #| eval: false
    import os

    from raghilda.crawl import CloudflareCrawler, CrawlScope

    crawler = CloudflareCrawler(
        account_id=os.environ["CLOUDFLARE_ACCOUNT_ID"],
        api_token=os.environ["CLOUDFLARE_API_TOKEN"],
        cache_dir=True,
        render=True,
    )
    scope = CrawlScope(
        roots=["https://example.com/docs/"],
        depth=2,
        include_patterns=["https://example.com/docs/**"],
    )

    for document in crawler.markdown_documents(scope):
        print(document.origin)
    ```
    """

    def __init__(
        self,
        *,
        account_id: str,
        api_token: str,
        cache_dir: bool | str | Path | None = None,
        session: requests.Session | Any | None = None,
        source: str = "all",
        render: bool = True,
        cache_stale_after: timedelta | None = None,
        modified_since: int | None = None,
        poll_interval: float = 5.0,
        max_poll_attempts: int = 60,
        max_workers: int = 1,
        base_url: str = "https://api.cloudflare.com/client/v4",
    ) -> None:
        assert max_workers >= 1
        self.account_id = account_id
        self.api_token = api_token
        self.cache_dir = _resolve_cache_dir(
            cache_dir,
            backend_name="cloudflare",
            default_factory=lambda: Path(
                tempfile.mkdtemp(prefix="raghilda-cloudflare-cache-")
            ),
        )
        self.session = session or requests.Session()
        self.source = source
        self.render = render
        self.cache_stale_after = cache_stale_after
        self.modified_since = modified_since
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts
        self.max_workers = max_workers
        self.base_url = base_url.rstrip("/")
        self._records: dict[str, _CloudflareRecordCacheEntry] = {}
        self._roots: dict[tuple[Any, ...], _CloudflareRootCacheEntry] = {}
        self._cache = _CloudflareCrawlerCache(self.cache_dir)

    def origins(
        self,
        scope: CrawlScope,
        *,
        progress: bool = True,
        cache_force_refresh: bool = False,
    ) -> Iterator[str]:
        """Discover origins by running a Cloudflare crawl for each root.

        Submits a crawl job for each root in the scope and yields the canonical
        URL of every completed page record that passes the scope's pattern and
        type filters. Results are cached per root so repeated runs can avoid new
        Cloudflare API calls while the cache is fresh.

        Parameters
        ----------
        scope
            The `CrawlScope` describing what to crawl. `roots` must be `http`
            or `https` URLs.
        progress
            Unused; accepted for interface compatibility.
        cache_force_refresh
            When `True`, submit a fresh crawl job instead of reusing cached
            results.

        Returns
        -------
        Iterator[str]
            A lazy iterator of unique canonical URLs returned by Cloudflare.
        """
        del progress
        resolved_scope = _resolve_crawl_scope(scope)
        yielded = 0
        yielded_origins: set[str] = set()
        crawled_roots: set[str] = set()
        for root in resolved_scope.roots:
            if resolved_scope.limit is not None and yielded >= resolved_scope.limit:
                return
            canonical_root = _canonicalize_web_url(str(root))
            assert canonical_root is not None
            if canonical_root in crawled_roots:
                continue
            crawled_roots.add(canonical_root)
            remaining = (
                None if resolved_scope.limit is None else resolved_scope.limit - yielded
            )
            root_limit = remaining if not yielded_origins else None
            records = self._crawl_root(
                canonical_root,
                cache_force_refresh=cache_force_refresh,
                depth=resolved_scope.depth,
                include_patterns=resolved_scope.include_patterns,
                exclude_patterns=resolved_scope.exclude_patterns,
                include_external_links=resolved_scope.include_external_links,
                include_subdomains=resolved_scope.include_subdomains,
                limit=root_limit,
            )
            for record in records:
                origin = record["url"]
                if origin in yielded_origins:
                    continue
                label = _detect_type_label(
                    path=None,
                    content_type="text/markdown",
                )
                if not _matches_types(
                    label,
                    include_types=resolved_scope.include_types,
                    exclude_types=resolved_scope.exclude_types,
                ):
                    continue
                yielded_origins.add(origin)
                yield origin
                yielded += 1
                if resolved_scope.limit is not None and yielded >= resolved_scope.limit:
                    return

    def fetch_raw(
        self,
        origin: str,
        *,
        cache_force_refresh: bool = False,
    ) -> FetchedSource:
        """Fetch the rendered Markdown record for one Cloudflare origin.

        Returns a fresh in-memory or on-disk cached record when available.
        Otherwise it runs a single-URL Cloudflare crawl to produce the record.
        The returned source's body path points at the rendered Markdown, which
        is already suitable for conversion without further processing.

        Parameters
        ----------
        origin
            The URL to fetch. It is canonicalized before use.
        cache_force_refresh
            When `True`, ignore cached records and request a fresh crawl.

        Returns
        -------
        FetchedSource
            The fetched source for `origin`, with its rendered Markdown body
            path and metadata.

        Raises
        ------
        ValueError
            If Cloudflare does not return a record for `origin`.
        """
        canonical_origin = _canonicalize_web_url(origin)
        assert canonical_origin is not None
        record_entry = (
            None if cache_force_refresh else self._records.get(canonical_origin)
        )
        if record_entry is not None and not self._cloudflare_cache_is_fresh(
            record_entry.fetched_at
        ):
            record_entry = None
        if record_entry is None and not cache_force_refresh:
            record_entry = self._load_record_cache_entry(canonical_origin)
            if record_entry is not None:
                self._records[canonical_origin] = record_entry
        if record_entry is None or cache_force_refresh:
            records = self._crawl_root(
                canonical_origin,
                cache_force_refresh=cache_force_refresh,
                depth=0,
                limit=1,
                apply_patterns=False,
                include_external_links=False,
                include_subdomains=False,
            )
            record = next(
                (item for item in records if item["url"] == canonical_origin),
                None,
            )
            if record is None and len(records) == 1:
                record = records[0]
            if record is None:
                raise ValueError(f"Cloudflare crawl did not return record for {origin}")
            record_entry = self._records.get(record["url"])
            assert record_entry is not None
            self._records[canonical_origin] = record_entry

        assert record_entry is not None
        return self._source_from_record_entry(canonical_origin, record_entry)

    def _fetch_raw_after_origin_discovery(self, origin: str) -> FetchedSource:
        canonical_origin = _canonicalize_web_url(origin)
        assert canonical_origin is not None
        record_entry = self._records.get(canonical_origin)
        if record_entry is None:
            record_entry = self._load_record_cache_entry(canonical_origin)
            assert record_entry is not None
            self._records[canonical_origin] = record_entry
        return self._source_from_record_entry(canonical_origin, record_entry)

    def _source_from_record_entry(
        self,
        canonical_origin: str,
        record_entry: _CloudflareRecordCacheEntry,
    ) -> FetchedSource:
        content_path, _ = self._store_record_cache_entry(
            canonical_origin,
            record=record_entry.record,
            fetched_at=record_entry.fetched_at,
        )
        assert content_path is not None
        record = record_entry.record
        return FetchedSource(
            origin=canonical_origin,
            resolved_origin=record.get("metadata", {}).get("url", canonical_origin),
            content_type="text/markdown",
            status_code=record.get("metadata", {}).get("status"),
            metadata={
                "crawler_status": record.get("status"),
                "title": record.get("metadata", {}).get("title"),
                "type_label": "markdown",
            },
            fetched_at=record_entry.fetched_at,
            body_path=content_path,
            markdown_path=content_path,
        )

    def _default_convert(self, source: FetchedSource) -> MarkdownDocument:
        markdown = source.body_path.read_text(encoding="utf-8")
        return MarkdownDocument(origin=source.origin, content=markdown)

    def _crawl_root(
        self,
        root: str,
        *,
        cache_force_refresh: bool,
        depth: int | None = None,
        include_patterns: Sequence[PatternInput] | None = None,
        exclude_patterns: Sequence[PatternInput] | None = None,
        include_external_links: bool,
        include_subdomains: bool,
        limit: int | None = None,
        apply_patterns: bool = True,
    ) -> list[dict[str, Any]]:
        resolved_depth = _DEFAULT_CRAWL_DEPTH if depth is None else depth
        resolved_include_patterns = list(include_patterns or [])
        resolved_exclude_patterns = list(exclude_patterns or [])
        resolved_limit = limit
        cache_key = (
            root,
            resolved_depth,
            resolved_limit,
            apply_patterns,
            tuple(_pattern_cache_token(p) for p in resolved_include_patterns),
            tuple(_pattern_cache_token(p) for p in resolved_exclude_patterns),
            include_external_links,
            include_subdomains,
        )
        cached_entry = self._roots.get(cache_key)
        if (
            not cache_force_refresh
            and cached_entry is not None
            and self._cloudflare_cache_is_fresh(cached_entry.fetched_at)
        ):
            return cached_entry.records
        if not cache_force_refresh and apply_patterns:
            cached_entry = self._load_root_cache_entry(cache_key)
            if cached_entry is not None:
                self._roots[cache_key] = cached_entry
                return cached_entry.records

        endpoint = f"{self.base_url}/accounts/{self.account_id}/browser-rendering/crawl"
        payload = self._crawl_payload(
            root,
            depth=resolved_depth,
            limit=resolved_limit,
            include_patterns=resolved_include_patterns,
            exclude_patterns=resolved_exclude_patterns,
            include_external_links=include_external_links,
            include_subdomains=include_subdomains,
            cache_force_refresh=cache_force_refresh,
            apply_patterns=apply_patterns,
        )
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        response = self.session.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        response_payload = response.json()
        job_id = response_payload["result"]

        result: dict[str, Any] | None = None
        for _ in range(self.max_poll_attempts):
            poll_response = self.session.get(
                f"{endpoint}/{job_id}",
                headers={"Authorization": f"Bearer {self.api_token}"},
                params={"limit": 1},
                timeout=30.0,
            )
            poll_response.raise_for_status()
            result = poll_response.json()["result"]
            assert result is not None
            status = result["status"]
            if status == "running":
                if self.poll_interval > 0:
                    time.sleep(self.poll_interval)
                continue
            if status not in _TERMINAL_CLOUDFLARE_STATUSES:
                raise ValueError(f"Unexpected Cloudflare crawl status: {status}")
            if status != "completed":
                raise ValueError(f"Cloudflare crawl ended with status '{status}'")
            break
        else:
            raise TimeoutError("Cloudflare crawl did not complete within the timeout")

        assert result is not None
        full_response = self.session.get(
            f"{endpoint}/{job_id}",
            headers={"Authorization": f"Bearer {self.api_token}"},
            params=None,
            timeout=30.0,
        )
        full_response.raise_for_status()
        result = full_response.json()["result"]
        assert result is not None

        records = list(result.get("records") or [])
        cursor = result.get("cursor")
        while cursor is not None:
            page_response = self.session.get(
                f"{endpoint}/{job_id}",
                headers={"Authorization": f"Bearer {self.api_token}"},
                params={"cursor": cursor, "status": "completed"},
                timeout=30.0,
            )
            page_response.raise_for_status()
            page_result = page_response.json()["result"]
            records.extend(page_result.get("records") or [])
            cursor = page_result.get("cursor")

        scope_origin = _web_origin_key(root)
        root_host = urlparse(root).hostname or ""
        completed_records = []
        for record in records:
            if record.get("status") != "completed":
                continue
            canonical_url = _canonicalize_web_url(record["url"])
            if canonical_url is None:
                continue
            if (
                apply_patterns
                and not _allow_web_origin(
                    canonical_url,
                    scope_origin,
                    root_host,
                    include_external_links=include_external_links,
                    include_subdomains=include_subdomains,
                )
                and not _is_cloudflare_seed_redirect_target(root, canonical_url)
            ):
                continue
            if canonical_url != record["url"]:
                record = dict(record)
                record["url"] = canonical_url
            completed_records.append(record)
        if apply_patterns:
            include_matchers = _compile_pattern_matchers(resolved_include_patterns)
            exclude_matchers = _compile_pattern_matchers(resolved_exclude_patterns)
            completed_records = [
                record
                for record in completed_records
                if _matches_patterns(
                    record["url"],
                    include_matchers=include_matchers,
                    exclude_matchers=exclude_matchers,
                )
            ]
        fetched_at = _utcnow()
        self._roots[cache_key] = _CloudflareRootCacheEntry(
            fetched_at=fetched_at,
            records=completed_records,
        )
        if apply_patterns:
            self._store_root_cache_entry(
                cache_key,
                records=completed_records,
                fetched_at=fetched_at,
            )
        for record in completed_records:
            self._records[record["url"]] = _CloudflareRecordCacheEntry(
                fetched_at=fetched_at,
                record=record,
            )
            self._store_record_cache_entry(
                record["url"],
                record=record,
                fetched_at=fetched_at,
            )
        return completed_records

    def _crawl_payload(
        self,
        root: str,
        *,
        depth: int,
        limit: int | None,
        include_patterns: Sequence[PatternInput],
        exclude_patterns: Sequence[PatternInput],
        include_external_links: bool,
        include_subdomains: bool,
        cache_force_refresh: bool,
        apply_patterns: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "url": root,
            "depth": depth,
            "formats": ["markdown"],
            "render": self.render,
            "source": self.source,
            "options": {
                "includeExternalLinks": include_external_links,
                "includeSubdomains": include_subdomains,
            },
        }
        if limit is not None:
            payload["limit"] = limit
        if apply_patterns:
            # Only glob (str) patterns can be forwarded to the Cloudflare API;
            # pre-compiled regexes are enforced client-side instead.
            include_globs = [p for p in include_patterns if isinstance(p, str)]
            exclude_globs = [p for p in exclude_patterns if isinstance(p, str)]
            include_has_regex = any(isinstance(p, re.Pattern) for p in include_patterns)
            # Excludes only ever remove pages, so forwarding the glob subset is
            # always safe. Includes restrict the result set, so forwarding a
            # glob subset alongside a regex include would wrongly drop the
            # regex matches; omit includePatterns entirely in that case and let
            # the client-side filter do the restriction.
            if include_globs and not include_has_regex:
                payload["options"]["includePatterns"] = include_globs
            if exclude_globs:
                payload["options"]["excludePatterns"] = exclude_globs
        if self.modified_since is not None:
            payload["modifiedSince"] = self.modified_since
        if cache_force_refresh:
            payload["maxAge"] = 0
        elif self.cache_stale_after is not None:
            payload["maxAge"] = int(self.cache_stale_after.total_seconds())
        return payload

    def _record_cache_signature(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "base_url": self.base_url,
            "render": self.render,
            "source": self.source,
            "modified_since": self.modified_since,
        }

    def _root_cache_key(self, cache_key: tuple[Any, ...]) -> str:
        payload = {
            "cache_key": cache_key,
            "signature": self._record_cache_signature(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"cloudflare-root:{encoded}"

    def _load_root_cache_entry(
        self,
        cache_key: tuple[Any, ...],
    ) -> _CloudflareRootCacheEntry | None:
        cached_entry = self._cache.fetch(self._root_cache_key(cache_key))
        if cached_entry is None:
            return None
        _, cached_meta = cached_entry
        if cached_meta is None:
            return None
        if cached_meta.get("signature") != self._record_cache_signature():
            return None
        fetched_at = _parse_datetime(cached_meta.get("fetched_at"))
        if fetched_at is None or not self._cloudflare_cache_is_fresh(fetched_at):
            return None
        records = cached_meta["records"]
        for record in records:
            self._records[record["url"]] = _CloudflareRecordCacheEntry(
                fetched_at=fetched_at,
                record=record,
            )
        return _CloudflareRootCacheEntry(
            fetched_at=fetched_at,
            records=records,
        )

    def _store_root_cache_entry(
        self,
        cache_key: tuple[Any, ...],
        *,
        records: list[dict[str, Any]],
        fetched_at: datetime,
    ) -> None:
        self._cache.upsert(
            self._root_cache_key(cache_key),
            content=None,
            metadata={
                "fetched_at": fetched_at.isoformat(),
                "records": records,
                "signature": self._record_cache_signature(),
            },
            content_ext=None,
        )

    def _load_record_cache_entry(
        self,
        origin: str,
    ) -> _CloudflareRecordCacheEntry | None:
        cached_entry = self._cache.fetch(origin)
        if cached_entry is None:
            return None
        _, cached_meta = cached_entry
        if cached_meta is None:
            return None
        if cached_meta.get("signature") != self._record_cache_signature():
            return None
        fetched_at = _parse_datetime(cached_meta.get("fetched_at"))
        if fetched_at is None or not self._cloudflare_cache_is_fresh(fetched_at):
            return None
        record = cached_meta["record"]
        return _CloudflareRecordCacheEntry(
            fetched_at=fetched_at,
            record=record,
        )

    def _store_record_cache_entry(
        self,
        origin: str,
        *,
        record: dict[str, Any],
        fetched_at: datetime,
    ) -> CacheValue:
        cached_entry = self._cache.upsert(
            origin,
            content=record["markdown"],
            metadata={
                "origin": origin,
                "fetched_at": fetched_at.isoformat(),
                "record": record,
                "signature": self._record_cache_signature(),
            },
            content_ext=".md",
        )
        assert cached_entry is not None
        return cached_entry

    def _cloudflare_cache_is_fresh(self, fetched_at: datetime) -> bool:
        if self.cache_stale_after is None:
            return True
        return _utcnow() - fetched_at <= self.cache_stale_after


def _coerce_roots(roots: RootsInput) -> list[RootInput]:
    if isinstance(roots, (str, Path)):
        return [roots]
    return list(roots)


def _resolve_crawl_scope(scope: CrawlScope) -> _ResolvedCrawlScope:
    include_patterns = _coerce_pattern_sequence(scope.include_patterns)
    exclude_patterns = _coerce_pattern_sequence(scope.exclude_patterns)
    return _ResolvedCrawlScope(
        roots=_coerce_roots(scope.roots),
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        include_matchers=_compile_pattern_matchers(include_patterns),
        exclude_matchers=_compile_pattern_matchers(exclude_patterns),
        depth=_DEFAULT_CRAWL_DEPTH if scope.depth is None else scope.depth,
        limit=scope.limit,
        include_types=_normalize_types(scope.include_types),
        exclude_types=_normalize_types(scope.exclude_types),
        include_external_links=scope.include_external_links,
        include_subdomains=scope.include_subdomains,
    )


def _coerce_pattern_sequence(values: PatternsInput) -> list[PatternInput]:
    if values is None:
        return []
    if isinstance(values, (str, re.Pattern)):
        return [values]
    return list(values)


def _canonicalize_web_url(target: str, *, base: str | None = None) -> str | None:
    url = urljoin(base, target) if base else target
    if not url:
        return None
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme != parsed.scheme:
        parsed = parsed._replace(scheme=scheme)
        url = urlunparse(parsed)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    try:
        parsed.port
    except ValueError:
        return None
    netloc = _canonical_netloc(parsed)
    if netloc != parsed.netloc:
        parsed = parsed._replace(netloc=netloc)
    if parsed.path == "/" and not parsed.params:
        parsed = parsed._replace(path="")
    return urlunparse(parsed)


def _canonical_netloc(parsed: Any) -> str:
    userinfo = ""
    if "@" in parsed.netloc:
        userinfo = f"{parsed.netloc.rsplit('@', 1)[0]}@"
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = parsed.port
    if port is None:
        return f"{userinfo}{host}"
    if parsed.scheme == "http" and port == 80:
        return f"{userinfo}{host}"
    if parsed.scheme == "https" and port == 443:
        return f"{userinfo}{host}"
    return f"{userinfo}{host}:{port}"


def _web_origin_key(origin: str) -> WebOriginKey:
    parsed = urlparse(origin)
    scheme = parsed.scheme.lower()
    port = parsed.port
    if port is None and scheme == "http":
        port = 80
    elif port is None and scheme == "https":
        port = 443
    return scheme, parsed.hostname or "", port


def _allow_web_origin(
    origin: str,
    scope_origin: WebOriginKey,
    root_host: str,
    *,
    include_external_links: bool,
    include_subdomains: bool,
) -> bool:
    parsed = urlparse(origin)
    host = parsed.hostname or ""
    if not host:
        return False
    origin_key = _web_origin_key(origin)
    if origin_key == scope_origin:
        return True
    if include_external_links:
        return True
    if not include_subdomains:
        return False
    return (
        origin_key[0] == scope_origin[0]
        and origin_key[2] == scope_origin[2]
        and host.endswith(f".{root_host}")
    )


def _is_cloudflare_seed_redirect_target(root: str, target: str) -> bool:
    root_parsed = urlparse(root)
    target_parsed = urlparse(target)
    if root_parsed.scheme not in {"http", "https"}:
        return False
    if target_parsed.scheme not in {"http", "https"}:
        return False
    if root_parsed.port is not None or target_parsed.port is not None:
        return False

    root_host = _redirect_host_key(root_parsed.hostname or "")
    target_host = _redirect_host_key(target_parsed.hostname or "")
    return root_host != "" and root_host == target_host


def _redirect_host_key(host: str) -> str:
    host = host.lower()
    if host.startswith("www."):
        return host[4:]
    return host


def _resolve_cache_dir(
    cache_dir: bool | str | Path | None,
    *,
    backend_name: str,
    default_factory: Callable[[], Path | None],
) -> Path | None:
    if cache_dir is None:
        return default_factory()
    if isinstance(cache_dir, bool):
        if cache_dir is True:
            return Path.cwd() / ".raghilda" / "cache" / backend_name
        raise TypeError("cache_dir must be None, True, or a filesystem path")
    return Path(cache_dir).resolve()


def _to_directory_path(root: str | Path) -> Path:
    if isinstance(root, Path):
        return root
    value = str(root)
    if re.match(r"^[A-Za-z]:(?:[\\/]|$)", value):
        return Path(value)
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return _path_from_file_uri(value)
    assert parsed.scheme in {"", "file"}
    return Path(value)


def _iter_directory_files(root: Path, *, max_depth: int) -> Iterator[Path]:
    yield from _iter_directory_files_from(
        root,
        root=root,
        resolved_root=root.resolve(),
        max_depth=max_depth,
    )


def _iter_directory_files_from(
    directory: Path,
    *,
    root: Path,
    resolved_root: Path,
    max_depth: int,
) -> Iterator[Path]:
    for child in sorted(directory.iterdir()):
        if not child.resolve().is_relative_to(resolved_root):
            continue
        if child.is_file():
            yield child
            continue
        if child.is_symlink():
            continue
        if not child.is_dir():
            continue
        child_depth = len(child.relative_to(root).parts) - 1
        if child_depth < max_depth:
            yield from _iter_directory_files_from(
                child,
                root=root,
                resolved_root=resolved_root,
                max_depth=max_depth,
            )


def _path_from_file_uri(origin: str) -> Path:
    parsed = urlparse(origin)
    assert parsed.scheme == "file"
    raw_path = parsed.path
    if parsed.netloc and parsed.netloc != "localhost":
        raw_path = f"//{parsed.netloc}{parsed.path}"
    return Path(url2pathname(raw_path))


def _path_from_file_origin(origin: str) -> Path:
    parsed = urlparse(origin)
    if parsed.scheme == "file":
        return _path_from_file_uri(origin)
    return Path(origin)


def _normalize_types(types: Sequence[str] | None) -> set[str]:
    if types is None:
        return set()
    if isinstance(types, str):
        types = [types]
    return {item.strip().lower() for item in types}


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a Cloudflare-style glob into a compiled regex.

    ``*`` matches any characters except ``/`` and ``**`` matches any
    characters including ``/``. A trailing ``/**`` also matches the bare
    parent (so ``/docs/**`` matches ``/docs`` as well as ``/docs/page``).
    """
    placeholder = "\0"
    regex = re.escape(pattern)
    regex = regex.replace(r"/\*\*", "(?:/.*)?")
    regex = regex.replace(r"\*\*", placeholder)
    regex = regex.replace(r"\*", "[^/]*")
    regex = regex.replace(placeholder, ".*")
    return re.compile(regex)


def _compile_pattern_matchers(
    patterns: Sequence[PatternInput],
) -> list[Callable[[str], bool]]:
    """Compile mixed glob/regex patterns into URL matcher callables.

    ``str`` patterns are treated as globs and matched with ``fullmatch``.
    Pre-compiled ``re.Pattern`` objects are treated as regexes and matched
    with ``search`` (the historical behavior for regex patterns).
    """
    matchers: list[Callable[[str], bool]] = []
    for pattern in patterns:
        if isinstance(pattern, re.Pattern):
            matchers.append(lambda url, p=pattern: p.search(url) is not None)
        else:
            compiled = _glob_to_regex(pattern)
            matchers.append(lambda url, c=compiled: c.fullmatch(url) is not None)
    return matchers


def _pattern_cache_token(pattern: PatternInput) -> str:
    """Return a stable, hashable token identifying a glob or regex pattern."""
    if isinstance(pattern, re.Pattern):
        return f"re:{pattern.flags}:{pattern.pattern}"
    return pattern


def _matches_patterns(
    origin: str,
    *,
    include_matchers: Sequence[Callable[[str], bool]],
    exclude_matchers: Sequence[Callable[[str], bool]],
) -> bool:
    if any(matcher(origin) for matcher in exclude_matchers):
        return False
    if not include_matchers:
        return True
    return any(matcher(origin) for matcher in include_matchers)


def _matches_types(
    label: str | None,
    *,
    include_types: set[str],
    exclude_types: set[str],
) -> bool:
    normalized = label.lower() if label is not None else None
    if normalized is not None and normalized in exclude_types:
        return False
    if not include_types:
        return True
    return normalized in include_types


def _detect_type_label(
    *,
    path: Path | None,
    content_type: str | None,
) -> str | None:
    if path is not None:
        alias = _TYPE_ALIASES.get(path.suffix.lower())
        if alias is not None:
            return alias
    normalized_content_type = _normalize_content_type(content_type)
    if normalized_content_type in _CONTENT_TYPE_LABELS:
        return _CONTENT_TYPE_LABELS[normalized_content_type]
    if path is not None and path.exists() and _MAGIKA is not None:
        result = _MAGIKA.identify_path(path)
        return _MAGIKA_LABELS.get(result.output.label, result.output.label)
    return None


def _normalize_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    return content_type.split(";", 1)[0].strip().lower()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _known_body_suffix(origin: str, *, content_type: str | None) -> str | None:
    normalized = _normalize_content_type(content_type)
    if normalized == "text/html":
        return ".html"
    if normalized == "text/markdown":
        return ".md"
    if normalized == "text/plain":
        return ".txt"
    if normalized in {"application/xml", "text/xml"}:
        return ".xml"
    if normalized == "text/x-python":
        return ".py"
    if normalized == "application/json":
        return ".json"
    if normalized == "application/pdf":
        return ".pdf"
    parsed = urlparse(origin)
    suffix = Path(parsed.path).suffix
    if suffix:
        return suffix
    return None


def _body_suffix(origin: str, *, content_type: str | None) -> str:
    suffix = _known_body_suffix(origin, content_type=content_type)
    if suffix is not None:
        return suffix
    return ".bin"


def _type_hint_path(origin: str, *, content_type: str | None) -> Path:
    suffix = _body_suffix(origin, content_type=content_type)
    return Path("source").with_suffix(suffix)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)
