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
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TypeVar
import threading
import unicodedata
from urllib.parse import urlparse
from urllib.request import url2pathname

import requests

from .document import MarkdownDocument
from .read import _convert_to_markdown
from .scrape import _canonicalize, _extract_links

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
CacheValue = tuple[Path | None, dict[str, Any] | None]
CacheEntry = tuple[str, Path | None, dict[str, Any] | None]
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


@dataclass(frozen=True)
class CrawlScope:
    roots: RootsInput
    include_patterns: Sequence[str] | None = None
    exclude_patterns: Sequence[str] | None = None
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
    include_patterns: list[str]
    exclude_patterns: list[str]
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
        for path in self.root.glob(f"{base}.*"):
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

        for path in self.root.glob(f"{base}.*"):
            if not path.is_file():
                continue
            if path.name in keep:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _write_content(self, content_path: Path, content: bytes | str | Path) -> None:
        """Write content directly to its destination path."""
        if isinstance(content, bytes):
            with content_path.open("wb") as handle:
                handle.write(content)
            return

        if isinstance(content, str):
            with content_path.open("w", encoding="utf-8") as handle:
                handle.write(content)
            return

        if isinstance(content, Path):
            if content == content_path:
                return
            shutil.copyfile(content, content_path)
            return

        raise TypeError(f"Unsupported content type: {type(content)!r}")

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
    max_workers: int

    @abstractmethod
    def origins(
        self,
        scope: CrawlScope,
        *,
        progress: bool = True,
        cache_force_refresh: bool = False,
    ) -> Iterator[str]:
        pass

    @abstractmethod
    def fetch_raw(
        self,
        origin: str,
        *,
        cache_force_refresh: bool = False,
    ) -> FetchedSource:
        pass

    def fetch_markdown(
        self,
        origin: str,
        *,
        convert: Callable[[FetchedSource], MarkdownDocument] | None = None,
        cache_force_refresh: bool = False,
    ) -> MarkdownDocument:
        source = self.fetch_raw(origin, cache_force_refresh=cache_force_refresh)
        converter = convert or self._default_convert
        return converter(source)

    def markdown_documents(
        self,
        scope: CrawlScope,
        *,
        convert: Callable[[FetchedSource], MarkdownDocument] | None = None,
        progress: bool = True,
        cache_force_refresh: bool = False,
    ) -> Iterator[MarkdownDocument]:
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
                # origins(..., cache_force_refresh=True) already refreshed the source
                # for this crawl, so reuse that cached snapshot here.
                cache_force_refresh=False,
            ),
        )

    def _default_convert(self, source: FetchedSource) -> MarkdownDocument:
        raise NotImplementedError


class DirectoryCrawler(BaseCrawler):
    """Crawl local files and optionally cache converted markdown.

    Directory traversal always reads the current filesystem state. The cache
    stores converted markdown per file origin and is reused only when the
    current file hash and modification time still match the cached metadata.
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
        del progress, cache_force_refresh
        resolved_scope = _resolve_crawl_scope(scope)
        if resolved_scope.limit == 0:
            return
        count = 0
        for root in resolved_scope.roots:
            path = _to_directory_path(root)
            assert path.exists(), f"Root does not exist: {path}"
            if path.is_file():
                origin = path.resolve().as_uri()
                if self._include_path(
                    path,
                    origin,
                    include_patterns=resolved_scope.include_patterns,
                    exclude_patterns=resolved_scope.exclude_patterns,
                    include_types=resolved_scope.include_types,
                    exclude_types=resolved_scope.exclude_types,
                ):
                    yield origin
                    count += 1
                    if (
                        resolved_scope.limit is not None
                        and count >= resolved_scope.limit
                    ):
                        return
                continue
            for file_path in sorted(path.rglob("*")):
                if not file_path.is_file():
                    continue
                relative_depth = len(file_path.relative_to(path).parts) - 1
                if relative_depth > resolved_scope.depth:
                    continue
                origin = file_path.resolve().as_uri()
                if not self._include_path(
                    file_path,
                    origin,
                    include_patterns=resolved_scope.include_patterns,
                    exclude_patterns=resolved_scope.exclude_patterns,
                    include_types=resolved_scope.include_types,
                    exclude_types=resolved_scope.exclude_types,
                ):
                    continue
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
        include_patterns: Sequence[str],
        exclude_patterns: Sequence[str],
        include_types: set[str],
        exclude_types: set[str],
    ) -> bool:
        if not _matches_patterns(
            origin,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        ):
            return False
        label = _detect_type_label(
            path=path, content_type=mimetypes.guess_type(path.name)[0]
        )
        return _matches_types(
            label,
            include_types=include_types,
            exclude_types=exclude_types,
        )


class WebCrawler(BaseCrawler):
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        cache_dir: bool | str | Path | None = None,
        cache_stale_after: timedelta | None = None,
        max_workers: int = 1,
    ) -> None:
        assert max_workers >= 1
        self.session = session or requests.Session()
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
        del progress
        resolved_scope = _resolve_crawl_scope(scope)
        if resolved_scope.limit == 0:
            return
        visited: set[str] = set()
        yielded = 0
        frontier: list[tuple[str, str]] = []

        for root in resolved_scope.roots:
            canonical_root = _canonicalize(str(root))
            assert canonical_root is not None
            parsed = urlparse(canonical_root)
            assert parsed.scheme in {"http", "https"}
            root_host = parsed.hostname or ""
            frontier.append((canonical_root, root_host))

        current_depth = 0
        while frontier:
            batch: list[tuple[str, str]] = []
            for origin, root_host in frontier:
                if origin in visited:
                    continue
                visited.add(origin)
                if not self._allow_origin(
                    origin,
                    root_host,
                    include_external_links=resolved_scope.include_external_links,
                    include_subdomains=resolved_scope.include_subdomains,
                ):
                    continue
                batch.append((origin, root_host))

            next_frontier: list[tuple[str, str]] = []
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
                for (origin, root_host), source in fetched_sources:
                    type_label = (source.metadata or {}).get("type_label")
                    matches_patterns = _matches_patterns(
                        origin,
                        include_patterns=resolved_scope.include_patterns,
                        exclude_patterns=resolved_scope.exclude_patterns,
                    )
                    matches_types = _matches_types(
                        type_label,
                        include_types=resolved_scope.include_types,
                        exclude_types=resolved_scope.exclude_types,
                    )
                    if matches_patterns and matches_types:
                        yield origin
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
                    resolved_host = urlparse(resolved_origin).hostname or root_host
                    for link in sorted(_extract_links(text)):
                        canonical = _canonicalize(link, base=resolved_origin)
                        if canonical is None:
                            continue
                        parsed = urlparse(canonical)
                        if parsed.scheme not in {"http", "https"}:
                            continue
                        next_frontier.append((canonical, resolved_host))
                offset += chunk_size
            frontier = next_frontier
            current_depth += 1

    def fetch_raw(
        self,
        origin: str,
        *,
        cache_force_refresh: bool = False,
    ) -> FetchedSource:
        canonical_origin = _canonicalize(origin)
        assert canonical_origin is not None
        parsed = urlparse(canonical_origin)
        assert parsed.scheme in {"http", "https"}

        cached_entry = self._cache.fetch(canonical_origin)
        body_path: Path | None = None
        cached_meta: dict[str, Any] | None = None
        if cached_entry is not None:
            body_path, cached_meta = cached_entry
        has_cache = body_path is not None and cached_meta is not None
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
        resolved_origin = _canonicalize(response.url) or response.url
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
        return self._source_from_meta(meta, body_path=body_path)

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

    def _allow_origin(
        self,
        origin: str,
        root_host: str,
        *,
        include_external_links: bool,
        include_subdomains: bool,
    ) -> bool:
        host = urlparse(origin).hostname or ""
        if not host:
            return False
        if host == root_host:
            return True
        if include_external_links:
            return True
        if not include_subdomains:
            return False
        return host.endswith(f".{root_host}")


class CloudflareCrawler(BaseCrawler):
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
        del progress
        resolved_scope = _resolve_crawl_scope(scope)
        yielded = 0
        for root in resolved_scope.roots:
            if resolved_scope.limit is not None and yielded >= resolved_scope.limit:
                return
            canonical_root = _canonicalize(str(root))
            assert canonical_root is not None
            remaining = (
                None if resolved_scope.limit is None else resolved_scope.limit - yielded
            )
            records = self._crawl_root(
                canonical_root,
                cache_force_refresh=cache_force_refresh,
                depth=resolved_scope.depth,
                include_patterns=resolved_scope.include_patterns,
                exclude_patterns=resolved_scope.exclude_patterns,
                include_external_links=resolved_scope.include_external_links,
                include_subdomains=resolved_scope.include_subdomains,
                limit=remaining,
            )
            for record in records:
                origin = record["url"]
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
        canonical_origin = _canonicalize(origin)
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
            if record is None:
                raise ValueError(f"Cloudflare crawl did not return record for {origin}")
            record_entry = self._records[canonical_origin]
        else:
            record = record_entry.record

        assert record_entry is not None
        content_path, _ = self._store_record_cache_entry(
            canonical_origin,
            record=record,
            fetched_at=record_entry.fetched_at,
        )
        assert content_path is not None
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
        include_patterns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = None,
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
            tuple(resolved_include_patterns),
            tuple(resolved_exclude_patterns),
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

        completed_records = [
            record for record in records if record.get("status") == "completed"
        ]
        if apply_patterns:
            completed_records = [
                record
                for record in completed_records
                if _matches_cloudflare_patterns(
                    record["url"],
                    include_patterns=resolved_include_patterns,
                    exclude_patterns=resolved_exclude_patterns,
                )
            ]
        fetched_at = _utcnow()
        self._roots[cache_key] = _CloudflareRootCacheEntry(
            fetched_at=fetched_at,
            records=completed_records,
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
        include_patterns: Sequence[str],
        exclude_patterns: Sequence[str],
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
        if apply_patterns and include_patterns:
            payload["options"]["includePatterns"] = list(include_patterns)
        if apply_patterns and exclude_patterns:
            payload["options"]["excludePatterns"] = list(exclude_patterns)
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
        assert record["url"] == origin
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
    return _ResolvedCrawlScope(
        roots=_coerce_roots(scope.roots),
        include_patterns=list(scope.include_patterns or []),
        exclude_patterns=list(scope.exclude_patterns or []),
        depth=_DEFAULT_CRAWL_DEPTH if scope.depth is None else scope.depth,
        limit=scope.limit,
        include_types=_normalize_types(scope.include_types),
        exclude_types=_normalize_types(scope.exclude_types),
        include_external_links=scope.include_external_links,
        include_subdomains=scope.include_subdomains,
    )


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
    return Path(cache_dir)


def _to_directory_path(root: str | Path) -> Path:
    if isinstance(root, Path):
        return root
    parsed = urlparse(str(root))
    if parsed.scheme == "file":
        return _path_from_file_uri(str(root))
    assert parsed.scheme in {"", "file"}
    return Path(str(root))


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
    return {item.strip().lower() for item in types or []}


def _matches_patterns(
    origin: str,
    *,
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
) -> bool:
    for pattern in exclude_patterns:
        if re.search(pattern, origin):
            return False
    if not include_patterns:
        return True
    return any(re.search(pattern, origin) for pattern in include_patterns)


def _matches_cloudflare_patterns(
    origin: str,
    *,
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
) -> bool:
    for pattern in exclude_patterns:
        if _wildcard_matches(origin, pattern):
            return False
    if not include_patterns:
        return True
    return any(_wildcard_matches(origin, pattern) for pattern in include_patterns)


def _wildcard_matches(origin: str, pattern: str) -> bool:
    placeholder = "\0"
    regex = re.escape(pattern)
    regex = regex.replace(r"/\*\*", "(?:/.*)?")
    regex = regex.replace(r"\*\*", placeholder)
    regex = regex.replace(r"\*", "[^/]*")
    regex = regex.replace(placeholder, ".*")
    return re.fullmatch(regex, origin) is not None


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
    parsed = urlparse(origin)
    suffix = Path(parsed.path).suffix
    if suffix:
        return suffix
    normalized = _normalize_content_type(content_type)
    if normalized == "text/html":
        return ".html"
    if normalized == "text/markdown":
        return ".md"
    if normalized == "application/json":
        return ".json"
    if normalized == "application/pdf":
        return ".pdf"
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
