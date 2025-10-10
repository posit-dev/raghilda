from __future__ import annotations

from collections import deque
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import urldefrag, urljoin, urlparse, unquote
import xml.etree.ElementTree as ET

import requests

try:  # pragma: no cover - tqdm is optional at runtime
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


class _AnchorParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = set()

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if (name.lower() == "href") and value:
                self.links.add(value.strip())


def _extract_links(txt: str) -> set[str]:
    links = set()
    try:
        parser = _AnchorParser()
        parser.feed(txt)
        links.update(parser.links)
    except Exception:
        pass

    # Now try to parse as a sitemap and get
    try:
        root = ET.fromstring(txt)
        for loc in root.findall(".//{*}url/{*}loc"):
            if loc is not None and loc.text:
                links.update(loc.text.strip())
    except Exception:
        pass

    return links


def find_links(
    x: str | Path | Sequence[str | Path],
    depth: int = 0,
    children_only: bool = False,
    progress: bool = True,
    *,
    url_filter: Callable[[str], [str]] | None = None,
    validate: bool = False,
    **request_kwargs: object,
) -> list[str]:
    """
    Discover hyperlinks starting from one or many documents and return them as URLs.

    Parameters
    ----------
    x
        Starting URL(s). Accepts strings or paths; inputs must expand to HTTP(S)
        URLs.
    depth
        Maximum traversal depth from each starting document. ``0`` inspects the
        starting pages only, ``1`` also inspects their direct children, and so on.
    children_only
        When ``True``, only links that stay under the originating host are
        returned and traversed.
    progress
        Whether to display a progress bar while traversing links. Falls back to
        a no-op when :mod:`tqdm` is not available.
    url_filter
        Receives a list of URL's and decides returns a list of urls that should
        be kept. POssibly smaller.
    validate
        When ``True``, perform a lightweight validation to ensure targets are
        reachable before including them in the results.
    **request_kwargs
        Additional keyword arguments forwarded to :func:`requests.Session.get`
        (and ``head`` during validation) when fetching HTTP resources.

    Yields
    ------
    str
        Absolute link targets, deduplicated and ordered as discovered.
    """
    if isinstance(x, (str, Path)):
        entries: list[str] = [str(x)]
    else:
        entries = [str(item) for item in x]

    if len(entries) < 1:
        return []

    # Queue of url that we are looking for pages
    # queue contains tuples of (url, depth, root_prefix)
    # root_prefix is used when children_only is True
    queue: deque[tuple[str, int, str]] = deque()
    # set of discovered urls
    discovered: set[str] = set()
    # set of visited urls
    visited: set[str] = set()

    # Prepare initial entries
    for entry in entries:
        url = _canonicalize(entry)
        if url is None:
            continue
        parsed = urlparse(url)
        if parsed.scheme == "file":
            prefix = "file://" + str(Path(url).parent)
            prefix = "" if children_only else prefix
        else:
            prefix = url if children_only else ""
            # sitemaps are common, but we don't want them to be part of the prefix.
            prefix = prefix.removesuffix("sitemap.xml")
        queue.append((url, 0, prefix))
        discovered.add(url)

    session = requests.Session()
    pbar = tqdm(disable=not progress)
    while queue:
        url, cur_depth, root_prefix = queue.popleft()

        if url in visited:
            continue

        if children_only and not url.startswith(root_prefix):
            continue

        if validate and not is_valid_uri(url):
            # invalid uris are marked as visited so we don't have to re-check
            visited.add(url)
        else:
            # if not validating or valid uris are marked as discovered.
            discovered.add(url)

        if cur_depth > depth:
            continue

        visited.add(url)

        try:
            parsed = urlparse(url)
            if parsed.scheme == "file":
                with open(parsed.path) as f:
                    text = f.read()
            else:
                response = session.get(url, *request_kwargs)
                response.raise_for_status()
                text = response.text
        except Exception:
            continue

        links = _extract_links(text)

        if url_filter:
            links = url_filter(links)

        # add all links to the queue
        for link in links:
            link = _canonicalize(link, base=url)
            if link is None:
                continue
            if link in visited:
                continue

            queue.append((link, cur_depth + 1, root_prefix))

        pbar.set_description(
            f"URLs discovered {len(discovered)} | Remaining {len(queue)}"
        )
        pbar.update(1)

    return list(discovered)


def _canonicalize(target: str, *, base: str | None = None) -> tuple[str, str] | None:
    """
    Canonicalize a URL by making them absolute, removing fragments, and
    validating that they have a valid scheme and netloc.
    """
    url = urljoin(base, target) if base else target
    if not url:
        return None
    url, _ = urldefrag(url)
    url = unquote(url)
    parsed = urlparse(url)

    # Allow http(s) and file URLs
    if parsed.scheme in {"http", "https"}:
        if not parsed.netloc:
            return None
        return url

    # Handle local file paths
    if parsed.scheme == "file" or not parsed.scheme:
        path = Path(parsed.path)
        if not path.is_absolute() and base:
            path = Path(base).parent / path
        abs_path = path.resolve()
        return abs_path.as_uri()


def is_valid_uri(uri, check_remote=True):
    p = urlparse(uri)
    if not p.scheme:
        return Path(uri).exists()
    if p.scheme in ("file",):
        return Path(p.path).exists()
    if p.scheme in ("http", "https"):
        if not check_remote:
            return True
        try:
            r = requests.head(uri, allow_redirects=True, timeout=5)
            return r.status_code < 400
        except requests.RequestException:
            return False
    return False
