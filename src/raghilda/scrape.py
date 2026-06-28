from __future__ import annotations

from collections import deque
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urldefrag, urljoin, urlparse, unquote
import xml.etree.ElementTree as ET

import requests
from tqdm.auto import tqdm


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
                links.add(loc.text.strip())
    except Exception:
        pass

    return links


def find_links(
    x: str | Path | Sequence[str | Path],
    depth: int = 0,
    children_only: bool = False,
    progress: bool = True,
    *,
    url_filter: Callable[[set[str]], list[str]] | None = None,
    validate: bool = False,
    **request_kwargs: Any,
) -> list[str]:
    """Discover links by crawling one or more starting pages.

    `find_links()` is the simple way to gather a set of URLs to index: give it a
    starting page (or several) and it follows links up to `depth` levels,
    returning the URLs it finds. It reads both HTML pages (following `<a href>`
    links) and XML sitemaps (collecting their `<loc>` entries), which makes it a
    convenient first step before reading and chunking each page with
    `read_as_markdown()`. For larger or repeatable crawls with caching and
    concurrency, see the [`crawl`](crawl.WebCrawler.qmd) module.

    Parameters
    ----------
    x
        Starting URL(s) or path(s): a single string or `Path`, or a sequence of
        them.
    depth
        Maximum traversal depth from each starting page. `0` inspects the starting
        pages only, `1` also follows their direct links, and so on.
    children_only
        When `True`, only links at or below the starting URL are kept and
        followed (for example links under `https://site/docs/` when you start
        there). This is the easiest way to stay within one section of a site.
    progress
        Whether to display a `tqdm` progress bar while crawling.
    url_filter
        Optional callback that receives the set of links found on a page and
        returns the (possibly smaller) list of URLs to keep and follow. Use it to
        apply custom include/exclude rules.
    validate
        When `True`, check that each URL is reachable (with an HTTP `HEAD`
        request) before including it in the results.
    request_kwargs
        Extra keyword arguments forwarded to `requests.Session.get` (and to
        `head` during validation) when fetching pages.

    Returns
    -------
    list[str]
        A deduplicated list of absolute URLs discovered during the crawl.

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.scrape import find_links

    # Discover pages under a documentation section, one level deep
    links = find_links(
        "https://quarto.org/docs/guide/",
        depth=1,
        children_only=True,
    )
    print(f"Found {len(links)} pages")
    ```
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

        if validate and not is_valid_uri(url, **request_kwargs):
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
                response = session.get(url, **request_kwargs)
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


def _canonicalize(target: str, *, base: str | None = None) -> str | None:
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


def is_valid_uri(uri: str, check_remote: bool = True, **request_kwargs: Any) -> bool:
    p = urlparse(uri)
    if not p.scheme:
        return Path(uri).exists()
    if p.scheme in ("file",):
        return Path(p.path).exists()
    if p.scheme in ("http", "https"):
        if not check_remote:
            return True
        try:
            head_kwargs: dict[str, Any] = {
                "allow_redirects": True,
                "timeout": 5,
            }
            head_kwargs.update(request_kwargs)
            r = requests.head(uri, **head_kwargs)
            return r.status_code < 400
        except requests.RequestException:
            return False
    return False
