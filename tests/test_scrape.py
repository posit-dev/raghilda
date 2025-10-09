import textwrap
from pathlib import Path

from ragnar.scrape import find_links


def _write(tmp_path: Path, relative: str, html: str) -> Path:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(html), encoding="utf-8")
    return path


def test_find_links_discovers_relative_and_absolute(tmp_path: Path) -> None:
    index = _write(
        tmp_path,
        "index.html",
        """\
        <html>
          <body>
            <a href="docs/page.html">Docs</a>
            <a href="https://example.com/privacy">Privacy</a>
            <a href="#fragment">Fragment</a>
            <a href="mailto:hello@example.com">Email</a>
          </body>
        </html>
        """,
    )
    _write(
        tmp_path,
        "docs/page.html",
        """\
        <html><body><p>Child page</p></body></html>
        """,
    )

    links = find_links(index)

    local = (tmp_path / "docs/page.html").resolve().as_uri()
    assert local in links
    assert "https://example.com/privacy" in links
    assert not any(link.startswith("mailto:") for link in links)
    assert not any("#" in link for link in links)


def test_find_links_children_only_and_filters(tmp_path: Path) -> None:
    index = _write(
        tmp_path,
        "index.html",
        """\
        <html>
          <body>
            <a href="docs/page.html">Docs</a>
            <a href="https://example.com/privacy">Privacy</a>
            <a href="assets/style.css">CSS</a>
          </body>
        </html>
        """,
    )
    _write(
        tmp_path,
        "docs/page.html",
        """\
        <html><body><p>Child page</p></body></html>
        """,
    )

    links = find_links(
        index,
        children_only=True,
        url_filter=lambda url: url if url.endswith(".html") else False,
    )

    assert links == [(tmp_path / "docs/page.html").resolve().as_uri()]


def test_find_links_depth_and_validate(tmp_path: Path) -> None:
    index = _write(
        tmp_path,
        "index.html",
        """\
        <html>
          <body>
            <a href="docs/page.html">Docs</a>
          </body>
        </html>
        """,
    )
    _write(
        tmp_path,
        "docs/page.html",
        """\
        <html>
          <body>
            <a href="../page2.html">Next</a>
            <a href="../missing.html">Missing</a>
          </body>
        </html>
        """,
    )
    _write(
        tmp_path,
        "page2.html",
        """\
        <html><body><p>Terminal page</p></body></html>
        """,
    )

    links_depth0 = find_links(
        index,
        depth=0,
        children_only=True,
        validate=True,
    )

    links_depth1 = find_links(
        index,
        depth=1,
        children_only=True,
        validate=True,
    )

    page = (tmp_path / "docs/page.html").resolve().as_uri()
    page2 = (tmp_path / "page2.html").resolve().as_uri()
    missing = (tmp_path / "missing.html").resolve().as_uri()

    assert page in links_depth0
    assert page2 not in links_depth0

    assert page in links_depth1
    assert page2 in links_depth1
    assert missing not in links_depth1
