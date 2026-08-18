import textwrap
import threading

from markitdown.converters._markdownify import _CustomMarkdownify

from raghilda.read import _patched_markitdown, read_as_markdown


def _write_html(tmp_path, name, html):
    path = tmp_path / name
    path.write_text(textwrap.dedent(html), encoding="utf-8")
    return str(path)


def _strip_title(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    return "\n".join(lines).strip()


def test_read_as_markdown_extracts_main_and_zaps_nav(tmp_path):
    html = """\
        <!DOCTYPE html>
        <html>
        <head><title>Example</title></head>
        <body>
        <nav>Skip me</nav>
        <main>
          <h1>Main Title</h1>
          <p>Main body</p>
          <pre class="language-python"><code>print("hello")</code></pre>
        </main>
        <aside>Sidebar content</aside>
        </body>
        </html>
    """
    path = _write_html(tmp_path, "doc.html", html)

    result = read_as_markdown(path).content

    assert result.startswith("# Example")
    assert "Main body" in result
    assert "Skip me" not in result
    assert "Sidebar content" not in result
    assert "```language-python" in result


def test_read_as_markdown_extract_selectors_scope_content(tmp_path):
    html = """\
        <!DOCTYPE html>
        <html>
        <head><title>Example</title></head>
        <body>
        <nav>Skip me</nav>
        <main>
          <h1>Main Title</h1>
          <p>Main body</p>
          <pre class="language-r"><code>1 + 1</code></pre>
        </main>
        <aside>Sidebar content</aside>
        </body>
        </html>
    """
    path = _write_html(tmp_path, "doc.html", html)

    main_scope = read_as_markdown(path).content
    full_scope = read_as_markdown(path, html_extract_selectors=[]).content

    assert len(main_scope) < len(full_scope)
    assert "Sidebar content" not in main_scope
    assert "Sidebar content" in full_scope
    assert _strip_title(main_scope) in full_scope


def test_patched_markitdown_serializes_global_converter_patches():
    original_convert_soup = _CustomMarkdownify.convert_soup
    original_convert_pre = getattr(_CustomMarkdownify, "convert_pre")  # noqa: B009
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()

    def hold_first_patch():
        with _patched_markitdown(html_extract_selectors=["main"]):
            first_entered.set()
            assert release_first.wait(timeout=1.0)

    def hold_second_patch():
        assert first_entered.wait(timeout=1.0)
        with _patched_markitdown(html_extract_selectors=[]):
            second_entered.set()
            assert release_second.wait(timeout=1.0)

    first_thread = threading.Thread(target=hold_first_patch)
    second_thread = threading.Thread(target=hold_second_patch)

    try:
        first_thread.start()
        assert first_entered.wait(timeout=1.0)

        second_thread.start()
        assert not second_entered.wait(timeout=0.2)

        release_first.set()
        first_thread.join(timeout=1.0)
        assert not first_thread.is_alive()

        assert second_entered.wait(timeout=1.0)

        release_second.set()
        second_thread.join(timeout=1.0)
        assert not second_thread.is_alive()

        assert _CustomMarkdownify.convert_soup is original_convert_soup
        assert (
            getattr(_CustomMarkdownify, "convert_pre") is original_convert_pre  # noqa: B009
        )
    finally:
        release_first.set()
        release_second.set()
        first_thread.join(timeout=1.0)
        second_thread.join(timeout=1.0)
        _CustomMarkdownify.convert_soup = original_convert_soup
        setattr(_CustomMarkdownify, "convert_pre", original_convert_pre)  # noqa: B010


def test_read_as_markdown_expands_nested_fences(tmp_path):
    html = """\
        <!DOCTYPE html>
        <html>
        <head><title>Example</title></head>
        <body>
        <main>
          <pre><code>```
```{r}
1 + 1
```
```
</code></pre>
        </main>
        </body>
        </html>
    """
    path = _write_html(tmp_path, "nested.html", html)

    result = read_as_markdown(path).content

    assert "````" in result
    assert "```{r}" in result


def test_read_as_markdown_handles_empty_file(tmp_path):
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"\xff\xd8\xff\xd9")

    result = read_as_markdown(str(empty)).content

    assert result == ""
