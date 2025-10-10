import textwrap

from ragnar.read import read_as_markdown


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


def test_read_as_markdown_main_only_is_subset(tmp_path):
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

    main_only = read_as_markdown(path, main_only=True).content
    not_main_only = read_as_markdown(path, main_only=False).content

    assert len(main_only) < len(not_main_only)
    assert "Sidebar content" not in main_only
    assert "Sidebar content" in not_main_only
    assert _strip_title(main_only) in not_main_only


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
