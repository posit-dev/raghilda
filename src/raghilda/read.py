import re
import warnings
from typing import Optional

import requests as _requests

from .document import MarkdownDocument

with warnings.catch_warnings():
    # Ignore: "Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work"
    # that is raised when importing markitdown
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import markitdown

from markitdown.converters._markdownify import _CustomMarkdownify


def read_as_markdown(
    uri: str,
    html_extract_selectors: Optional[list[str]] = None,
    html_zap_selectors: Optional[list[str]] = None,
    *args,
    **kwargs,
) -> MarkdownDocument:
    """
    Read a markdown file from a URI and return its content as a string.

    Parameters
    ----------
    uri
        The URI of the markdown file to read. Supported schemes are:
        - path/to/file.md
        - http://example.com/file.md
        - https://example.com/file.md

    Returns
    -------
    str
        The content of the markdown file as a string.

    html_extract_selectors
        A list of CSS selectors to extract specific parts of the HTML content
        when the URI points to an HTML page. Defaults to ['main'].

    html_zap_selectors
        A list of CSS selectors to remove specific parts of the HTML content
        when the URI points to an HTML page. Defaults to ['nav'].

    Examples
    --------
    ```{python}
    #| eval: false
    from raghilda.read import read_as_markdown

    # Read from a local file
    md_content = read_as_markdown("path/to/file.md")
    print(md_content)

    # Read from an HTTP URL
    md_content = read_as_markdown("https://raw.githubusercontent.com/user/repo/branch/file.md")
    print(md_content)
    ```
    """

    if html_extract_selectors is None:
        html_extract_selectors = ["main"]

    if html_zap_selectors is None:
        html_zap_selectors = ["nav"]

    md = _convert_to_markdown(
        uri,
        html_extract_selectors=html_extract_selectors,
        html_zap_selectors=html_zap_selectors,
        *args,
        **kwargs,
    )

    return MarkdownDocument(origin=uri, content=md)


_session = _requests.Session()
_session.headers.update({"User-Agent": "raghilda"})
md = markitdown.MarkItDown(requests_session=_session)


def _maybe_insert_info_string(text, class_):
    """
    Insert the desired info-string (`class_`) after the first code fence if it
    is not already present.
    """
    if not class_:
        return text
    if isinstance(class_, list):
        try:
            class_ = class_[class_.index("sourceCode") + 1]
        except Exception:
            class_ = " ".join(class_)

    class_ = str(class_).strip()
    if not class_:
        return text

    # find the first code fence
    m = re.match(r"^(\s*)(`{3,})([^\n]*)", text)
    if not m:  # no code fence
        return text

    indent, fence, info = m.groups()
    info_tokens = info.strip().split()
    class_tokens = class_.split()

    # add only the missing tokens from `class_`
    missing = [t for t in class_tokens if t not in info_tokens]
    if not missing:  # already present
        return text

    new_info = " ".join(info_tokens + missing).strip()
    return f"{indent}{fence}{new_info}{text[m.end() :]}"


def _maybe_expand_outer_code_fence(text):
    # take a 'pre' string like this:
    #     ```
    #     ```{r}
    #     foo
    #     ```
    #     ```
    # and converts it to this:
    #     ````
    #     ```{r}
    #     foo
    #     ```
    #     ````
    if text.count("```") > 2:
        new_fence = ""
        for n in range(4, 25):
            new_fence = "`" * n
            if new_fence not in text:
                break
        old_fence = "```"
        old_fence_start = text.find(old_fence)
        old_fence_end = text.rfind(old_fence)
        if (
            old_fence_start != -1
            and old_fence_end != -1
            and old_fence_start != old_fence_end
        ):
            text = "".join(
                [
                    text[:old_fence_start],
                    new_fence,
                    text[old_fence_start + len(old_fence) : old_fence_end],
                    new_fence,
                    text[old_fence_end + len(old_fence) :],
                ]
            )
    return text


class _patched_markitdown:
    def __init__(
        self,
        html_extract_selectors=None,
        html_zap_selectors=None,
    ):
        self.html_extract_selectors = html_extract_selectors or []
        self.html_zap_selectors = html_zap_selectors or []

    def __enter__(self):
        self.og_convert_soup = og_convert_soup = _CustomMarkdownify.convert_soup
        _self = self

        def convert_soup(self, soup):
            for selector in _self.html_extract_selectors:
                if (tag := soup.select_one(selector)) is not None:
                    soup = tag.extract()

            for selector in _self.html_zap_selectors:
                while (tag := soup.select_one(selector)) is not None:
                    tag.decompose()

            return og_convert_soup(self, soup)

        _CustomMarkdownify.convert_soup = convert_soup

        self.og_convert_pre = og_convert_pre = _CustomMarkdownify.convert_pre  # type: ignore[attr-defined]

        def convert_pre(self, el, text, parent_tags):
            class_ = el.get("class", [])
            text = og_convert_pre(self, el, text, parent_tags)
            text = _maybe_expand_outer_code_fence(text)
            text = _maybe_insert_info_string(text, class_)
            return text

        _CustomMarkdownify.convert_pre = convert_pre  # type: ignore[attr-defined]

    def __exit__(self, exc_type, exc_val, exc_tb):
        _CustomMarkdownify.convert_pre = self.og_convert_pre  # type: ignore[attr-defined]
        _CustomMarkdownify.convert_soup = self.og_convert_soup


def _as_str_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _convert_to_markdown(
    x,
    *args,
    html_extract_selectors=None,
    html_zap_selectors=None,
    **kwargs,
):
    html_extract_selectors = _as_str_list(html_extract_selectors)
    html_zap_selectors = _as_str_list(html_zap_selectors)

    with _patched_markitdown(
        html_extract_selectors=html_extract_selectors,
        html_zap_selectors=html_zap_selectors,
    ):
        result = md.convert(x, *args, **kwargs)
        text = result.markdown.strip()

        if result.title is not None:
            title = f"# {result.title}"
            if not text.startswith(title):
                text = f"{title}\n\n{text}"

        text = text.replace("\f", "\n\n---\n\n")

        return text
