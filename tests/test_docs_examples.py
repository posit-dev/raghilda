from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_blocks(path: Path) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    in_block = False

    for line in path.read_text(encoding="utf-8").splitlines():
        if not in_block and line.startswith("```{python}"):
            in_block = True
            current = []
            continue
        if in_block and line.startswith("```"):
            blocks.append("\n".join(current))
            in_block = False
            continue
        if in_block:
            current.append(line)

    return blocks


def _block_containing(path: Path, text: str) -> str:
    for block in _python_blocks(path):
        if text in block:
            return block
    raise AssertionError(f"No Python block containing {text!r} found in {path}")


def test_getting_started_custom_prepare_example_imports_ingest():
    path = ROOT / "user_guide" / "00-getting-started.qmd"
    block = _block_containing(
        path, "result = ingest(links, store=store, prepare=prepare, num_workers=4)"
    )

    assert "from raghilda.ingest import ingest" in block


def test_chunking_custom_prepare_example_imports_ingest():
    path = ROOT / "user_guide" / "02-chunking.qmd"
    block = _block_containing(path, "ingest(files, store=store, prepare=prepare)")

    assert "from raghilda.ingest import ingest" in block
