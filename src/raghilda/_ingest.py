from __future__ import annotations

from collections.abc import Callable, Iterable, Sized
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from tqdm import tqdm

from ._store import BaseStore, WriteResult
from ._utils import lazy_map
from .document import Document

OnError = Literal["raise", "skip"]


@dataclass
class ItemError(Exception):
    """Failure captured for a single ingest item."""

    item: Any
    error: Exception

    def __str__(self) -> str:
        return f"Failed to ingest {self.item!r}: {self.error}"


@dataclass
class IngestResult:
    """Aggregate result from a shared ingest run."""

    inserted: int = 0
    replaced: int = 0
    skipped: int = 0
    errors: list[ItemError] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return len(self.errors)

    def record_write(self, result: WriteResult) -> None:
        if result.action == "inserted":
            self.inserted += 1
        elif result.action == "replaced":
            self.replaced += 1
        else:
            self.skipped += 1


class Ingestor:
    """Configurable shared ingestor for future higher-level orchestrators."""

    def __init__(
        self,
        prepare: Callable[[Any], Document] | None = None,
        num_workers: int = 4,
        on_error: OnError = "raise",
    ) -> None:
        self.prepare = prepare
        self.num_workers = num_workers
        self.on_error: OnError = on_error

    def run(
        self,
        items: Iterable[Any],
        *,
        store: BaseStore,
        progress: bool = True,
    ) -> IngestResult:
        """Run ingestion with this ingestor's configured defaults.

        Examples
        --------
        ```{python}
        from raghilda.ingest import Ingestor

        ingestor = Ingestor(prepare=prepare, num_workers=4, on_error="skip")
        result = ingestor.run(urls, store=store)
        ```
        """
        return ingest(
            items,
            store=store,
            prepare=self.prepare,
            num_workers=self.num_workers,
            on_error=self.on_error,
            progress=progress,
        )


def ingest(
    items: Iterable[Any],
    *,
    store: BaseStore,
    prepare: Callable[[Any], Document] | None = None,
    num_workers: int = 4,
    on_error: OnError = "raise",
    progress: bool = True,
) -> IngestResult:
    """Ingest items into a store with shared orchestration.

    Parameters
    ----------
    items
        Items to ingest. By default these are treated as file paths or URLs.
    store
        Store that will receive prepared documents.
    prepare
        Optional item-to-document function. If omitted, uses `store.default_prepare()`.
    num_workers
        Worker threads used to prepare and upsert items.
    on_error
        `"raise"` to fail fast on the first error, or `"skip"` to collect
        per-item failures and continue.
    progress
        Whether to render a progress bar.

    Returns
    -------
    IngestResult
        Aggregate inserted, replaced, skipped, and failed counts.

    Examples
    --------
    ```{python}
    from raghilda.ingest import ingest

    result = ingest(paths, store=store)
    print(result.inserted, result.replaced, result.failed)
    ```
    """
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be one of: 'raise', 'skip'")
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    resolved_prepare = prepare if prepare is not None else store.default_prepare()
    result = IngestResult()
    total = len(items) if isinstance(items, Sized) else None

    def do_ingest_work(item: Any) -> WriteResult:
        try:
            document = resolved_prepare(item)
            return store.upsert(document)
        except ItemError:
            raise
        except Exception as error:
            raise ItemError(item=item, error=error) from error

    pool = ThreadPoolExecutor(max_workers=num_workers)
    try:
        for future in tqdm(
            lazy_map(pool, do_ingest_work, items), total=total, disable=not progress
        ):
            try:
                result.record_write(future.result())
            except ItemError as error:
                if on_error == "raise":
                    raise error
                result.errors.append(error)
    except Exception:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)

    return result
