from __future__ import annotations

from collections.abc import Callable, Iterable, Sized
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import threading
from typing import Any, Literal

from tqdm import tqdm

from ._store import BaseStore, WriteResult
from .document import Document


@dataclass
class ItemError(Exception):
    """Failure captured for a single ingest item."""

    item: Any
    error: Exception
    partial_results: IngestResults | None = None

    def __str__(self) -> str:
        return f"Failed to ingest {self.item!r}: {self.error}"


@dataclass
class ItemOutcome:
    """Outcome recorded for a single ingest item."""

    item: Any
    status: Literal["inserted", "replaced", "skipped", "failed", "cancelled"]
    write_result: WriteResult | None = None
    error: ItemError | None = None

    @classmethod
    def from_write_result(cls, item: Any, write_result: WriteResult) -> ItemOutcome:
        return cls(item=item, status=write_result.action, write_result=write_result)


@dataclass
class IngestResults:
    """Itemized results from a shared ingest run."""

    total: int | None = None
    outcomes: list[ItemOutcome] = field(default_factory=list)

    @property
    def inserted(self) -> int:
        return sum(outcome.status == "inserted" for outcome in self.outcomes)

    @property
    def replaced(self) -> int:
        return sum(outcome.status == "replaced" for outcome in self.outcomes)

    @property
    def skipped(self) -> int:
        return sum(outcome.status == "skipped" for outcome in self.outcomes)

    @property
    def cancelled(self) -> int:
        return sum(outcome.status == "cancelled" for outcome in self.outcomes)

    @property
    def errors(self) -> list[ItemError]:
        return [outcome.error for outcome in self.outcomes if outcome.error is not None]

    @property
    def failed(self) -> int:
        return len(self.errors)

    @property
    def write_results(self) -> list[WriteResult]:
        return [
            outcome.write_result
            for outcome in self.outcomes
            if outcome.write_result is not None
        ]

    @property
    def pending(self) -> int | None:
        if self.total is None:
            return None
        return max(self.total - len(self.outcomes), 0)

    def add_outcome(self, outcome: ItemOutcome) -> None:
        self.outcomes.append(outcome)


def _record_future_outcome(
    results: IngestResults, future: Future[ItemOutcome]
) -> ItemError | None:
    outcome = future.result()
    results.add_outcome(outcome)
    return outcome.error


class Ingestor:
    """Configurable shared ingestor for future higher-level orchestrators."""

    def __init__(
        self,
        prepare: Callable[[Any], Document] | None = None,
        num_workers: int = 4,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> None:
        self.prepare = prepare
        self.num_workers = num_workers
        self.on_error: Literal["raise", "skip"] = on_error

    def run(
        self,
        items: Iterable[Any],
        *,
        store: BaseStore,
        progress: bool = True,
    ) -> IngestResults:
        """Run ingestion with this ingestor's configured defaults.

        Examples
        --------
        ```{python}
        from raghilda.ingest import Ingestor

        ingestor = Ingestor(prepare=prepare, num_workers=4, on_error="skip")
        results = ingestor.run(urls, store=store)
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
    on_error: Literal["raise", "skip"] = "raise",
    progress: bool = True,
) -> IngestResults:
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
        `"raise"` to stop submitting new items after the first error, drain
        already-submitted work, and then raise, or `"skip"` to collect
        per-item failures and continue.
    progress
        Whether to render a progress bar.

    Returns
    -------
    IngestResults
        Itemized outcomes plus aggregate inserted, replaced, skipped,
        cancelled, pending, and failed counts.

    Examples
    --------
    ```{python}
    from raghilda.ingest import ingest

    results = ingest(paths, store=store)
    print(results.inserted, results.replaced, results.failed)
    ```
    """
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be one of: 'raise', 'skip'")
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    resolved_prepare = prepare if prepare is not None else store.default_prepare()
    total = len(items) if isinstance(items, Sized) else None
    results = IngestResults(total=total)
    items_iter = iter(items)
    cancel_event = threading.Event()

    def submit_next(
        pool: ThreadPoolExecutor,
        pending: dict[Future[ItemOutcome], Any],
    ) -> bool:
        try:
            item = next(items_iter)
        except StopIteration:
            return False

        pending[pool.submit(do_ingest_work, item)] = item
        return True

    def do_ingest_work(item: Any) -> ItemOutcome:
        try:
            if cancel_event.is_set():
                return ItemOutcome(item=item, status="cancelled")
            document = resolved_prepare(item)
            return ItemOutcome.from_write_result(item, store.upsert(document))
        except Exception as error:
            item_error = (
                error
                if isinstance(error, ItemError)
                else ItemError(item=item, error=error)
            )
            if on_error == "raise":
                cancel_event.set()
            return ItemOutcome(item=item, status="failed", error=item_error)

    fail_fast_error: ItemError | None = None
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pending: dict[Future[ItemOutcome], Any] = {}
        with tqdm(total=total, disable=not progress) as progress_bar:
            for _ in range(num_workers):
                if not submit_next(pool, pending):
                    break

            while pending:
                done, _ = wait(set(pending), return_when=FIRST_COMPLETED)

                for future in done:
                    pending.pop(future)
                    error = _record_future_outcome(results, future)
                    if (
                        error is not None
                        and on_error == "raise"
                        and fail_fast_error is None
                    ):
                        fail_fast_error = error
                        cancel_event.set()
                    progress_bar.update(1)

                if fail_fast_error is None:
                    for _ in done:
                        if not submit_next(pool, pending):
                            break

    if fail_fast_error is not None:
        fail_fast_error.partial_results = results
        raise fail_fast_error

    return results
