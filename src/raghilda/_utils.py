"""Internal utilities shared across modules."""

from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait, Future
from typing import Callable, Iterable


def lazy_map(pool: ThreadPoolExecutor, fn: Callable, items: Iterable) -> Iterable[Future]:
    """Like pool.map but doesn't eagerly consume the iterator.

    Yields futures as they complete, maintaining at most max_workers pending.
    Note: unlike map, results may be out of order.

    Parameters
    ----------
    pool
        A ThreadPoolExecutor to submit tasks to.
    fn
        The function to apply to each item.
    items
        An iterable of items to process.

    Yields
    ------
    Future
        Completed futures in the order they finish (not submission order).
    """
    pending: set[Future] = set()
    items_iter = iter(items)

    def submit_next() -> bool:
        """Submit next item if available. Returns True if submitted."""
        try:
            item = next(items_iter)
            pending.add(pool.submit(fn, item))
            return True
        except StopIteration:
            return False

    # Fill initial batch
    for _ in range(pool._max_workers):
        if not submit_next():
            break

    # Process until done
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            submit_next()  # Refill as we complete
            yield future
